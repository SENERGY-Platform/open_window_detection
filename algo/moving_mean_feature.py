from algo import utils
from datetime import datetime, timedelta
from operator_lib.util.persistence import save, load
from operator_lib.util import logger
import numpy as np
import os


class MovingMeanFeature:
    """
    model parameters:
        'init_phase_days'   initial days in which all periods are added to the avg unconditionally if there are enough points
                            periods with missing data are only excluded afterward
                            (increases stability of mean & std for point counts, and prevents unnecessary dropouts)
        'slope_minutes':    minutes for the window length, over which humidity has to be falling to count as open window
                            (prevents reoccurring alarms for ups and downs)
        'std_ratio':        ratio that the standard deviation is multiplied with to check if a point is an outlier
                            (balance between FP and TP)
        'days':             total amount of days for length of the moving avg. Use in combination with hours_per_period.
                            IMPORTANT: amount of days must be evenly(!) divisible by hours_per_period (see example)
                            (provides longer term memory, more days lead to less reactive moving avg)
        'hours_per_period': length of period, after which the mov avg is updated
                            i.e. days=1 and hours_per_period=6 => 24/6 = 4 periods of 6 hours in moving avg
                            (more frequent updates (smaller numbers) enhance adaptability of feature,
                            however to short periods might lead to over adaptation and FP)
        'error_influence':  weight (0 <= error_influence <= 1) with which positives are still contributing to the moving avg
                            error_influence=0 turns off the feature, and excludes positives completely in avg calculation
                            (some integration allows better real time adaptability then none,
                            also effective do integrate possible FP and early stop long FP when Humidity baseline decreases naturally)
    """
    def __init__(self, data_path:str, detections_file:str, dismissed_point_count_file:str, pending_positive_file:str,
                 wait_for_reset_file, model_parameters=None):
        default_parameters = {
            'init_phase_days': 10,
            'slope_minutes': 10,
            'std_ratio': 3,             # good tested values: 3, 4
            'days': 1,                  # good tested values: 1, 2
            'hours_per_period': 6,      # good tested values: 6, 8
            'error_influence': 0.6      # good tested values: 0.5 - 0.75
        }

        model_parameters = default_parameters if model_parameters is None else self.__update_parameters(default_parameters, model_parameters)
        self._model_parameters = model_parameters

        self._data_path = f'{data_path}/movingMean'
        if not os.path.exists(self._data_path):
            os.mkdir(self._data_path)

        self._detections_file = detections_file
        self._dismissed_point_counts_file =  dismissed_point_count_file
        self._pending_positive_file = pending_positive_file
        self._waiting_for_reset_file = wait_for_reset_file

        self._model_parameter_file = 'model_parameters.pickle'
        self._mean_file = "current_mean.pickle"
        self._std_file = "current_std.pickle"
        self._mean_count_per_day_file = "current_mean_count_per_day.pickle"
        self._std_count_per_day_file = "current_std_count_per_day.pickle"

        self._init_phase_days = model_parameters['init_phase_days']


        self._days = model_parameters['days']
        self._hours_per_period = model_parameters['hours_per_period']

        self._error_influence = model_parameters['error_influence']

        self._end_of_init_phase = None
        self._current_period_start = None
        self._period_measurements = []
        self._period_weights = []
        self._period_point_count = 0
        self._period_vals_last_n_days = {
            'mean': [],
            'std': [],
            'point_count': []
        }

        self._std_ratio = model_parameters['std_ratio']
        self._current_mean = None
        self._current_std = None
        self._mean_count_per_period = None
        self._std_count_per_period = None

        # for reset functionality of feature
        self._pending_positive = False
        self._waiting_for_reset = False
        self._slope_min = model_parameters['slope_minutes']

        self._detections = load(self._data_path, self._detections_file, [])
        self._pending_positive_counts = load(self._data_path, self._pending_positive_file, [])
        self._waiting_for_reset_counts = load(self._data_path, self._waiting_for_reset_file, [])
        self._dismissed_point_counts = load(self._data_path, self._dismissed_point_counts_file, [])

        self._current_mean = load(self._data_path, self._mean_file, None)
        self._current_std = load(self._data_path, self._std_file, None)
        self._mean_count_per_period = load(self._data_path, self._mean_count_per_day_file, None)
        self._std_count_per_period = load(self._data_path, self._std_count_per_day_file, None)

        para_string = ', '.join([f'{key} = {value}' for key, value in model_parameters.items()])
        logger.debug('movAvg model initiated with the following parameters: ' + para_string)
        save(self._data_path, self._model_parameter_file, self._model_parameters)


    def stop(self):
        save(self._data_path, self._detections_file, self._detections)
        save(self._data_path, self._dismissed_point_counts_file, self._dismissed_point_counts)
        save(self._data_path, self._pending_positive_file, self._pending_positive_counts)
        save(self._data_path, self._waiting_for_reset_file, self._waiting_for_reset_counts)

    @property
    def data_path(self):
        return self._data_path

    @property
    def model_parameters(self):
        return self.model_parameters

    @property
    def detections(self):
        return self._detections

    @property
    def current_mean(self):
        return self._current_mean

    @property
    def current_std(self):
        return self._current_std

    @property
    def std_ratio(self):
        return self._std_ratio

    @property
    def slope_min(self):
        return self._slope_min

    def detect(self, current_ts, current_value, sampled_sliding_window):
        if self._current_mean and self.__is_outlier(current_value):
            slope = utils.compute_n_min_slope(sampled_sliding_window, self._slope_min)
            if slope<=0 and not self._waiting_for_reset:                        # reset functionality on false Positives
                self._detections.append((current_ts, current_value, self._current_mean, self._current_std, slope))
                save(self._data_path, self._detections_file, self._detections)
                self.activate_pending_positive(current_ts, current_value)
                # print(current_ts, current_value, slope, self._current_mean, self._current_std, self._pending_positive)
                return True
            return False
        return False

    def activate_pending_positive(self, current_ts, current_value):
        if not self._pending_positive:
            self._pending_positive = True
            self._pending_positive_counts.append(
                (current_ts, current_value, self._current_mean, self._current_std, True))
            save(self._data_path, self._pending_positive_file, self._pending_positive_counts)

    def update_stats(self, current_ts:datetime, value:float, window_already_open:bool)-> (float, float):
            if not self._current_period_start:                          # initial setup
                self.__setup_new_period(current_ts, value)
                self._end_of_init_phase = current_ts + timedelta(days=self._init_phase_days)
                return self._current_mean, self._current_std

            if self._current_period_start > current_ts:                 # skip datapoints that arrive not in the same period measured
                return self._current_mean, self._current_std

            if (current_ts - self._current_period_start) >= timedelta(hours=self._hours_per_period):  # a new period puts the old one to statistics
                self.__calc_new_stats(current_ts)
                self.__setup_new_period(current_ts, value)
                return self._current_mean, self._current_std

                                                                        # normal add if no special case
            self._period_point_count += 1
            if not window_already_open:
                self._period_measurements.append(value)
                self._period_weights.append(1)                          # normal points get a full weight in influencing the long therm avg
            else:
                if 0 < self._error_influence <= 1:
                    self._period_measurements.append(value)
                    self._period_weights.append(self._error_influence)  # decreased influence, however some influence in case it is a false positive

            return self._current_mean, self._current_std

    def __update_parameters(self, defaults, updates):
        def log_day_hour_warning(day, hour):
            logger.warn(
                f"Invalid parameter combination found: the amount of hours in 'days' ({day}) must be evenly divisible by 'hours_per_period' ({hour})! "
                f"Continuing with default values: days={defaults['days']}, hours_per_period={defaults['hours_per_period']}")

        final_parameters = defaults.copy()
        for parameter, value in updates.items():
            add = True  # check constrictions

            if (parameter == 'days' and 'hours_per_period' in updates) or (parameter == 'hours_per_period' and 'days' in updates):
                if (updates['days'] * 24) % updates['hours_per_period'] != 0:
                    log_day_hour_warning(updates['days'], updates['hours_per_period'])
                    add = False
            elif (parameter == 'hours_per_period') and ('days' not in updates):
                if (defaults['days'] * 24) % updates['hours_per_period'] != 0:
                    log_day_hour_warning(defaults['days'], updates['hours_per_period'])
                    add = False
            elif (parameter == 'days') and ('hours_per_period' not in updates):
                if (updates['days'] * 24) % defaults['hours_per_period'] != 0:
                    log_day_hour_warning(updates['days'], defaults['hours_per_period'])
                    add = False

            if parameter == 'error_influence' and not (0<=value<=1):
                logger.warn(f"Parameter 'error_influence' ({updates.items['error_influence']}) must be between 0 and 1! "
                        f"Continuing with default values: error_influence={defaults['error_influence']}")
                add = False

            if add:
                final_parameters[parameter] = value
        return final_parameters

    def update_reset_functionality(self, current_ts, current_value, sampled_sliding_window):
        self.__update_reset(current_ts, current_value, sampled_sliding_window)
        self.__update_waiting_for_reset(current_ts, current_value)

    def __update_reset(self, current_ts, current_value, sampled_sliding_window):
        slope = utils.compute_n_min_slope(sampled_sliding_window, self._slope_min)
        if self._pending_positive and slope > 0.05:
            self._waiting_for_reset = True
            self._pending_positive = False

            self._waiting_for_reset_counts.append((current_ts, current_value, self._current_mean, self._current_std, True))
            self._pending_positive_counts.append((current_ts, current_value, self._current_mean, self._current_std, False))

            save(self._data_path, self._pending_positive_file, self._pending_positive_counts)
            save(self._data_path, self._waiting_for_reset_file, self._waiting_for_reset_counts)


    def __update_waiting_for_reset(self, current_ts, current_value) -> None:
        if self._current_mean:
            ratio = self._std_ratio
            lower_thresh = self._current_mean - ratio * self._current_std
            higher_thresh = self._current_mean + ratio * self._current_std
            if (lower_thresh) < current_value < (higher_thresh) and self._waiting_for_reset:
                self._waiting_for_reset = False
                self._waiting_for_reset_counts.append(
                    (current_ts, current_value, self._current_mean, self._current_std, False))
                save(self._data_path, self._waiting_for_reset_file, self._waiting_for_reset_counts)

    def __is_outlier(self, value) -> bool:  # change to median
        ratio = self._std_ratio
        too_low = (value < (self._current_mean - ratio * self._current_std))
        too_high = (value > (self._current_mean + ratio * self._current_std))

        is_outlier = False
        if too_low and (not utils.is_summer(self._current_period_start)):
            is_outlier = True
        elif too_high and (utils.is_summer(self._current_period_start)):
            is_outlier = True
        return is_outlier

    def __setup_new_period(self, current_ts, value):
        self._current_period_start = current_ts
        self._period_measurements = [value]
        self._period_weights = [1]
        self._period_point_count = 1

    def __calc_new_stats(self, current_ts:datetime):
        if self.__check_add_period_to_statistics(current_ts):
            # weighted mean
            period_mean = np.average(self._period_measurements, weights=self._period_weights)

            # weighted standard deviation
            variance = np.average((np.array(self._period_measurements) - period_mean) ** 2, weights=self._period_weights)
            period_std = np.sqrt(variance)

            self._period_vals_last_n_days['mean'].append(period_mean)
            self._period_vals_last_n_days['std'].append(period_std)
            self._period_vals_last_n_days['point_count'].append(self._period_point_count)

            n_periods = int(self._days*(24/self._hours_per_period))
            if len(self._period_vals_last_n_days['mean'])>=n_periods:
                self._period_vals_last_n_days['mean'] = self._period_vals_last_n_days['mean'][-n_periods:]              # if list was full before the oldest item is slided out
                self._period_vals_last_n_days['std'] = self._period_vals_last_n_days['std'][-n_periods:]
                self._period_vals_last_n_days['point_count'] = self._period_vals_last_n_days['point_count'][-n_periods:]

                # will be calculated once enough periods are gathered
                weights = np.arange(1, n_periods + 1, dtype=np.float64) # linear weights for window of periods
                weights /= weights.sum()  # Normalize to sum to 1

                # Weighted mean and standard deviation
                self._current_mean = np.average(self._period_vals_last_n_days['mean'], weights=weights)
                self._current_std = np.average(self._period_vals_last_n_days['std'], weights=weights)
                self._mean_count_per_period = np.mean(self._period_vals_last_n_days['point_count'])
                self._std_count_per_period = np.std(self._period_vals_last_n_days['point_count'])


    def __check_add_period_to_statistics(self, current_ts:datetime) -> bool:
        period_point_count = self._period_point_count
        mean = self._mean_count_per_period
        std = self._std_count_per_period
        # tstd=None if not std else 2*std

        # init phase to stabilize std and mean. Also use to fill window, if window is bigger than amount of init phase
        if not mean or current_ts < self._end_of_init_phase:
            #logger.info(f'{self._current_period_start} - point count for {self._hours_per_period}hour period: {period_point_count}', 'TAKEN INITPHASE', mean, std, tstd, current_ts)
            return True

        # points after init phase should have a minimum amount or for smaller datasets be within the common range
        outlier = (period_point_count < (mean - 3 * std))
        static_min = 70 if self._mean_count_per_period > 100 else 20
        min_per_hour = 5
        min_count_per_period = (min_per_hour*self._hours_per_period)
        if not outlier:
            #logger.info(f'{self._current_period_start} - point count for {self._hours_per_period}hour period:', period_point_count, 'TAKEN NO OUTLIER', mean, std, tstd, min_count_per_period)
            return True
        if period_point_count > min_count_per_period:
            #logger.info(f'{self._current_period_start} - point count for {self._hours_per_period}hour period:', period_point_count, 'TAKEN MIN COUNT', mean, std, tstd, min_count_per_period)
            return True
        if period_point_count > static_min:
            #logger.info(f'{self._current_period_start} - point count for {self._hours_per_period}hour period:', period_point_count, 'TAKEN STATIC MIN', mean, std, tstd, min_count_per_period, static_min)
            return True
        else:
            #logger.info(f'{self._current_period_start} - point count for {self._hours_per_period}hour period:', period_point_count, 'DISMISSED', mean, std, tstd, min_count_per_period)
            self._dismissed_point_counts.append(
                (self._current_period_start, period_point_count, mean, std, self._period_vals_last_n_days['point_count']))
            return False