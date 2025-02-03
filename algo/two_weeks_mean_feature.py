from algo import utils
from datetime import datetime
from operator_lib.util.persistence import save, load
import numpy as np
import os


class TwoWeekMeanFeature:
    def __init__(self, data_path:str, detections_file:str, dismissed_point_count_file:str, pending_positive_file:str,
                 wait_for_reset_file, std_ratio=3, slope_minutes=10):
        self._data_path = f'{data_path}/twoWeeksMean'
        if not os.path.exists(self._data_path):
            os.mkdir(self._data_path)

        self._detections_file = detections_file
        self._dismissed_point_counts_file =  dismissed_point_count_file
        self._pending_positive_file = pending_positive_file
        self._waiting_for_reset_file = wait_for_reset_file

        self._mean_2week_file = "current_2week_mean.pickle"
        self._std_2week_file = "current_2week_std.pickle"
        self._mean_count_per_day_file = "current_mean_count_per_day.pickle"
        self._std_count_per_day_file = "current_std_count_per_day.pickle"

        self._current_day = None
        self._day_measurements = []
        self._day_count = 0
        self._day_vals_last2weeks = {
            'mean': [],
            'std': [],
            'point_count': []
        }

        self._ratio_2weeks = std_ratio
        self._mean_2week = None
        self._std_2week = None
        self._mean_count_per_day = None
        self._std_count_per_day = None

        # for reset functionality of feature
        self._pending_positive = False
        self._waiting_for_reset = False
        self._slope_min = slope_minutes

        self._detections = load(self._data_path, self._detections_file, [])
        self._pending_positive_counts = load(self._data_path, self._pending_positive_file, [])
        self._waiting_for_reset_counts = load(self._data_path, self._waiting_for_reset_file, [])
        self._dismissed_point_counts = load(self._data_path, self._dismissed_point_counts_file, [])

        self._mean_2week = load(self._data_path, self._mean_2week_file, None)
        self._std_2week = load(self._data_path, self._std_2week_file, None)
        self._mean_count_per_day = load(self._data_path, self._mean_count_per_day_file, None)
        self._std_count_per_day = load(self._data_path, self._std_count_per_day_file, None)

    def stop(self):
        save(self._data_path, self._detections_file, self._detections)
        save(self._data_path, self._dismissed_point_counts_file, self._dismissed_point_counts)
        save(self._data_path, self._pending_positive_file, self._pending_positive_counts)
        save(self._data_path, self._waiting_for_reset_file, self._waiting_for_reset_counts)

    @property
    def data_path(self):
        return self._data_path

    @property
    def detections(self):
        return self._detections

    @property
    def mean_2weeks(self):
        return self._mean_2week

    @property
    def std_2week(self):
        return self._std_2week

    @property
    def ratio_2weeks(self):
        return self._ratio_2weeks

    @property
    def slope_min(self):
        return self._slope_min

    def detect(self, current_ts, current_value, sampled_sliding_window):
        if self._mean_2week and self.__is_outlier_2week_mean(current_value):
            slope = utils.compute_n_min_slope(sampled_sliding_window, self._slope_min)
            if slope<=0 and not self._waiting_for_reset: # reset functionality on false Positives
                self._detections.append((current_ts, current_value, self._mean_2week, self._std_2week, slope))
                save(self._data_path, self._detections_file, self._detections)

                if not self._pending_positive:
                    self._pending_positive = True
                    self._pending_positive_counts.append((current_ts, current_value, self._mean_2week, self._std_2week, True))
                    save(self._data_path, self._pending_positive_file, self._pending_positive_counts)
                return True
            return False
        return False

    def update_stats(self, current_ts:datetime, value:float, window_already_open:bool)-> (float, float):
        if not window_already_open:
            if not self._current_day: # initial setup
                self.__setup_new_day(current_ts, value)
                return self._mean_2week, self._std_2week

            if self._current_day > current_ts: # skip datapoints that arrive not on the same day measured
                return self._mean_2week, self._std_2week

            if self._current_day.date() < current_ts.date():  # a new day puts the old day to statistics
                self.__calc_new_stats()
                self.__setup_new_day(current_ts, value)
                return self._mean_2week, self._std_2week

            # normal add if no special case
            self._day_measurements.append(value)
            self._day_count += 1
            return self._mean_2week, self._std_2week

    def update_reset_functionality(self, current_ts, current_value, sampled_sliding_window):
        self.__update_reset(current_ts, current_value, sampled_sliding_window)
        self.__update_waiting_for_reset(current_ts, current_value)

    def __update_reset(self, current_ts, current_value, sampled_sliding_window):
        slope = utils.compute_n_min_slope(sampled_sliding_window, self._slope_min)
        if self._pending_positive and slope > 0.12:
            self._waiting_for_reset = True
            self._pending_positive = False

            self._waiting_for_reset_counts.append((current_ts, current_value, self._mean_2week, self._std_2week, True))
            self._pending_positive_counts.append((current_ts, current_value, self._mean_2week, self._std_2week, False))

            save(self._data_path, self._pending_positive_file, self._pending_positive_counts)
            save(self._data_path, self._waiting_for_reset_file, self._waiting_for_reset_counts)

    def __update_waiting_for_reset(self, current_ts, current_value) -> None:
        if self._mean_2week:
            ratio = self._ratio_2weeks
            lower_thresh = self._mean_2week - ratio * self._std_2week
            higher_thresh = self._mean_2week + ratio * self._std_2week
            if lower_thresh <= current_value <= higher_thresh and self._waiting_for_reset:
                self._waiting_for_reset = False
                self._waiting_for_reset_counts.append(
                    (current_ts, current_value, self._mean_2week, self._std_2week, False))
                save(self._data_path, self._waiting_for_reset_file, self._waiting_for_reset_counts)

    def __is_outlier_2week_mean(self, value) -> bool:  # change to median
        ratio = self._ratio_2weeks
        too_low = (value < (self._mean_2week - ratio * self._std_2week))
        too_high = (value > (self._mean_2week + ratio * self._std_2week))

        is_outlier = False
        if too_low and (not utils.is_summer(self._current_day)):
            is_outlier = True
        elif too_high and (utils.is_summer(self._current_day)):
            is_outlier = True
        return is_outlier

    def __setup_new_day(self, current_ts, value):
        self._current_day = current_ts
        self._day_measurements = [value]
        self._day_count = 1

    def __calc_new_stats(self):
        if self.__check_add_day_to_statistics():
            day_avg = np.mean(self._day_measurements)
            day_std = np.std(self._day_measurements)
            self._day_vals_last2weeks['mean'].append(day_avg)
            self._day_vals_last2weeks['std'].append(day_std)
            self._day_vals_last2weeks['point_count'].append(self._day_count)

            if len(self._day_vals_last2weeks['mean'])>=14:
                self._day_vals_last2weeks['mean'] = self._day_vals_last2weeks['mean'][-14:] # if list was full before last item is slided out
                self._day_vals_last2weeks['std'] = self._day_vals_last2weeks['std'][-14:]
                self._day_vals_last2weeks['point_count'] = self._day_vals_last2weeks['point_count'][-14:]

                self._mean_2week = np.mean(self._day_vals_last2weeks['mean'])   # will be calculated once enough days are gathered
                self._std_2week = np.mean(self._day_vals_last2weeks['std'])
                self._mean_count_per_day = np.mean(self._day_vals_last2weeks['point_count'])
                self._std_count_per_day = np.std(self._day_vals_last2weeks['point_count'])


    def __check_add_day_to_statistics(self):
        day_count = self._day_count
        mean = self._mean_count_per_day
        std = self._std_count_per_day
        if not mean or not ((day_count < (mean - 2 * std)) or (day_count > (mean + 2 * std))):
            return True
        else:
            self._dismissed_point_counts.append(
                (self._current_day, day_count, mean, std, self._day_vals_last2weeks['point_count']))
            return False