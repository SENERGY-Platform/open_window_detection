"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("Operator", )

from datetime import datetime
from shutil import Error
import dotenv
dotenv.load_dotenv()

from operator_lib.util import OperatorBase, logger, InitPhase, todatetime, timestamp_to_str
from operator_lib.util.persistence import save, load
import os
import pandas as pd
from algo import utils
import pickle
import numpy as np

UNUSUAL_FILENAME = "unusual_drop_detections.pickle"
UNUSUAL_2WEEKS_FILENAME = "unusual_2weeks_drop_detections.pickle"
POINTCOUNT_FILENAME = "point_count_2weeks.pickle"
WINDOW_FILENAME = "window_closing_times.pickle"
FIRST_DATA_FILENAME = "first_data_time.pickle"
TOO_FAST_TOO_HIGH_HUMID_FILENAME = "too_fast_too_high_humid.pickle"
LAST_CLOSING_TIME_FILENAME = "last_closing_time.pickle"

from operator_lib.util import Config
class CustomConfig(Config):
    data_path = "/opt/data"
    init_phase_length: float = 2
    init_phase_level: str = "d"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)

        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 2
        
        if self.init_phase_level == '':
            self.init_phase_level = 'd'

class Operator(OperatorBase):
    configType = CustomConfig

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.data_path = self.config.data_path

        self.current_day: datetime = None
        self.day_measurements = []
        self.day_count = 0
        self.day_vals_last2weeks = {
            'mean': [],
            'std': [],
            'point_count': []
        }

        self.mean_2week = None
        self.std_2week = None
        self.mean_count_per_day = None
        self.std_count_per_day = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.too_fast_high_humid_detected = False
        self.too_fast_humid_risings = load(self.config.data_path, TOO_FAST_TOO_HIGH_HUMID_FILENAME, [])

        self.last_closing_time = load(self.config.data_path, LAST_CLOSING_TIME_FILENAME, False)

        self.first_data_time = load(self.config.data_path, FIRST_DATA_FILENAME)

        self.sliding_window_humid = [] # This contains the data from the last hour. Entries of the list are pairs of the form {"timestamp": ts, "value": humidity}
        self.sliding_window_temp = []
        
        self.unsusual_drop_detections = load(self.data_path, UNUSUAL_FILENAME, [])
        self.unsusual_2week_detections = load(self.data_path, UNUSUAL_2WEEKS_FILENAME, [])
        self.dismissed_point_counts = load(self.data_path, POINTCOUNT_FILENAME, [])
 
        self.window_open = False
        self.window_closing_times = load(self.data_path, WINDOW_FILENAME, [])

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)        
        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        value = {
            "window_open": False,
            "timestamp": timestamp_to_str(pd.Timestamp.now())
        }
        self.init_phase_handler.send_first_init_msg(value)    
        
    def stop(self):
        super().stop()
        save(self.data_path, UNUSUAL_FILENAME, self.unsusual_drop_detections)
        save(self.data_path, WINDOW_FILENAME, self.window_closing_times)
        save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
        save(self.data_path, UNUSUAL_2WEEKS_FILENAME, self.unsusual_2week_detections)
        save(self.data_path, POINTCOUNT_FILENAME, self.dismissed_point_counts)
    
    def run(self, data, selector = None, device_id=None):
        current_timestamp = todatetime(data['Humidity_Time'])
        if not self.first_data_time:
            self.first_data_time = current_timestamp
            save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
            self.init_phase_handler = InitPhase(self.config.data_path, self.init_phase_duration, self.first_data_time, self.produce)

        new_humid = float(data['Humidity'])
        new_temp = float(data['Temperature'])
        logger.debug('Humidity: '+str(new_humid)+'  '+'Temperature: '+str(new_temp)+'   '+'Humidity Time: '+ timestamp_to_str(current_timestamp))

        self.update_2week_stats(current_timestamp, new_humid)
        self.sliding_window_humid = utils.update_sliding_window(self.sliding_window_humid, new_humid, current_timestamp)
        self.sliding_window_temp = utils.update_sliding_window(self.sliding_window_temp, new_temp, current_timestamp)
        sampled_sliding_window_humid = utils.minute_resampling(self.sliding_window_humid)
        sampled_sliding_window_temp = utils.minute_resampling(self.sliding_window_temp)
        front_mean_humid, front_std_humid, end_mean_humid = utils.compute_front_end_measures(sampled_sliding_window_humid)
        front_mean_temp, front_std_temp, end_mean_temp = utils.compute_front_end_measures(sampled_sliding_window_temp)

        if (self.unusual_drop_detected(new_humid, current_timestamp, front_mean_humid, front_std_humid, end_mean_humid, sampled_sliding_window_humid) and 
            self.temp_change_detected(front_mean_temp, front_std_temp, end_mean_temp)):
                self.unsusual_drop_detections.append((current_timestamp, new_humid, utils.compute_10min_slope(sampled_sliding_window_humid)))
                save(self.data_path, UNUSUAL_FILENAME, self.unsusual_drop_detections)
                logger.info("Unusual humidity drop!")
                self.window_open = True
        else:
            if self.window_open and self.sliding_window_humid[-1]["value"] - self.unsusual_drop_detections[-1][1] > 1:
                self.window_open = False
                self.window_closing_times.append((current_timestamp, new_humid))
                save(self.data_path, WINDOW_FILENAME, self.window_closing_times)
                logger.info("Window closed!")

        if self.window_open == False and self.last_closing_time != False and current_timestamp - self.last_closing_time <= pd.Timedelta(30, "min"):
            time_window_since_last_closing = [entry["value"] for entry in sampled_sliding_window_humid if entry["timestamp"] >= self.last_closing_time]
            if np.mean(time_window_since_last_closing) >= 65 and self.too_fast_high_humid_detected == False:
                self.too_fast_humid_risings.append(data['Humidity_Time'])
                save(self.data_path, TOO_FAST_TOO_HIGH_HUMID_FILENAME, self.too_fast_humid_risings)
                logger.info("Humidity too fast too high after closing the window!")
                self.too_fast_high_humid_detected = True
        elif self.window_open == False and self.last_closing_time != False and current_timestamp - self.last_closing_time > pd.Timedelta(30, "min"):
            self.too_fast_high_humid_detected = False

        init_value = {
            "window_open": False,
            "timestamp": timestamp_to_str(current_timestamp),
            "humidity_too_fast_too_high": ""
        }
        operator_is_init = self.init_phase_handler.operator_is_in_init_phase(current_timestamp)
        if operator_is_init:
            logger.debug(self.init_phase_handler.generate_init_msg(current_timestamp, init_value))
            return self.init_phase_handler.generate_init_msg(current_timestamp, init_value)

        if self.init_phase_handler.init_phase_needs_to_be_reset():
            logger.debug(self.init_phase_handler.reset_init_phase(init_value))
            return self.init_phase_handler.reset_init_phase(init_value)
        
        if self.too_fast_high_humid_detected:
            logger.debug({"window_open": self.window_open, "timestamp": timestamp_to_str(current_timestamp), "humidity_too_fast_too_high": timestamp_to_str(current_timestamp), 
                    "initial_phase": ""})
            return {"window_open": self.window_open, "timestamp": timestamp_to_str(current_timestamp), "humidity_too_fast_too_high": timestamp_to_str(current_timestamp), 
                    "initial_phase": ""}
        else:
            logger.debug(logger.debug({"window_open": self.window_open, "timestamp": timestamp_to_str(current_timestamp), "humidity_too_fast_too_high": timestamp_to_str(current_timestamp), 
                    "initial_phase": ""}))
            return {"window_open": self.window_open, "timestamp": timestamp_to_str(current_timestamp), "humidity_too_fast_too_high": "", 
                    "initial_phase": ""}

    def unusual_drop_detected(self, current_value, current_timestamp, front_mean, front_std, end_mean, sampled_sliding_window) -> bool:
        if self.is_falling_unusually(front_mean, front_std, end_mean) and self.is_falling_last10min():
            return True
        elif self.mean_2week and self.is_outlier_2week_mean(current_value):
            self.unsusual_2week_detections.append(
                (current_timestamp, current_value, self.mean_2week, self.std_2week, utils.compute_10min_slope(sampled_sliding_window)))
            return True
        return False

    def is_falling_unusually(self, front_mean, front_std, end_mean) -> bool:
        falling = (end_mean < front_mean - front_std and self.sliding_window_humid[-1]["value"] < self.sliding_window_humid[-2]["value"])
        return falling

    def is_falling_last10min(self) -> bool:
        falling = utils.compute_10min_slope(self.sliding_window_humid) < -1
        return falling

    def is_outlier_2week_mean(self, value)-> bool: #change to median
        ratio = 3
        too_low = (value<(self.mean_2week-ratio*self.std_2week))
        too_high = (value>(self.mean_2week+ratio*self.std_2week))

        is_outlier = False
        if too_low and (not utils.is_summer(self.current_day)):
            is_outlier = True
        elif too_high and (utils.is_summer(self.current_day)):
            is_outlier = True
        return is_outlier

    def update_2week_stats(self, current_ts:datetime, value:float)-> (float, float):
        if not self.window_open:
            if not self.current_day: # initial setup
                self.setup_new_day(current_ts, value)
                return self.mean_2week, self.std_2week

            if self.current_day > current_ts: # skip datapoints that arrive not on the same day measured
                return self.mean_2week, self.std_2week

            if self.current_day.date() < current_ts.date():  # a new day puts the old day to statistics
                self.calc_new_stats()
                self.setup_new_day(current_ts, value)
                return self.mean_2week, self.std_2week

            # normal add if no special case
            self.day_measurements.append(value)
            self.day_count += 1
            return self.mean_2week, self.std_2week

    def setup_new_day(self, current_ts, value):
        self.current_day = current_ts
        self.day_measurements = [value]
        self.day_count = 1

    def calc_new_stats(self):
        if self.check_add_day_to_statistics():
            day_avg = np.mean(self.day_measurements)
            day_std = np.std(self.day_measurements)
            self.day_vals_last2weeks['mean'].append(day_avg)
            self.day_vals_last2weeks['std'].append(day_std)
            self.day_vals_last2weeks['point_count'].append(self.day_count)

            if len(self.day_vals_last2weeks['mean'])>=14:
                self.day_vals_last2weeks['mean'] = self.day_vals_last2weeks['mean'][-14:] # if list was full before last item is slided out
                self.day_vals_last2weeks['std'] = self.day_vals_last2weeks['std'][-14:]
                self.day_vals_last2weeks['point_count'] = self.day_vals_last2weeks['point_count'][-14:]

                self.mean_2week = np.mean(self.day_vals_last2weeks['mean'])   # will be calculated once enough days are gathered
                self.std_2week = np.mean(self.day_vals_last2weeks['std'])
                self.mean_count_per_day = np.mean(self.day_vals_last2weeks['point_count'])
                self.std_count_per_day = np.std(self.day_vals_last2weeks['point_count'])


    def check_add_day_to_statistics(self):
        day_count = self.day_count
        mean = self.mean_count_per_day
        std = self.std_count_per_day
        if not mean or not ((day_count < (mean - 2 * std)) or (day_count > (mean + 2 * std))):
            return True
        else:
            self.dismissed_point_counts.append((self.current_day, day_count, mean, std, self.day_vals_last2weeks['point_count']))
            return False
        
    def temp_change_detected(self, front_mean_temp, front_std_temp, end_mean_temp):
        if front_mean_temp-end_mean_temp > 0:
            return True
        else:
            return False

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="open-window-detection-operator", git_info_file='git_commit')
