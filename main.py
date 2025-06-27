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

import dotenv
dotenv.load_dotenv()

from operator_lib.util import OperatorBase, logger, InitPhase, todatetime, timestamp_to_str, Selector
from operator_lib.util.persistence import save, load
import os
import pandas as pd
from numpy import nan
from algo import utils
from algo.moving_mean_feature import MovingMeanFeature
from algo.drops_and_risings_feature import DropsAndRisingsFeature


# general
ALL_DETECTIONS_FILENAME = 'all_detections.pickle'
WINDOW_CLOSING_FILENAME = "window_closing_times.pickle"
FIRST_DATA_FILENAME = "first_data_time.pickle"
LAST_CLOSING_TIME_FILENAME = "last_closing_time.pickle"
# dropsAndRisings feature
DROP_DETECTIONS_FILENAME = "drop_detections.pickle"
DROP_HUMIDITY_REBOUNDS_FILENAME = "drop_humidity_rebounds.pickle"
# movingMean feature
MOV_MEAN_DETECTIONS_FILENAME = "movingMean_detections.pickle"
MOV_MEAN_POINT_COUNT_FILENAME = "movingMean_point_count.pickle"
MOV_MEAN_PENDING_POSITIVE_FILENAME = "movingMean_pending_positive.pickle"
MOV_MEAN_WAITING_FOR_RESET_FILENAME = "movingMean_waiting_for_reset.pickle"

SLIDING_WINDOW_HUMID_FILENAME = "sliding_window_humid.pickle"
SLIDING_WINDOW_TEMP_FILENAME = "sliding_window_temp.pickle"

LAST_TEMP_DROP_TIME_FILE = "last_temp_drop_time.pickle"

from operator_lib.util import Config
class CustomConfig(Config):
    data_path = "/opt/data"
    init_phase_length: float = 2
    init_phase_level: str = "d"
    notification_mode: str = "never"

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

    selectors = [
        Selector({"name": "humidity", "args": ["Humidity", "Humidity_Time"]}),
        Selector({"name": "temperature", "args": ["Temperature", "Temperature_Time"]})
    ]

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.data_path = self.config.data_path
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.unusualDrops_feature = DropsAndRisingsFeature(
            self.data_path,
            DROP_DETECTIONS_FILENAME,
            DROP_HUMIDITY_REBOUNDS_FILENAME,
            LAST_TEMP_DROP_TIME_FILE
        )

        mov_mean_model_parameters = None if 'moving_mean_model_parameters' not in kwargs else kwargs['moving_mean_model_parameters']
        self.movingMean_feature = MovingMeanFeature(
            self.data_path,
            MOV_MEAN_DETECTIONS_FILENAME,
            MOV_MEAN_POINT_COUNT_FILENAME,
            MOV_MEAN_PENDING_POSITIVE_FILENAME,
            MOV_MEAN_WAITING_FOR_RESET_FILENAME,
            model_parameters=mov_mean_model_parameters
        )

        self.first_data_time = load(self.data_path, FIRST_DATA_FILENAME)
        self.last_closing_time = load(self.data_path, LAST_CLOSING_TIME_FILENAME, False)

        self.sliding_window_humid = load(self.data_path, SLIDING_WINDOW_HUMID_FILENAME, []) # This contains the data from the last hour. Entries of the list are pairs of the form {"timestamp": ts, "value": humidity}
        self.sliding_window_temp = load(self.data_path, SLIDING_WINDOW_TEMP_FILENAME, []) # This contains the data from the last hour. Entries of the list are pairs of the form {"timestamp": ts, "value": temperature}
 
        self.window_open = False
        self.open_min = None
        self.window_closing_times = load(self.data_path, WINDOW_CLOSING_FILENAME, [])
        self.detections = load(self.data_path, ALL_DETECTIONS_FILENAME, [])

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)        
        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        value = {
            "window_open": False,
            "timestamp": timestamp_to_str(pd.Timestamp.now())
        }
        self.init_phase_handler.send_first_init_msg(value)    

        self.humid_drop_detected = False
        self.temp_drop_detected = False

        self.notification_mode = self.config.notification_mode
        self.window_state_changed = False
        
    def stop(self):
        super().stop()
        save(self.data_path, ALL_DETECTIONS_FILENAME, self.detections)
        save(self.data_path, WINDOW_CLOSING_FILENAME, self.window_closing_times)
        save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
        save(self.data_path, SLIDING_WINDOW_HUMID_FILENAME, self.sliding_window_humid)
        save(self.data_path, SLIDING_WINDOW_TEMP_FILENAME, self.sliding_window_temp)
        self.movingMean_feature.stop()
        self.unusualDrops_feature.stop()

    def run(self, data, selector = None, device_id=None):
        if selector == "humidity":
            current_humid_timestamp = todatetime(data['Humidity_Time'])
            current_humid_value = float(data['Humidity'])
            logger.debug('Humidity: ' + str(current_humid_value) + '  ' + 'Humidity Time: ' + timestamp_to_str(current_humid_timestamp))

            # init phase
            if not self.first_data_time:
                self.first_data_time = current_humid_timestamp
                save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
                self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
            
            # collect data in init phase
            self.movingMean_feature.update_stats(current_humid_timestamp, current_humid_value, self.window_open)
            self.sliding_window_humid = utils.update_sliding_window(self.sliding_window_humid, current_humid_value, current_humid_timestamp)

            outcome = self.check_for_init_phase(current_humid_timestamp)
            if outcome: return outcome  # init phase cuts of normal analysis

            # normal detection
            sampled_sliding_window_humid = utils.minute_resampling(self.sliding_window_humid)
            slope = utils.compute_n_min_slope(sampled_sliding_window_humid, 10)

            self.humid_drop_detected = False # only set to true if drops feature detected unusual humidity drop

            if (self.window_open and (utils.compute_n_min_slope(sampled_sliding_window_humid, 10) > 0.12
                                      or (self.open_min and (current_humid_value - self.open_min>1.2)))):
                    self.save_closed_window(current_humid_timestamp, current_humid_value)
            else:
                detected, feature = self.detect(current_humid_value, current_humid_timestamp, sampled_sliding_window_humid, selector)
                if detected:
                    if feature == "drops_humid":
                        self.humid_drop_detected = True
                        if current_humid_timestamp - self.unusualDrops_feature.last_temp_drop_time < pd.Timedelta(20,"min"): # open window only then detected if humidity and temperature dropped unusually
                            self.save_detection(current_humid_timestamp, current_humid_value, feature, slope)
                    else:
                        self.save_detection(current_humid_timestamp, current_humid_value, feature, slope)
                else:
                    if self.window_open:
                        self.save_detection(current_humid_timestamp, current_humid_value, 'None', slope) # window is still open but features are not detecting anymore

            self.movingMean_feature.update_reset_functionality(current_humid_timestamp, current_humid_value, sampled_sliding_window_humid)
            humidity_rebound_detected = self.unusualDrops_feature.check_for_fast_risings(
                current_humid_timestamp, sampled_sliding_window_humid, self.window_open, self.last_closing_time)

            return self.build_return_values(current_humid_timestamp, humidity_rebound_detected)
        
        elif selector == "temperature":
            current_temp_timestamp = todatetime(data['Temperature_Time'])
            current_temp_value = float(data['Temperature'])
            logger.debug("Temperature" + ":  " + str(current_temp_value) + '  ' + "Temperature Time: "+ timestamp_to_str(current_temp_timestamp))
            
            # init phase
            if not self.first_data_time:
                self.first_data_time = current_temp_timestamp
                save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
                self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)

            # initialize last_temp_drop_time
            if not self.unusualDrops_feature.last_temp_drop_time:
                self.unusualDrops_feature.last_temp_drop_time = current_temp_timestamp
            
            # collect data in init phase
            self.sliding_window_temp = utils.update_sliding_window(self.sliding_window_temp, current_temp_value, current_temp_timestamp)

            outcome = self.check_for_init_phase(current_temp_timestamp)
            if outcome: return outcome  # init phase cuts of normal analysis

            # normal detection
            sampled_sliding_window_temp = utils.minute_resampling(self.sliding_window_temp)

            self.temp_drop_detected = False # only set to true if drops feature detected unusual temperature drop

            detected, feature = self.detect(current_temp_value, current_temp_timestamp, sampled_sliding_window_temp, selector)
            if detected:
                self.temp_drop_detected = True
                self.unusualDrops_feature.last_temp_drop_time = current_temp_timestamp
                if self.humid_drop_detected:
                    self.save_detection(current_temp_timestamp, current_temp_value, feature, nan)


    def detect(self, current_value, current_timestamp, sampled_sliding_window, selector) -> (bool, str):
        if selector == "humidity":
            if self.unusualDrops_feature.detect(current_timestamp, current_value, self.sliding_window_humid,  sampled_sliding_window,
                                                self.window_open, selector, std_factor=1):
                return True, 'drops_humid'
            elif self.movingMean_feature.detect(current_timestamp, current_value, sampled_sliding_window):
                return True, 'moving_mean'
            return False, ''
        elif selector == "temperature":
            if self.unusualDrops_feature.detect(current_timestamp, current_value, self.sliding_window_temp,  sampled_sliding_window,
                                                self.window_open, selector, std_factor=0.3):
                return True, 'drops_temp'
            return False, ''

    def save_detection(self, current_timestamp, current_value, feature, slope):
        mov_mean_mean = self.movingMean_feature.current_mean
        mov_mean_std = self.movingMean_feature.current_std
        self.open_min = current_value if self.open_min is None else min(self.open_min, current_value)

        self.detections.append((current_timestamp, current_value, feature, slope, mov_mean_mean, mov_mean_std))
        save(self.data_path, ALL_DETECTIONS_FILENAME, self.detections)
        logger.info(f"{current_timestamp}:  Detected an open window!")
        if self.window_open == False:
            self.window_state_changed = True
        self.window_open = True

    def save_closed_window(self, current_timestamp, current_value):
        if self.window_open == True:
            self.window_state_changed = True
        self.window_open = False
        self.open_min = None
        self.last_closing_time=current_timestamp
        self.window_closing_times.append((current_timestamp, current_value))

        save(self.data_path, LAST_CLOSING_TIME_FILENAME, self.last_closing_time)
        save(self.data_path, WINDOW_CLOSING_FILENAME, self.window_closing_times)
        logger.info(f"{current_timestamp}:  Window closed!")

    def check_for_init_phase(self, current_timestamp):
        init_value = {
            "window_open": False,
            "timestamp": timestamp_to_str(current_timestamp),
            "humidity_too_fast_too_high": ""
        }
        if self.init_phase_handler.operator_is_in_init_phase(current_timestamp):
            logger.debug(self.init_phase_handler.generate_init_msg(current_timestamp, init_value))
            return self.init_phase_handler.generate_init_msg(current_timestamp, init_value)

        if self.init_phase_handler.init_phase_needs_to_be_reset():
            logger.debug(self.init_phase_handler.reset_init_phase(init_value))
            return self.init_phase_handler.reset_init_phase(init_value)

        return False

    def build_return_values(self, current_timestamp, humidity_rebound_detected:bool):
        answer = {
            "window_open": self.window_open,
            "timestamp": timestamp_to_str(current_timestamp),  #todo: change return parameter for humidity rebound (including in widgets and operator!!)
            "humidity_too_fast_too_high": timestamp_to_str(current_timestamp) if humidity_rebound_detected else "",
            "initial_phase": "",
            "notification": utils.notify(self.notification_mode, self.window_open, self.window_state_changed)
        }
        logger.debug(answer)
        return answer

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="open-window-detection-operator", git_info_file='git_commit')
