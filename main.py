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

from operator_lib.util import OperatorBase, logger, InitPhase, todatetime, timestamp_to_str
from operator_lib.util.persistence import save, load
import os
import pandas as pd
from algo import utils
from algo.two_weeks_mean_feature import TwoWeekMeanFeature
from algo.drops_and_risings_feature import DropsAndRisingsFeature


# general
ALL_DETECTIONS_FILENAME = 'all_detections.pickle'
WINDOW_CLOSING_FILENAME = "window_closing_times.pickle"
FIRST_DATA_FILENAME = "first_data_time.pickle"
LAST_CLOSING_TIME_FILENAME = "last_closing_time.pickle"
# dropsAndRisings feature
DROP_DETECTIONS_FILENAME = "drop_detections.pickle"
DROP_HUMIDITY_REBOUNDS_FILENAME = "drop_humidity_rebounds.pickle"
# 2weeksMean feature
TWO_WEEKS_DETECTIONS_FILENAME = "2weeks_detections.pickle"
TWO_WEEKS_POINT_COUNT_FILENAME = "2weeks_point_count.pickle"
TWO_WEEKS_PENDING_POSITIVE_FILENAME = "2weeks_pending_positive.pickle"
TWO_WEEKS_WAITING_FOR_RESET_FILENAME = "2weeks_waiting_for_reset.pickle"

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
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.unusualDrops_feature = DropsAndRisingsFeature(
            self.data_path,
            DROP_DETECTIONS_FILENAME,
            DROP_HUMIDITY_REBOUNDS_FILENAME,
        )

        self.twoWeeksMean_feature = TwoWeekMeanFeature(
            self.data_path,
            TWO_WEEKS_DETECTIONS_FILENAME,
            TWO_WEEKS_POINT_COUNT_FILENAME,
            TWO_WEEKS_PENDING_POSITIVE_FILENAME,
            TWO_WEEKS_WAITING_FOR_RESET_FILENAME,
            slope_minutes=10
        )

        self.first_data_time = load(self.data_path, FIRST_DATA_FILENAME)
        self.last_closing_time = load(self.data_path, LAST_CLOSING_TIME_FILENAME, False)

        self.sliding_window = [] # This contains the data from the last hour. Entries of the list are pairs of the form {"timestamp": ts, "value": humidity}
 
        self.window_open = False
        self.window_closing_times = load(self.data_path, WINDOW_CLOSING_FILENAME, [])
        self.detections = load(self.data_path, ALL_DETECTIONS_FILENAME, [])

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)        
        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        value = {
            "window_open": False,
            "timestamp": timestamp_to_str(pd.Timestamp.now())
        }
        self.init_phase_handler.send_first_init_msg(value)    
        
    def stop(self):
        super().stop()
        save(self.data_path, ALL_DETECTIONS_FILENAME, self.detections)
        save(self.data_path, WINDOW_CLOSING_FILENAME, self.window_closing_times)
        save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
        self.twoWeeksMean_feature.stop()
        self.unusualDrops_feature.stop()

    def run(self, data, selector = None, device_id=None):
        current_timestamp = todatetime(data['Humidity_Time'])
        current_value = float(data['Humidity'])
        logger.debug('Humidity: ' + str(current_value) + '  ' + 'Humidity Time: ' + timestamp_to_str(current_timestamp))

        # init phase
        if not self.first_data_time:
            self.first_data_time = current_timestamp
            save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
            self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)

        outcome = self.check_for_init_phase(current_timestamp)
        if outcome: return outcome  # init phase cuts of normal analysis

        # normal detection
        self.twoWeeksMean_feature.update_stats(current_timestamp, current_value, self.window_open)
        self.sliding_window = utils.update_sliding_window(self.sliding_window, current_value, current_timestamp)
        sampled_sliding_window = utils.minute_resampling(self.sliding_window)
        slope = utils.compute_n_min_slope(sampled_sliding_window, 10)

        if self.window_open and utils.compute_n_min_slope(sampled_sliding_window, 10) > 0.5:
                self.save_closed_window(current_timestamp, current_value)
        else:
            detected, feature = self.detect(current_value, current_timestamp, sampled_sliding_window)
            if detected:
                self.save_detection(current_timestamp, current_value, feature, slope)
            else:
                self.save_detection(current_timestamp, current_value, 'None', slope) # window is still open but features are not detecting anymore

        self.twoWeeksMean_feature.update_reset_functionality(current_timestamp, current_value, sampled_sliding_window)
        humidity_rebound_detected = self.unusualDrops_feature.check_for_fast_risings(
            current_timestamp, sampled_sliding_window, self.window_open, self.last_closing_time)

        return self.build_return_values(current_timestamp, humidity_rebound_detected)


    def detect(self, current_value, current_timestamp, sampled_sliding_window) -> (bool, str):
        if self.unusualDrops_feature.detect(current_timestamp, current_value, self.sliding_window,  sampled_sliding_window,
                                            self.window_open):
            return True, 'drops'
        elif self.twoWeeksMean_feature.detect(current_timestamp, current_value, sampled_sliding_window):
            return True, 'twoWeekMean'
        return False, ''

    def save_detection(self, current_timestamp, current_value, feature, slope):
        two_w_mean = self.twoWeeksMean_feature.mean_2weeks
        two_w_std = self.twoWeeksMean_feature.std_2week
        self.detections.append((current_timestamp, current_value, feature, slope, two_w_mean, two_w_std))
        save(self.data_path, ALL_DETECTIONS_FILENAME, self.detections)
        logger.info("Detected an open window!")
        self.window_open = True

    def save_closed_window(self, current_timestamp, current_value):
        self.window_open = False
        self.last_closing_time=current_timestamp
        self.window_closing_times.append((current_timestamp, current_value))

        save(self.data_path, LAST_CLOSING_TIME_FILENAME, self.last_closing_time)
        save(self.data_path, WINDOW_CLOSING_FILENAME, self.window_closing_times)
        logger.info("Window closed!")

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
            "initial_phase": ""
        }
        logger.debug(answer)
        return answer

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="open-window-detection-operator", git_info_file='git_commit')
