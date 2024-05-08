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
import pickle

UNUSUAL_FILENAME = "unusual_drop_detections.pickle"
WINDOW_FILENAME = "window_closing_times.pickle"

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

        self.first_data_time = load(self.config.data_path, "first_data_time.pickle")

        self.sliding_window = [] # This contains the data from the last hour. Entries of the list are pairs of the form {"timestamp": ts, "value": humidity}
        self.unsusual_drop_detections = load(self.data_path, UNUSUAL_FILENAME, [])
 
        self.window_open = False
        self.window_closing_times = load(self.data_path, WINDOW_FILENAME, [])

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)        
        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time)
        value = {
            "window_open": False,
            "timestamp": ""
        }
        if self.init_phase_handler.first_init_msg_needs_to_send():
            init_msg = self.init_phase_handler.generate_first_init_msg(value)
            self.produce(init_msg)
        
    def stop(self):
        super().stop()
        save(self.data_path, UNUSUAL_FILENAME, self.unsusual_drop_detections)
        save(self.data_path, WINDOW_FILENAME, self.window_closing_times)
    
    def run(self, data, selector = None, device_id=None):
        current_timestamp = todatetime(data['Humidity_Time'])
        if not self.first_data_time:
            self.first_data_time = current_timestamp
            self.init_phase_handler = InitPhase(self.config.data_path, self.init_phase_duration, self.first_data_time)

        new_value = float(data['Humidity'])
        logger.debug('Humidity: '+str(new_value)+'  '+'Humidity Time: '+ timestamp_to_str(current_timestamp))

        self.sliding_window = utils.update_sliding_window(self.sliding_window, new_value, current_timestamp)
        sampled_sliding_window = utils.minute_resampling(self.sliding_window)
        front_mean, front_std, end_mean = utils.compute_front_end_measures(sampled_sliding_window)
        if end_mean < front_mean - 2*front_std and front_mean - end_mean > 2 and self.sliding_window[-1]["value"] < self.sliding_window[-2]["value"]:
            if (self.window_open == False and utils.compute_10min_slope(sampled_sliding_window) < -1) or self.window_open == True:
                self.unsusual_drop_detections.append((current_timestamp, new_value, utils.compute_10min_slope(sampled_sliding_window)))
                save(self.data_path, UNUSUAL_FILENAME, self.unsusual_drop_detections)
                logger.info("Unusual humidity drop!")
                self.window_open = True
        else:
            if self.window_open and self.sliding_window[-1]["value"] - self.unsusual_drop_detections[-1][1] > 1:
                self.window_open = False
                self.window_closing_times.append((current_timestamp, new_value))
                save(self.data_path, WINDOW_FILENAME, self.window_closing_times)
                logger.info("Window closed!")

        init_value = {
            "window_open": False,
            "timestamp": ""
        }
        operator_is_init = self.init_phase_handler.operator_is_in_init_phase(current_timestamp)
        if operator_is_init:
            return self.init_phase_handler.generate_init_msg(current_timestamp, init_value)

        if self.init_phase_handler.init_phase_needs_to_be_reset():
            return self.init_phase_handler.reset_init_phase(init_value)
        
        return {"window_open": self.window_open, "timestamp": timestamp_to_str(current_timestamp), "initial_phase": ""}
    
from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="open-window-detection-operator", git_info_file='git_commit')
