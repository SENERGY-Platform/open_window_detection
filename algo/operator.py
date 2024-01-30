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

import util
import os
from algo import utils
import pickle

class Operator(util.OperatorBase):
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.sliding_window = [] # This contains the data from the last hour. Entries of the list are pairs of the form {"timestamp": ts, "value": humidity}
        self.unsusual_drop_detections = []
        self.unsusual_drop_detections_path = f"{data_path}/unusual_drop_detections.pickle"

        if os.path.exists(self.unsusual_drop_detections_path):
            with open(self.unsusual_drop_detections_path, "rb") as f:
                self.unsusual_drop_detections = pickle.load(f)

        self.window_open = False
        self.window_closing_times = []
        self.window_closing_times_path = f"{data_path}/window_closing_times.pickle"     
    
    def run(self, data, selector = None):
        current_timestamp = utils.todatetime(data['Humidity_Time']).tz_localize(None)
        new_value = float(data['Humidity'])
        print('Humidity: '+str(new_value)+'  '+'Humidity Time: '+str(current_timestamp))
        self.sliding_window = utils.update_sliding_window(self.sliding_window, new_value, current_timestamp)
        sampled_sliding_window = utils.minute_resampling(sampled_sliding_window)
        front_mean, front_std, end_mean = utils.compute_front_end_measures(self.sliding_window)
        if end_mean < front_mean - 2*front_std and front_mean - end_mean > 2 and self.sliding_window[-1]["value"] < self.sliding_window[-2]["value"]:
            self.unsusual_drop_detections.append(current_timestamp)
            with open(self.unsusual_drop_detections_path, "wb") as f:
                pickle.dump(self.unsusual_drop_detections, f)
            print("Unusual humidity drop!")
            self.window_open = True
        else:
            if self.window_open:
                self.window_open = False
                self.window_closing_times.append(data['Humidity_Time'])
                with open(self.window_closing_times_path, "wb") as f:
                    pickle.dump(self.window_closing_times, f)
                print("Window closed!")
