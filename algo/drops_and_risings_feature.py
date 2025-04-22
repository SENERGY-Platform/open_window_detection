from operator_lib.util.persistence import save, load
from operator_lib.util import logger
from algo import utils
import os
import pandas as pd
import numpy as np


class DropsAndRisingsFeature:
    def __init__(self, data_path:str, detections_file:str, humidity_spikes_file:str, last_temp_drop_time_file:str):
        self._data_path = f'{data_path}/dropsAndRisings'
        if not os.path.exists(self._data_path):
            os.mkdir(self._data_path)

        self._detections_file = detections_file
        self._humidity_rebounds_file = humidity_spikes_file
        self._last_temp_drop_time_file = last_temp_drop_time_file

        self._humidity_rebound_detected = False

        self._detections = load(self._data_path, self._detections_file, [])
        self._humidity_rebound_counts = load(self._data_path, self._humidity_rebounds_file, [])

        self.last_temp_drop_time = load(self._data_path, self._last_temp_drop_time_file, None)

    def stop(self):
        save(self._data_path, self._detections_file, self._detections)
        save(self._data_path, self._humidity_rebounds_file, self._humidity_rebound_counts)
        save(self._data_path, self._last_temp_drop_time_file, self.last_temp_drop_time)

    @property
    def data_path(self):
        return self._data_path

    @property
    def detections(self):
        return self._detections

    @property
    def humidity_rebound_detected(self):
        return self._humidity_rebound_detected


    def detect(self, current_ts, current_value, sliding_window,  sampled_sliding_window, window_currently_open, selector, std_factor=2):
        if selector == "humidity":
            if ((self.is_falling_unusually(sliding_window, sampled_sliding_window, selector, std_factor)) and
                ((window_currently_open == True) or
                (self.is_falling_last10min(window_currently_open, sampled_sliding_window)) or
                (self.is_falling_extreme_last2min(sliding_window)))):
                    self._detections.append((current_ts, current_value))
                    save(self._data_path, self._detections_file, self._detections)
                    return True
            return False
        elif selector == "temperature":
            if self.is_falling_unusually(sliding_window, sampled_sliding_window, selector, std_factor):
                return True
            return False

    def check_for_fast_risings(self, current_timestamp, sampled_sliding_window, window_currently_open, last_closing_time):
        if (window_currently_open == False and last_closing_time != False and
            current_timestamp - last_closing_time <= pd.Timedelta(30, "min")):

            time_window_since_last_closing = [entry["value"] for entry in sampled_sliding_window if entry["timestamp"] >= last_closing_time]
            if (len(time_window_since_last_closing) > 0 and
                    np.mean(time_window_since_last_closing) >= 65 and self.humidity_rebound_detected == False):
                self._humidity_rebound_counts.append(current_timestamp)
                save(self.data_path, self._humidity_rebounds_file, self._humidity_rebound_counts)
                logger.info(f"{current_timestamp}:  Humidity too fast too high after closing the window!")
                self._humidity_rebound_detected = True
            #2024-12-08T09:33:59.289000Z
        elif window_currently_open == False and last_closing_time != False and current_timestamp - last_closing_time > pd.Timedelta(30, "min"):
            self._humidity_rebound_detected = False
        return self._humidity_rebound_detected

    def is_falling_unusually(self, sliding_window, sampled_sliding_window, selector, std_factor) -> bool:
        front_mean, front_std, end_mean = utils.compute_front_end_measures(sampled_sliding_window)
        if selector == "humidity":
            falling = (end_mean < front_mean - std_factor * front_std and front_mean - end_mean > 2 and
                    sliding_window[-1]["value"] < sliding_window[-2]["value"])
            return falling
        elif selector == "temperature":
            falling = (end_mean < front_mean - std_factor * front_std and sliding_window[-1]["value"] < sliding_window[-2]["value"])
            return falling

    def is_falling_last10min(self, window_currently_open, sampled_sliding_window) -> bool:
        falling = (window_currently_open == False and utils.compute_n_min_slope(sampled_sliding_window, 10) < -.1)
        return falling

    def is_falling_extreme_last2min(self, sliding_window) -> bool:
        falling = len(sliding_window) >= 2 and sliding_window[-2]["value"] - sliding_window[-1]["value"] > 5
        return falling