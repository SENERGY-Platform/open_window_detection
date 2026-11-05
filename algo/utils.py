from datetime import datetime

import pandas as pd
import numpy as np


def todatetime(timestamp):
    if str(timestamp).isdigit():
        if len(str(timestamp))==13:
            return pd.to_datetime(int(timestamp), unit='ms')
        elif len(str(timestamp))==19:
            return pd.to_datetime(int(timestamp), unit='ns')
    else:
        return pd.to_datetime(timestamp)
    
def update_sliding_window(sliding_window, new_value, current_timestamp):
    sliding_window.append({"timestamp": current_timestamp, "value": new_value})
    cut_index = 0
    for i in range(len(sliding_window)-1, -1 , -1):
         if current_timestamp - sliding_window[i]["timestamp"] > pd.Timedelta(1,"h"):
                cut_index = i + 1
                break
    if len(sliding_window[cut_index:]) == 1:
        sliding_window = sliding_window[cut_index-1:]
        return sliding_window
    else:
        sliding_window = sliding_window[cut_index:]
        return sliding_window

def compute_front_end_measures(sliding_window):
    first_entry_timestamp = sliding_window[0]["timestamp"]
    last_entry_timestamp = sliding_window[-1]["timestamp"]
    if last_entry_timestamp - first_entry_timestamp > pd.Timedelta(40, "min"):
        startindex_of_the_end = len(sliding_window)- len([entry["timestamp"] for entry in sliding_window if last_entry_timestamp - entry["timestamp"] < pd.Timedelta(5, "min")])
        front_window = [sliding_window[i]["value"] for i in range(startindex_of_the_end)]
        end_window = [sliding_window[i]["value"] for i in range(startindex_of_the_end, len(sliding_window))]
        front_mean = np.mean(front_window)
        front_std = np.std(front_window)
        end_mean = np.mean(end_window)
        return front_mean, front_std, end_mean
    return 0, 0, 0

def minute_resampling(sliding_window):
    first_entry_timestamp = sliding_window[0]["timestamp"]
    last_entry_timestamp = sliding_window[-1]["timestamp"]
    if last_entry_timestamp - first_entry_timestamp > pd.Timedelta(1, "min"):
        data_series = pd.DataFrame(sliding_window).set_index("timestamp")
        data_series.index = data_series.index.map(lambda x: x.round("S"))
        data_series = data_series[~data_series.index.duplicated(keep='last')]
        resampled_data_series = data_series.resample('s').interpolate().resample('T').asfreq().dropna()
        resampled_sliding_window = resampled_data_series.reset_index().to_dict('records')
        return resampled_sliding_window
    return sliding_window

def compute_n_min_slope(sampled_sliding_window, n:int):
    if len(sampled_sliding_window) < 2:
        return 0
    else:
        last_n_min_window = sampled_sliding_window[-n:] # This simple slicing makes sense here because sampled_sliding_window has a frequency of 1/min
        slope = (last_n_min_window[-1]["value"] - last_n_min_window[0]["value"])/len(last_n_min_window)
        return slope

def is_summer(date:datetime):
    sum_start = 5
    sum_end = 8
    return sum_start<=date.month<=sum_end

def notify(notification_mode: str, window_open: bool, window_state_changed: bool):
    if window_state_changed:
        if notification_mode == "always":
            if window_open:
             return "Window opened."
            else:
                return "Window closed."
        elif notification_mode == "when_opened":
            if window_open:
                return "Window opened."
        elif notification_mode == "when_closed":
            if not window_open:
                return "Window closed."
