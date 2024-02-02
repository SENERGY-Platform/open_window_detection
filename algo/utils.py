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
    first_entry_timestamp = sliding_window[0]["timestamp"]
    if current_timestamp - first_entry_timestamp > pd.Timedelta(3,"h"):
        del sliding_window[0]
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
    if len(sliding_window) > 1:
        data_series = pd.DataFrame(sliding_window).set_index("timestamp")
        data_series = data_series[~data_series.index.duplicated(keep='first')]
        data_series.index = data_series.index.map(lambda x: x.round("S"))
        resampled_data_series = data_series.resample('s').interpolate().resample('T').asfreq().dropna()
        resampled_data_series = resampled_data_series.loc[resampled_data_series.index[-1] - pd.Timedelta(180,'min'):] #!!
        resampled_sliding_window = resampled_data_series.reset_index().to_dict('records')
        return resampled_sliding_window
    return sliding_window