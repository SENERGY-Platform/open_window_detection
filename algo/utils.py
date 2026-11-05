from datetime import datetime
import math

import pandas as pd
import numpy as np


def parse_reading(raw, low, high):
    """The reading as float, or None if it is missing, not a number or outside [low, high]."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not low <= value <= high:
        return None
    return value


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
    """The window linearly interpolated at every full minute between its first and last reading."""
    first_entry_timestamp = sliding_window[0]["timestamp"]
    last_entry_timestamp = sliding_window[-1]["timestamp"]
    if last_entry_timestamp - first_entry_timestamp > pd.Timedelta(1, "min"):
        latest = {}  # a later reading with the same timestamp replaces the earlier one
        for entry in sliding_window:
            latest[pd.Timestamp(entry["timestamp"])] = entry["value"]
        times = sorted(latest)
        seconds = [(t - times[0]).total_seconds() for t in times]
        minutes = pd.date_range(times[0].ceil("min"), times[-1].floor("min"), freq="min")
        values = np.interp([(m - times[0]).total_seconds() for m in minutes], seconds, [latest[t] for t in times])
        return [{"timestamp": m, "value": float(v)} for m, v in zip(minutes, values)]
    return sliding_window

def compute_n_min_slope(sampled_sliding_window, n:int):
    """Least-squares slope per minute over the readings of the last n minutes."""
    if len(sampled_sliding_window) < 2:
        return 0
    last_timestamp = sampled_sliding_window[-1]["timestamp"]
    recent = [entry for entry in sampled_sliding_window if last_timestamp - entry["timestamp"] < pd.Timedelta(n, "min")]
    if len(recent) < 2:
        return 0
    minutes = [(entry["timestamp"] - last_timestamp).total_seconds() / 60 for entry in recent]
    return float(np.polyfit(minutes, [entry["value"] for entry in recent], 1)[0])

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
