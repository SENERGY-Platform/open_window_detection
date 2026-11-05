import math

import pandas as pd
import numpy as np


# the detection logs are pickled whole on every entry, so they keep only their newest entries
LOG_LIMIT = 1000


def append_capped(log, entry):
    log.append(entry)
    del log[:-LOG_LIMIT]


def parse_reading(raw, low, high):
    """The reading as float, or None if it is missing, not a number or outside [low, high]."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not low <= value <= high:
        return None
    return value


def saturation_vapour_pressure(temperature):
    """hPa over water at temperature in °C, Magnus formula with the WMO coefficients."""
    return 6.112 * math.exp(17.62 * temperature / (243.12 + temperature))


def absolute_humidity(temperature, relative_humidity):
    """g/m³ of water vapour in air of temperature °C and relative_humidity %."""
    return 216.7 * (relative_humidity / 100 * saturation_vapour_pressure(temperature)) / (273.15 + temperature)


def relative_humidity(temperature, absolute_humidity):
    """% that absolute_humidity g/m³ amounts to in air of temperature °C."""
    return 100 * absolute_humidity * (273.15 + temperature) / 216.7 / saturation_vapour_pressure(temperature)


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

def compute_front_end_measures(sliding_window, noise_floor):
    """Splits the window into its last 5 minutes (end) and everything before (front).

    Returns the front's median as the level before a change, robust against a passing spike such as a shower,
    the front's spread around its linear trend as its noise (at least noise_floor, the sensor's resolution), and the
    end's mean. All three are None while the window spans 40 minutes or less.
    """
    first_entry_timestamp = sliding_window[0]["timestamp"]
    last_entry_timestamp = sliding_window[-1]["timestamp"]
    if last_entry_timestamp - first_entry_timestamp <= pd.Timedelta(40, "min"):
        return None, None, None
    front = [entry for entry in sliding_window if last_entry_timestamp - entry["timestamp"] >= pd.Timedelta(5, "min")]
    end = [entry["value"] for entry in sliding_window if last_entry_timestamp - entry["timestamp"] < pd.Timedelta(5, "min")]
    minutes = [(entry["timestamp"] - last_entry_timestamp).total_seconds() / 60 for entry in front]
    values = np.array([entry["value"] for entry in front])
    trend = np.polyval(np.polyfit(minutes, values, 1), minutes)
    noise = max(float(np.std(values - trend)), noise_floor)
    return float(np.median(values)), noise, float(np.mean(end))

def recent_mean(sliding_window, current_timestamp, minutes):
    """Mean of the readings of the last `minutes` minutes before current_timestamp, or None if there are none."""
    values = [entry["value"] for entry in sliding_window if current_timestamp - entry["timestamp"] < pd.Timedelta(minutes, "min")]
    return float(np.mean(values)) if values else None

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
