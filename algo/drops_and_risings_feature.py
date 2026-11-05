from operator_lib.util.persistence import save, load
from operator_lib.util import logger
from algo import utils
import os
import pandas as pd
import numpy as np

# below this the front window's spread is sensor resolution, not noise
AH_NOISE_FLOOR = 0.05  # g/m³


class DropsAndRisingsFeature:
    """Short-term drops of absolute humidity, their confirmation by the temperature, and humidity rebounds after
    closing. Absolute humidity falls within minutes when a window opens, temperature much more slowly, because walls
    and furniture buffer heat far more than moisture; so a humidity drop comes first and the temperature confirms it
    within config.temperature_confirm_minutes. All thresholds are fields of the operator config."""

    def __init__(self, data_path:str, detections_file:str, humidity_spikes_file:str, config):
        self._data_path = f'{data_path}/dropsAndRisings'
        if not os.path.exists(self._data_path):
            os.mkdir(self._data_path)

        self._detections_file = detections_file
        self._humidity_rebounds_file = humidity_spikes_file
        self._config = config

        self._humidity_rebound_detected = False

        self._detections = load(self._data_path, self._detections_file, [])
        self._humidity_rebound_counts = load(self._data_path, self._humidity_rebounds_file, [])

    def stop(self):
        save(self._data_path, self._detections_file, self._detections)
        save(self._data_path, self._humidity_rebounds_file, self._humidity_rebound_counts)

    @property
    def data_path(self):
        return self._data_path

    @property
    def detections(self):
        return self._detections

    @property
    def humidity_rebound_detected(self):
        return self._humidity_rebound_detected


    def detect(self, current_ts, current_value, sliding_window, sampled_sliding_window):
        """Whether absolute humidity drops unusually; only the temperature makes an open window of it."""
        if (self.is_falling_unusually(sliding_window, sampled_sliding_window) and
            (self.is_falling_last10min(sampled_sliding_window) or self.is_falling_extreme_last2min(sliding_window))):
                utils.append_capped(self._detections, (current_ts, current_value))
                save(self._data_path, self._detections_file, self._detections)
                return True
        return False

    def reference_temperature(self, temperature_window, current_ts, last_closing_time):
        """Mean temperature 5 to 15 minutes before current_ts: from before a humidity drop that shows now. Only since
        the last closing, so the room still cold from that airing does not confirm the humidity still low from it."""
        values = [entry["value"] for entry in temperature_window
                  if pd.Timedelta(5, "min") <= current_ts - entry["timestamp"] <= pd.Timedelta(15, "min") and
                  (last_closing_time is False or entry["timestamp"] > last_closing_time)]
        return float(np.mean(values)) if values else None

    def temperature_confirms(self, temperature_window, current_ts, reference_temperature):
        current = utils.recent_mean(temperature_window, current_ts, 3)
        return (reference_temperature is not None and current is not None and
                current <= reference_temperature - self._config.temperature_confirm_drop)

    def temperature_drops_strongly(self, temperature_window, sampled_temperature_window, current_ts, last_closing_time):
        """A drop large enough on its own, for airing in mild weather when absolute humidity hardly changes. Only
        while the temperature still falls, and only since the last closing, so the drop of that airing does not
        open the window again while the room warms up."""
        span = pd.Timedelta(self._config.temperature_strong_drop_minutes, "min")
        recent = [entry["value"] for entry in temperature_window if current_ts - entry["timestamp"] <= span and
                  (last_closing_time is False or entry["timestamp"] > last_closing_time)]
        current = utils.recent_mean(temperature_window, current_ts, 3)
        return (bool(recent) and current is not None and
                max(recent) - current >= self._config.temperature_strong_drop and
                utils.compute_n_min_slope(sampled_temperature_window, 5) < 0)

    def check_for_fast_risings(self, current_timestamp, sampled_sliding_window, window_currently_open, last_closing_time, reference_temperature):
        """Whether humidity came back too high within config.rebound_minutes after closing. Judged on the relative
        humidity the air would have at reference_temperature, the room temperature before opening, so the cold air
        right after closing does not pass for humid air."""
        rebound_window = pd.Timedelta(self._config.rebound_minutes, "min")
        if (window_currently_open == False and last_closing_time != False and
            current_timestamp - last_closing_time <= rebound_window):

            time_window_since_last_closing = [utils.relative_humidity(reference_temperature, entry["value"])
                                              for entry in sampled_sliding_window if entry["timestamp"] >= last_closing_time]
            if (len(time_window_since_last_closing) > 0 and
                    np.mean(time_window_since_last_closing) >= self._config.rebound_humidity and self.humidity_rebound_detected == False):
                utils.append_capped(self._humidity_rebound_counts, current_timestamp)
                save(self.data_path, self._humidity_rebounds_file, self._humidity_rebound_counts)
                logger.info(f"{current_timestamp}:  Humidity too fast too high after closing the window!")
                self._humidity_rebound_detected = True
        elif window_currently_open == False and last_closing_time != False and current_timestamp - last_closing_time > rebound_window:
            self._humidity_rebound_detected = False
        return self._humidity_rebound_detected

    def is_falling_unusually(self, sliding_window, sampled_sliding_window) -> bool:
        front_level, front_noise, end_mean = utils.compute_front_end_measures(sampled_sliding_window, AH_NOISE_FLOOR)
        if front_level is None:
            return False
        drop = front_level - end_mean
        return (drop > self._config.ah_drop_std_factor * front_noise and drop >= self._config.ah_drop_min and
                len(sliding_window) >= 2 and sliding_window[-1]["value"] < sliding_window[-2]["value"])

    def is_falling_last10min(self, sampled_sliding_window) -> bool:
        return utils.compute_n_min_slope(sampled_sliding_window, 10) < -self._config.ah_falling_slope

    def is_falling_extreme_last2min(self, sliding_window) -> bool:
        return len(sliding_window) >= 2 and sliding_window[-2]["value"] - sliding_window[-1]["value"] > self._config.ah_extreme_drop
