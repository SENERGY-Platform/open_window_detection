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

import typing

import dotenv
dotenv.load_dotenv()

from operator_lib.util import OperatorBase, logger, InitPhase, timestamp_to_str, Selector
from operator_lib.util.helpers import provide_historic_frames_local
from operator_lib.util.persistence import save, load
import os
import pandas as pd
import datetime
from algo import utils
from algo.moving_mean_feature import MovingMeanFeature
from algo.drops_and_risings_feature import DropsAndRisingsFeature


# general
ALL_DETECTIONS_FILENAME = 'all_detections.pickle'
WINDOW_CLOSING_FILENAME = "window_closing_times.pickle"
FIRST_DATA_FILENAME = "first_data_time.pickle"
LAST_CLOSING_TIME_FILENAME = "last_closing_time.pickle"
WINDOW_STATE_FILENAME = "window_state.pickle"
# dropsAndRisings feature
DROP_DETECTIONS_FILENAME = "drop_detections.pickle"
DROP_HUMIDITY_REBOUNDS_FILENAME = "drop_humidity_rebounds.pickle"
# movingMean feature
MOV_MEAN_DETECTIONS_FILENAME = "movingMean_detections.pickle"
MOV_MEAN_POINT_COUNT_FILENAME = "movingMean_point_count.pickle"
MOV_MEAN_PENDING_POSITIVE_FILENAME = "movingMean_pending_positive.pickle"
MOV_MEAN_WAITING_FOR_RESET_FILENAME = "movingMean_waiting_for_reset.pickle"

# the humidity window holds absolute humidity now; the relative humidity of the former file must not be read as such
SLIDING_WINDOW_AH_FILENAME = "sliding_window_ah.pickle"
SLIDING_WINDOW_TEMP_FILENAME = "sliding_window_temp.pickle"

# readings outside these ranges are sensor faults, not room climate
READING_RANGES = {"Temperature": (-40.0, 85.0), "Humidity": (0.0, 100.0)}

# detection thresholds of CustomConfig, see the README
THRESHOLDS = ("max_temperature_age_minutes", "ah_drop_min", "ah_drop_std_factor", "ah_falling_slope",
              "ah_extreme_drop", "temperature_confirm_drop", "temperature_confirm_minutes", "temperature_strong_drop",
              "temperature_strong_drop_minutes", "close_temperature_rise", "close_temperature_slope", "close_ah_rise",
              "max_open_minutes", "rebound_humidity", "rebound_minutes")

from operator_lib.util import Config
class CustomConfig(Config):
    data_path = "/opt/data"
    init_phase_length: float = 2
    init_phase_level: str = "d"
    notification_mode: str = "never"
    # model parameters of the moving mean feature (see MovingMeanFeature); unset ones keep its defaults
    moving_mean_init_phase_days = None
    moving_mean_slope_minutes = None
    moving_mean_std_ratio = None
    moving_mean_days = None
    moving_mean_hours_per_period = None
    moving_mean_error_influence = None
    moving_mean_reset_slope = None
    # detection thresholds; humidity in absolute humidity (g/m³), temperature in K
    max_temperature_age_minutes = 10.0      # older temperature readings give no absolute humidity
    ah_drop_min = 0.4                       # drop of the last 5 minutes below the hour before
    ah_drop_std_factor = 1.0                # the same drop, in units of the hour's noise
    ah_falling_slope = 0.02                 # per minute over the last 10 minutes ...
    ah_extreme_drop = 0.9                   # ... or between two readings
    temperature_confirm_drop = 0.3          # below the temperature before a humidity drop ...
    temperature_confirm_minutes = 20.0      # ... within this many minutes of it
    temperature_strong_drop = 1.5           # a temperature drop that opens the window on its own ...
    temperature_strong_drop_minutes = 15.0  # ... within this many minutes
    close_temperature_rise = 0.3            # above the lowest temperature while open: the room no longer cools
    close_temperature_slope = 0.05          # per minute over 10 minutes: heating brings the room back
    close_ah_rise = 0.3                     # above the lowest humidity while open
    max_open_minutes = 0.0                  # after this the window counts as closed; 0: no limit
    rebound_humidity = 65.0                 # % relative humidity at the temperature before opening ...
    rebound_minutes = 30.0                  # ... within this many minutes after closing

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)

        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 2

        if self.init_phase_level == '':
            self.init_phase_level = 'd'

        # the flow engine hands every config value over as a string
        for name in THRESHOLDS:
            value = getattr(self, name)
            setattr(self, name, float(value) if value not in (None, '') else getattr(CustomConfig, name))

# a function, not a CustomConfig method: simple_struct copies every public class attribute onto the instance,
# methods included, which leaves them unbound there
def moving_mean_parameters(config):
    types = {'init_phase_days': float, 'slope_minutes': int, 'std_ratio': float, 'days': int,
             'hours_per_period': int, 'error_influence': float, 'reset_slope': float}
    values = {name: getattr(config, f'moving_mean_{name}') for name in types}
    return {name: types[name](value) for name, value in values.items() if value not in (None, '')}

def history_readings(frames):
    """(timestamp, readings) of every row of the history frames, one frame per input topic, in time order. A row
    of a topic that carries both inputs came as one message, and is replayed as one."""
    readings = []
    for frame in frames:
        inputs = [key for key in READING_RANGES if key in frame.columns]
        if "time" not in frame.columns or not inputs:
            continue
        # UTC-naive from the database, UTC-aware from timescale-wrapper
        times = pd.to_datetime(frame["time"], utc=True)
        for timestamp, values in zip(times, frame[inputs].itertuples(index=False)):
            data = {key: value for key, value in zip(inputs, values) if pd.notna(value)}
            if data and pd.notna(timestamp):
                readings.append((timestamp, data))
    readings.sort(key=lambda reading: reading[0])
    return readings

class Operator(OperatorBase):
    """Detects open windows from a room's temperature and relative humidity.

    Relative humidity alone is ambiguous: an open window lowers the water content of the air but also the
    temperature, which raises relative humidity. So detection runs on absolute humidity, computed from both, and
    on the temperature: humidity falls within minutes, the temperature follows more slowly and confirms.
    """
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
            self.config
        )

        self.movingMean_feature = MovingMeanFeature(
            self.data_path,
            MOV_MEAN_DETECTIONS_FILENAME,
            MOV_MEAN_POINT_COUNT_FILENAME,
            MOV_MEAN_PENDING_POSITIVE_FILENAME,
            MOV_MEAN_WAITING_FOR_RESET_FILENAME,
            model_parameters=moving_mean_parameters(self.config) or None
        )

        self.first_data_time = load(self.data_path, FIRST_DATA_FILENAME)
        self.last_closing_time = load(self.data_path, LAST_CLOSING_TIME_FILENAME, False)

        self.sliding_window_ah = load(self.data_path, SLIDING_WINDOW_AH_FILENAME, []) # This contains the data from the last hour. Entries of the list are pairs of the form {"timestamp": ts, "value": absolute humidity}
        self.sliding_window_temp = load(self.data_path, SLIDING_WINDOW_TEMP_FILENAME, []) # This contains the data from the last hour. Entries of the list are pairs of the form {"timestamp": ts, "value": temperature}

        window_state = load(self.data_path, WINDOW_STATE_FILENAME, {})
        self.window_open = window_state.get("window_open", False)
        self.open_since = window_state.get("open_since")
        self.open_min = window_state.get("open_min")  # lowest smoothed absolute humidity while open
        self.open_temperature_min = window_state.get("open_temperature_min")  # lowest smoothed temperature while open
        self.temperature_before_opening = window_state.get("temperature_before_opening")
        self.pending_drop = window_state.get("pending_drop")  # a humidity drop the temperature has not confirmed yet
        self.window_closing_times = load(self.data_path, WINDOW_CLOSING_FILENAME, [])
        self.detections = load(self.data_path, ALL_DETECTIONS_FILENAME, [])

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)
        now = pd.Timestamp.now(tz="UTC")
        # before the first reading the end of the init phase can only be estimated from now;
        # run() replaces the handler once the first reading arrives
        self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time or now, self.produce)

        self.notification_mode = self.config.notification_mode
        self.window_state_changed = False
        self.missing_temperature_reported = False
        self.history_end = {}  # per input, the time of the last reading replayed from the history

        # nothing to continue from: a fresh deployment, or the first start of this version
        if not self.sliding_window_ah and not self.movingMean_feature.started:
            self.replay_history()

        value = {
            "window_open": False,
            "timestamp": timestamp_to_str(now)
        }
        self.init_phase_handler.send_first_init_msg(value)

    def replay_history(self):
        """Runs the Timescale history of the inputs through the detection, at the times Timescale stored them, so
        the operator starts with the windows, the baseline and the init phase it would have had running that long.
        It reads the init phase and the hour of the short-term windows on top, so that the init phase is over at the
        start if the history is complete. The outputs of the replay are not sent, nor the notifications they carry."""
        duration = self.init_phase_duration + pd.Timedelta(1, "h")
        try:
            readings = history_readings(provide_historic_frames_local(duration.to_pytimedelta()))
        except Exception as ex:  # no Timescale configured, an input only Kafka has, or the database is unreachable
            logger.warning(f"starting without history: {type(ex).__name__}: {ex}")
            return
        if not readings:
            logger.info("starting without history: there is none")
            return

        self.sliding_window_temp = []  # left by a former version, it ends after the history starts
        for timestamp, data in readings:
            self.run(data, selector=None, device_id=None, timestamp=timestamp)
            for key in data:
                self.history_end[key] = timestamp
        self.save_state()
        logger.info(f"replayed {len(readings)} readings from {readings[0][0]} to {readings[-1][0]}")

    def stop(self):
        super().stop()
        self.save_state()

    def save_state(self):
        save(self.data_path, ALL_DETECTIONS_FILENAME, self.detections)
        save(self.data_path, WINDOW_CLOSING_FILENAME, self.window_closing_times)
        save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
        save(self.data_path, SLIDING_WINDOW_AH_FILENAME, self.sliding_window_ah)
        save(self.data_path, SLIDING_WINDOW_TEMP_FILENAME, self.sliding_window_temp)
        self.save_window_state()
        self.movingMean_feature.stop()
        self.unusualDrops_feature.stop()

    def run(self, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        current_timestamp = pd.Timestamp(timestamp)
        data = self.drop_replayed_readings(self.drop_invalid_readings(data), current_timestamp)
        if not data:
            return None

        outcome = None
        if "Temperature" in data.keys():
            outcome = self.handle_temperature(current_timestamp, data['Temperature'])
        if "Humidity" in data.keys():
            outcome = self.handle_humidity(current_timestamp, data['Humidity'])
        return outcome

    def handle_temperature(self, current_timestamp, temperature):
        logger.debug("Temperature" + ":  " + str(temperature) + '  ' + "Temperature Time: "+ timestamp_to_str(current_timestamp))
        self.record_first_data_time(current_timestamp)
        self.sliding_window_temp = utils.update_sliding_window(self.sliding_window_temp, temperature, current_timestamp)
        # the window state is decided with each humidity reading, which also carries the output
        return self.check_for_init_phase(current_timestamp) or None

    def handle_humidity(self, current_timestamp, humidity):
        logger.debug('Humidity: ' + str(humidity) + '  ' + 'Humidity Time: ' + timestamp_to_str(current_timestamp))
        self.record_first_data_time(current_timestamp)

        temperature = self.latest_temperature(current_timestamp)
        if temperature is None:
            if not self.missing_temperature_reported:
                logger.warning(f"no temperature reading in the last {self.config.max_temperature_age_minutes:g} minutes: "
                               "without it there is no absolute humidity and no detection")
                self.missing_temperature_reported = True
            return self.check_for_init_phase(current_timestamp) or self.build_return_values(current_timestamp, False)
        self.missing_temperature_reported = False
        absolute_humidity = utils.absolute_humidity(temperature, humidity)

        # collect data in init phase
        self.movingMean_feature.update_stats(current_timestamp, absolute_humidity, self.window_open)
        self.sliding_window_ah = utils.update_sliding_window(self.sliding_window_ah, absolute_humidity, current_timestamp)

        outcome = self.check_for_init_phase(current_timestamp)
        if outcome: return outcome  # init phase cuts of normal analysis

        # normal detection
        sampled_sliding_window_ah = utils.minute_resampling(self.sliding_window_ah)
        sampled_sliding_window_temp = utils.minute_resampling(self.sliding_window_temp)

        if self.window_open:
            if self.is_closed(current_timestamp, sampled_sliding_window_temp):
                self.save_closed_window(current_timestamp, absolute_humidity)
        else:
            feature = self.detect(current_timestamp, absolute_humidity, sampled_sliding_window_ah, sampled_sliding_window_temp)
            if feature:
                self.save_detection(current_timestamp, absolute_humidity, feature, utils.compute_n_min_slope(sampled_sliding_window_ah, 10))

        self.movingMean_feature.update_reset_functionality(current_timestamp, absolute_humidity, sampled_sliding_window_ah)
        humidity_rebound_detected = self.unusualDrops_feature.check_for_fast_risings(
            current_timestamp, sampled_sliding_window_ah, self.window_open, self.last_closing_time,
            self.temperature_before_opening or temperature)

        return self.build_return_values(current_timestamp, humidity_rebound_detected)

    def drop_invalid_readings(self, data):
        readings = {}
        for key, (low, high) in READING_RANGES.items():
            if key in data:
                value = utils.parse_reading(data[key], low, high)
                if value is None:
                    logger.warning(f"ignoring {key} reading {data[key]!r}: not a number between {low} and {high}")
                else:
                    readings[key] = value
        return readings

    def drop_replayed_readings(self, data, current_timestamp):
        """Kafka starts again at the offset committed last, which can lie within the history replayed at start."""
        return {key: value for key, value in data.items()
                if key not in self.history_end or current_timestamp > self.history_end[key]}

    def record_first_data_time(self, current_timestamp):
        if not self.first_data_time:
            self.first_data_time = current_timestamp
            save(self.data_path, FIRST_DATA_FILENAME, self.first_data_time)
            self.init_phase_handler = InitPhase(self.data_path, self.init_phase_duration, self.first_data_time, self.produce)

    def latest_temperature(self, current_timestamp):
        if not self.sliding_window_temp:
            return None
        latest = self.sliding_window_temp[-1]
        if abs(current_timestamp - latest["timestamp"]) > pd.Timedelta(self.config.max_temperature_age_minutes, "min"):
            return None
        return latest["value"]

    def detect(self, current_timestamp, absolute_humidity, sampled_sliding_window_ah, sampled_sliding_window_temp) -> typing.Optional[str]:
        """The feature that detects an open window now, or None."""
        drops = self.unusualDrops_feature
        if (self.pending_drop and current_timestamp - self.pending_drop["timestamp"] >
                pd.Timedelta(self.config.temperature_confirm_minutes, "min")):
            self.set_pending_drop(None)  # the temperature never followed
        if not self.pending_drop and drops.detect(current_timestamp, absolute_humidity, self.sliding_window_ah, sampled_sliding_window_ah):
            reference = drops.reference_temperature(self.sliding_window_temp, current_timestamp, self.last_closing_time)
            if reference is not None:  # none yet within 5 minutes of a closing; the drop is judged again then
                self.set_pending_drop({"timestamp": current_timestamp, "reference_temperature": reference})
        if self.pending_drop and drops.temperature_confirms(self.sliding_window_temp, current_timestamp, self.pending_drop["reference_temperature"]):
            return 'drops_humid'
        if drops.temperature_drops_strongly(self.sliding_window_temp, sampled_sliding_window_temp, current_timestamp, self.last_closing_time):
            return 'drops_temp'
        if self.movingMean_feature.detect(current_timestamp, absolute_humidity, sampled_sliding_window_ah):
            return 'moving_mean'
        return None

    def is_closed(self, current_timestamp, sampled_sliding_window_temp) -> bool:
        smoothed = utils.recent_mean(self.sliding_window_ah, current_timestamp, 3)
        self.open_min = smoothed if self.open_min is None else min(self.open_min, smoothed)
        if smoothed - self.open_min >= self.config.close_ah_rise:
            return True
        # the room cools as long as the window is open; a short airing in mild weather cools it so little that the
        # rise afterwards stays below close_temperature_slope, but it does rise above its lowest value
        temperature = utils.recent_mean(self.sliding_window_temp, current_timestamp, 3)
        if temperature is not None:
            self.open_temperature_min = (temperature if self.open_temperature_min is None
                                         else min(self.open_temperature_min, temperature))
            if temperature - self.open_temperature_min >= self.config.close_temperature_rise:
                return True
        if utils.compute_n_min_slope(sampled_sliding_window_temp, 10) >= self.config.close_temperature_slope:
            return True
        max_open = pd.Timedelta(self.config.max_open_minutes, "min")
        return (max_open > pd.Timedelta(0) and self.open_since is not None and
                current_timestamp - self.open_since >= max_open)

    def save_detection(self, current_timestamp, current_value, feature, slope):
        mov_mean_mean = self.movingMean_feature.current_mean
        mov_mean_std = self.movingMean_feature.current_std

        utils.append_capped(self.detections, (current_timestamp, current_value, feature, slope, mov_mean_mean, mov_mean_std))
        save(self.data_path, ALL_DETECTIONS_FILENAME, self.detections)
        logger.info(f"{current_timestamp}:  Detected an open window!")
        if self.window_open == False:
            self.window_state_changed = True
        self.window_open = True
        self.open_since = current_timestamp
        self.open_min = utils.recent_mean(self.sliding_window_ah, current_timestamp, 3)
        self.open_temperature_min = utils.recent_mean(self.sliding_window_temp, current_timestamp, 3)
        self.temperature_before_opening = (self.pending_drop["reference_temperature"] if self.pending_drop else None) or \
            self.unusualDrops_feature.reference_temperature(self.sliding_window_temp, current_timestamp, self.last_closing_time)
        self.pending_drop = None
        self.save_window_state()

    def save_closed_window(self, current_timestamp, current_value):
        if self.window_open == True:
            self.window_state_changed = True
        self.window_open = False
        self.open_since = None
        self.open_min = None
        self.open_temperature_min = None
        self.last_closing_time=current_timestamp
        utils.append_capped(self.window_closing_times, (current_timestamp, current_value))
        # humidity stays below its baseline for a while after airing; that is no new opening
        self.movingMean_feature.wait_for_reset(current_timestamp, current_value)

        save(self.data_path, LAST_CLOSING_TIME_FILENAME, self.last_closing_time)
        save(self.data_path, WINDOW_CLOSING_FILENAME, self.window_closing_times)
        self.save_window_state()
        logger.info(f"{current_timestamp}:  Window closed!")

    def set_pending_drop(self, pending_drop):
        self.pending_drop = pending_drop
        self.save_window_state()

    def save_window_state(self):
        save(self.data_path, WINDOW_STATE_FILENAME, {
            "window_open": self.window_open,
            "open_since": self.open_since,
            "open_min": self.open_min,
            "open_temperature_min": self.open_temperature_min,
            "temperature_before_opening": self.temperature_before_opening,
            "pending_drop": self.pending_drop,
        })

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
        self.window_state_changed = False  # the change is reported with this answer
        logger.debug(answer)
        return answer

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="open-window-detection-operator", git_info_file='git_commit')
