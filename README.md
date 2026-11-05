# Open Window Detection
Detects whether a window is open from a room's temperature and relative humidity.

## How it works

Relative humidity alone is ambiguous: an open window lowers the water content of the air, but it also lowers the
temperature, which raises relative humidity. The operator therefore converts each humidity reading into absolute
humidity (g/m³, Magnus formula) using the latest temperature reading, and decides on absolute humidity and
temperature:

- **Opening.** Absolute humidity falls within minutes when a window opens; the temperature follows more slowly,
  because walls and furniture buffer heat far more than moisture. A drop of absolute humidity counts as an open window
  once the temperature has fallen below its level from before that drop. A strong temperature drop alone also counts,
  for airing in mild weather when outdoor air holds about as much water as the room. After an initial phase, a
  long-term baseline of absolute humidity detects values far below normal as well.
- **Closing.** The room cools as long as the window is open. So the window counts as closed when the temperature or
  absolute humidity rises again above its lowest value while open, or when the heating brings the temperature back up
  quickly.
- **Humidity rebound.** Within 30 minutes after closing, the operator reports if humidity comes back too high, judged
  on the relative humidity the air would have at the room temperature from before the airing, so the cold air right
  after closing does not pass for humid air.

Every reading is evaluated at the time Kafka stored the message.

## Start from history

A fresh deployment, or the first start of this version, has nothing to continue from. It then reads the history of its
inputs from Timescale, as long as the initialization phase plus one hour, and runs it through the detection at the
times Timescale stored the readings, without sending outputs. So it starts with the short-term windows, the long-term
baseline and the initialization phase it would have had running that long; with a complete history, the
initialization phase is over at the start. Kafka messages up to the end of that history are skipped, they are part of
it already.

The history is read with the database connection the platform gives a deployed operator (`TS_CONN`, or `ts_conn` in
the operator config). Without one, or for an input only Kafka has, such as the output of another operator, the
operator logs a warning and starts without history. A restart continues from the saved state and reads no history.

## Input

| key           | type  | description                    |
|---------------|-------|--------------------------------|
| `Temperature` | float | Room temperature in °C         |
| `Humidity`    | float | Relative humidity in percent   |

Both are required: without a temperature reading from the last `max_temperature_age_minutes`, the operator reports the
window state but detects nothing. They may come in one message or in separate ones. Missing, non-numeric and
out-of-range readings are logged and skipped.

## Output

| key                          | type    | description                                                                          |
|------------------------------|---------|--------------------------------------------------------------------------------------|
| `window_open`                | boolean | Whether the window is detected as open                                               |
| `timestamp`                  | string  | Time the state was evaluated for                                                     |
| `humidity_too_fast_too_high` | string  | The timestamp, while humidity came back too high after closing; otherwise empty      |
| `initial_phase`              | string  | Message whether and how long the operator is in its initialization phase            |
| `notification`               | string  | `Window opened.` or `Window closed.` once per change, as `notification_mode` selects |

There is one output per humidity reading; temperature readings produce none outside the initialization phase.

## Config options

| key                 | type   | description                                                                                         | required |
|---------------------|--------|-----------------------------------------------------------------------------------------------------|----------|
| `logger_level`      | string | `info`, `warning` (default), `error`, `critical`, `debug`                                           | no       |
| `data_path`         | string | Path to the mounted volume                                                                          | no       |
| `init_phase_length` | number | Length of the initialization phase (default: 2), and of the history read at the start, plus 1 hour  | no       |
| `init_phase_level`  | string | Unit of the initialization phase length (pandas timedelta string, e.g. `d` for days)                | no       |
| `notification_mode` | string | `never` (default), `always`, `when_opened`, `when_closed`                                           | no       |

### Detection thresholds

All optional; humidity values are absolute humidity in g/m³, temperatures in K.

| key                               | default | description                                                                       |
|-----------------------------------|---------|-----------------------------------------------------------------------------------|
| `max_temperature_age_minutes`     | 10      | Older temperature readings are not used to compute absolute humidity              |
| `ah_drop_min`                     | 0.4     | Drop of the last 5 minutes below the median of the hour before                    |
| `ah_drop_std_factor`              | 1       | The same drop in units of the hour's noise around its trend                        |
| `ah_falling_slope`                | 0.02    | Fall per minute over the last 10 minutes, or ...                                   |
| `ah_extreme_drop`                 | 0.9     | ... fall between two consecutive readings, that the drop needs as well             |
| `temperature_confirm_drop`        | 0.3     | Temperature fall below its level from 5 to 15 minutes before the humidity drop ... |
| `temperature_confirm_minutes`     | 20      | ... within this many minutes of the humidity drop                                  |
| `temperature_strong_drop`         | 1.5     | Temperature fall that counts as an open window on its own ...                      |
| `temperature_strong_drop_minutes` | 15      | ... within this many minutes                                                       |
| `close_temperature_rise`          | 0.3     | Rise above the lowest temperature while open that counts as closed                 |
| `close_temperature_slope`         | 0.05    | Temperature rise per minute over 10 minutes that counts as closed                  |
| `close_ah_rise`                   | 0.3     | Rise above the lowest absolute humidity while open that counts as closed           |
| `max_open_minutes`                | 0       | Minutes after which an open window counts as closed; 0 for no limit                |
| `rebound_humidity`                | 65      | Relative humidity in % at the room temperature before opening ...                  |
| `rebound_minutes`                 | 30      | ... within this many minutes after closing that counts as a rebound                |

### Long-term baseline

All optional; see `algo/moving_mean_feature.py` for their meaning.

| key                            | default |
|--------------------------------|---------|
| `moving_mean_init_phase_days`  | 10      |
| `moving_mean_slope_minutes`    | 10      |
| `moving_mean_std_ratio`        | 3       |
| `moving_mean_days`             | 1       |
| `moving_mean_hours_per_period` | 6       |
| `moving_mean_error_influence`  | 0.6     |
| `moving_mean_reset_slope`      | 0.01    |

The defaults come from a simple room model, not from field data with known window states; calibrate them on real
data before relying on them.
