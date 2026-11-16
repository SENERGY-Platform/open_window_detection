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

### How large a drop has to be

How far absolute humidity has to fall to count is measured in the room's own terms, not as one number for every
room. Over the same span the drop itself is measured across — the minutes between the two halves of the operator's
hour, about half an hour — it estimates how far this room's level moves by itself: the readings are put on a
one-minute grid with every reading held until the next one, and the scale is the median change in magnitude between
two grid points that far apart, as the standard deviation it stands for. A drop has to exceed `ah_drop_std_factor`
times that scale, and `ah_drop_min` on top of that as an absolute floor.

So a quiet room reports a smaller drop as unusual than a restless one does, with the same configuration, and
`ah_drop_std_factor` is the one threshold that is meant to carry over between rooms. Two properties of the estimate
matter for reading it:

- **Both sides of the comparison span the same time**, so nothing converts between spans. Nothing can: an earlier
  version divided each change between two readings by the square root of its own gap, and two real rooms refuted it.
  A sensor that reports on change breaks that assumption twice. It publishes its whole state on every attribute
  change, so one event arrives as several readings within the same second, which no length of time separates — on a
  Bosch BTH-RM230Z those were two thirds of all readings. And a long gap between readings means the room held still
  rather than that it had longer to move: measured over nine days, the changes grew with the gap to the power 0.23
  on one room and −0.02 on the other, where a random walk gives 0.50.
- **The level is held, not interpolated.** A reading holds until the next one, which is what a sensor reporting on
  change means. An interpolated quiet half hour is a straight line of small changes and an interpolated jump is a
  gentle slope; both are movement the room did not make.

A window shorter than twice the span, or with fewer than ten changes on the grid, falls back to a floor of
0.05 g/m³, and the detection then rests on `ah_drop_min` alone. The operator holds one hour, which is about two
spans, so its own estimate is coarse: measured over ten hours of a drifting series, the growth between two spans
comes out anywhere between 1.6 and 3.0 where it should be 2.0, and it takes two days to settle within a tenth. A
scale worth trusting comes from weeks of history, which is what `tools/calibrate.py` measures — see **Calibration**.
The defaults have not been calibrated against field data.

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

Both are required. They may come in one message or in separate ones, at fixed intervals or on change: the operator
decides with every message, on the last reading of each input. That matters for a humidity sensor that reports on
change, because while a window is open the relative humidity often hardly changes: the falling temperature makes up
for the falling water content. A reading older than `max_temperature_age_minutes` or `max_humidity_age_minutes` counts
as missing; then the operator logs a warning, reports the window state and detects nothing. Missing, non-numeric and
out-of-range readings are logged and skipped.

## Output

| key                          | type    | description                                                                          |
|------------------------------|---------|--------------------------------------------------------------------------------------|
| `window_open`                | boolean | Whether the window is detected as open                                               |
| `timestamp`                  | string  | Time the state was evaluated for                                                     |
| `humidity_too_fast_too_high` | string  | The timestamp, while humidity came back too high after closing; otherwise empty      |
| `initial_phase`              | string  | Message whether and how long the operator is in its initialization phase            |
| `notification`               | string  | `Window opened.` or `Window closed.` once per change, as `notification_mode` selects |

There is one output per message.

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
| `max_temperature_age_minutes`     | 90      | Older temperature readings count as missing; 90 covers sensors reporting hourly    |
| `max_humidity_age_minutes`        | 90      | Older humidity readings count as missing                                           |
| `ah_drop_min`                     | 0.4     | Smallest drop of the last 5 minutes below the median of the hour before that counts at all |
| `ah_drop_std_factor`              | 1       | The same drop in units of how far the room's level moves by itself over that span  |
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

## Calibration

`tools/calibrate.py` is a development tool, not part of the image. It reads a CSV of one room's history
(`time,Temperature,Humidity`) and prints three reports:

1. **How far the room's level moves by itself, over the spans a drop is measured across.** The scale
   `ah_drop_std_factor` multiplies, measured over the whole history and, beside it, over one hour at a time, which is
   all a deployed operator holds. Where the two differ, the operator is judging a drop against a scale its own window
   was too short to see.
1b. **The same between two messages, per gap length.** A diagnostic rather than what the operator decides on: it is
   where normalising a change by the square root of its own gap was refuted, and it is where that shows up again for
   a new sensor.
2. **Airings found by looking forward.** Deep, sustained drops of absolute humidity that recover afterwards,
   accompanied by a temperature fall. Looking forward is what a deployed operator cannot do, which is why these are
   usable as labels without a window contact. The thresholds are deliberately strict: a missed airing costs report 3
   one label, a wrong one costs it its meaning.
3. **A sweep over `ah_drop_std_factor`.** Per value, how many of those airings are caught and after how long, and how
   often the detection fires outside them. The sweep runs the operator's own code, not a copy of it.

Two things the reports cannot tell you. The airings of report 2 inherit their own blind spots — an airing in mild,
humid weather leaves no humidity signature for them to find either — so the sweep is calibrated on the winter case.
And a tilted window is barely represented in them at all.

```bash
uv run python -m tools.calibrate history.csv
```

## Development

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`; change them with `uv add` or `uv lock`.

```bash
uv sync
uv run python -m unittest discover -s tests -t .
```

`operator-manifest.json` declares the inputs, outputs and config values the platform registers for a version. It is
generated from the declarations in `main.py`, and the release workflow regenerates it before each version:

```bash
uv run python -m operator_lib manifest main.py
```
