"""
   Copyright (c) 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Calibrates the humidity-drop trigger from a room's own history, without a window contact.

    uv run python -m tools.calibrate history.csv

The history is a CSV of what the operator receives: a `time` column and a `Temperature` and a `Humidity` column,
one row per message, empty where that message carried only the other reading. Three to four weeks of one room.

It prints three reports.

  1. How far the room's level moves by itself over the spans a drop is measured across - the scale
     ah_drop_std_factor multiplies. Measured over the whole history and, beside it, over one hour at a time,
     which is all a deployed operator holds. Where the two differ, the operator is judging a drop against a
     scale its own window was too short to see.

  1b. How far the readings move between two messages. Not a measure of the room when the sensor reports on
     change, and the numbers say why: several readings arrive within the same second for one event, and a long
     gap means the room held still rather than that it had longer to move.

  2. Airings found by looking forward, which an operator deciding in real time cannot do: a deep drop of absolute
     humidity that stays down, with the temperature falling too, and both coming back afterwards. These are not
     every airing - mild, humid weather hides one from this detector as much as from the operator - but the ones
     it finds are airings. They are the labels for report 3, and they cost no window contact.

  3. What ah_drop_std_factor does: for each candidate value, how many of those airings are caught and how fast,
     and how often the trigger fires outside them.

Report 3 drives the operator's own code, not a copy of it. It measures the humidity-drop trigger - the stage
ah_drop_std_factor controls - with and without the temperature confirmation, but without the open/close state
machine around it, which only ever suppresses detections. Its false alarm counts are therefore an upper bound on
the deployed operator's.
"""

import argparse
import sys
import tempfile

import numpy as np
import pandas as pd

from algo import utils
from algo.drops_and_risings_feature import AH_CHANGE_FLOOR, DropsAndRisingsFeature
from main import CustomConfig

INPUTS = ("Temperature", "Humidity")

# the spans report 1 measures the room's own movement over, in minutes. A drop is measured across the minutes
# between the halves of the operator's window, which is about half an hour of the hour it holds.
SPAN_MINUTES = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0)

# the gap lengths report 1b groups readings into, in minutes
GAP_EDGES = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0)


def read_history(path, max_age_minutes=90.0):
    """The CSV as (time, Temperature, Humidity, ah) in time order, each reading held until the next one and
    dropped once it is older than max_age_minutes, exactly as the operator holds and drops it."""
    frame = pd.read_csv(path)
    missing = [name for name in ("time",) + INPUTS if name not in frame.columns]
    if missing:
        raise ValueError(f"the history needs the columns time, Temperature and Humidity; it has no {', '.join(missing)}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time").reset_index(drop=True)
    held = frame[list(INPUTS)].ffill()
    for name in INPUTS:
        last_reading = frame["time"].where(frame[name].notna()).ffill()
        held.loc[frame["time"] - last_reading > pd.Timedelta(max_age_minutes, "min"), name] = np.nan
    held["time"] = frame["time"]
    held = held.dropna().reset_index(drop=True)
    held["ah"] = [utils.absolute_humidity(temperature, humidity)
                  for temperature, humidity in zip(held["Temperature"], held["Humidity"])]
    return held


def robust_scale(values):
    """The median absolute deviation of values as the standard deviation it stands for, 0.0 for fewer than two."""
    if len(values) < 2:
        return 0.0
    values = np.asarray(values, dtype=float)
    return utils.MAD_TO_STD * float(np.median(np.abs(values - np.median(values))))


def as_window(history):
    """The history as the sliding window of absolute humidity the operator's own code takes."""
    return [{"timestamp": pd.Timestamp(time), "value": value}
            for time, value in zip(history["time"], history["ah"])]


def scale_by_span(history, spans=SPAN_MINUTES):
    """Report 1: how far the room's level moves over each span by itself, measured the way the operator measures
    it - on a grid with every reading held until the next one. A frame with one row per span.

    `scale` has the whole history to measure it. `median_scale_per_hour` is the same measurement repeated on one
    hour at a time, which is all a deployed operator holds, and `hours_without_a_scale` counts the hours too
    short for the span or too still to show anything.
    """
    window = as_window(history)
    hours = _hourly_windows(window)
    rows = []
    for span in spans:
        per_hour = [utils.level_change_scale(hour, span, 0.0) for hour in hours]
        measured = [value for value in per_hour if value > 0]
        rows.append({
            "span_min": span,
            "scale": utils.level_change_scale(window, span, 0.0),
            "hours_measured": len(measured),
            "hours_without_a_scale": len(per_hour) - len(measured),
            "median_scale_per_hour": float(np.median(measured)) if measured else 0.0,
        })
    return pd.DataFrame(rows)


def _hourly_windows(window):
    """The window cut into calendar hours: one hour is as much of its own history as a deployed operator holds."""
    hours, current, edge = [], [], None
    for entry in window:
        hour = entry["timestamp"].floor("h")
        if hour != edge:
            if len(current) > 1:
                hours.append(current)
            current, edge = [], hour
        current.append(entry)
    if len(current) > 1:
        hours.append(current)
    return hours


def scale_by_gap(times, values, edges=GAP_EDGES):
    """Report 1b: for each range of gap lengths, the scale of the changes across it, raw and per square root of
    a minute. A frame with one row per range.

    A diagnostic, not what the operator decides on. It is where the square root the operator used to normalise
    its changes with was found to be the wrong idea for a sensor that reports on change.
    """
    # through pandas rather than numpy: the times carry a time zone, which numpy has no room for
    minutes = pd.to_datetime(pd.Series(list(times)), utc=True).diff().dt.total_seconds().to_numpy()[1:] / 60
    changes = np.diff(np.asarray(values, dtype=float))
    usable = minutes > 0
    minutes, changes = minutes[usable], changes[usable]

    rows = []
    for low, high in zip(edges, edges[1:]):
        inside = (minutes > low) & (minutes <= high)
        if not inside.any():
            continue
        rows.append({
            "gap_from_min": low,
            "gap_to_min": high,
            "readings": int(inside.sum()),
            "median_gap_min": float(np.median(minutes[inside])),
            "scale_raw": robust_scale(changes[inside]),
            "scale_per_sqrt_min": robust_scale(changes[inside] / np.sqrt(minutes[inside])),
        })
    return pd.DataFrame(rows)


def gap_exponent(report):
    """The exponent in scale ~ gap**exponent that report 1b implies. 0.5 is what a random walk gives, 0.0 a
    series whose changes do not grow with the gap at all. None for fewer than two usable ranges."""
    usable = report[(report["scale_raw"] > 0) & (report["median_gap_min"] > 0)]
    if len(usable) < 2:
        return None
    return float(np.polyfit(np.log(usable["median_gap_min"]), np.log(usable["scale_raw"]), 1)[0])


def minute_grid(history, max_fill_minutes=15):
    """The history on a regular minute grid, interpolated across gaps of at most max_fill_minutes and left empty
    across longer ones, so that an outage cannot pass for a slow change."""
    frame = history.set_index("time")[["ah", "Temperature"]]
    frame = frame.resample("1min").mean()
    return frame.interpolate(limit=max_fill_minutes, limit_area="inside")


def find_airings(grid, before=30, fall=20, hold=20, recovery=120, min_depth=0.8, hold_fraction=0.5,
                 recovery_fraction=0.3, min_temperature_fall=0.5):
    """Report 2: airings found by looking forward. A frame with one row per airing, `start` being the minute the
    fall begins and `end` the minute from which an alarm no longer belongs to it.

    A minute is a candidate when absolute humidity falls by at least min_depth g/m³ within the next `fall`
    minutes, is still at least hold_fraction of that fall below its earlier level for the `hold` minutes after
    that, has the temperature falling by at least min_temperature_fall K over the same span, and comes back to
    within recovery_fraction of the fall within the `recovery` minutes afterwards.

    That last condition is what separates an airing from weather: a room whose humidity falls and stays down was
    not aired, it is in a drier week. It is also what separates it from a shower ending, where the level comes
    back to below where it started rather than up to it.

    A whole run of minutes before a fall is a candidate, because each of them sees the fall ahead. The airing
    starts where humidity was last at its highest, which is where the window opened, not where the run began.

    The thresholds are deliberately strict. A missed airing costs report 3 one label; a wrong one costs report 3
    its meaning.
    """
    ah, temperature = grid["ah"], grid["Temperature"]
    level = ah.rolling(before, min_periods=before).median()
    lowest = ah.rolling(fall, min_periods=fall).min().shift(-fall)
    while_held = ah.rolling(hold, min_periods=hold).median().shift(-(fall + hold))
    # the closest it comes back to its earlier level, not where it spends the recovery
    highest_after = ah.rolling(recovery, min_periods=recovery).max().shift(-(fall + hold + recovery))
    temperature_level = temperature.rolling(before, min_periods=before).median()
    temperature_lowest = temperature.rolling(fall + hold, min_periods=fall + hold).min().shift(-(fall + hold))

    depth = level - lowest
    temperature_fall = temperature_level - temperature_lowest
    # a comparison against a missing value is False, so an incomplete window is no candidate
    candidate = ((depth >= min_depth) &
                 (level - while_held >= hold_fraction * depth) &
                 (level - highest_after <= recovery_fraction * depth) &
                 (temperature_fall >= min_temperature_fall))

    rows = []
    for first, last in _runs(candidate.to_numpy()):
        began = first + int(np.nanargmax(ah.to_numpy()[first:last + 1 + fall]))
        rows.append({
            "start": grid.index[began],
            "end": grid.index[last] + pd.Timedelta(fall + hold, "min"),
            "depth": float(depth.iloc[first:last + 1].max()),
            "temperature_fall": float(temperature_fall.iloc[first:last + 1].max()),
        })
    return pd.DataFrame(rows, columns=["start", "end", "depth", "temperature_fall"])


def _runs(mask):
    """(first, last) index of every run of True in mask."""
    runs, start = [], None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def replay(history, config):
    """Every reading through the operator's own humidity-drop trigger, with the normalised drop it decides on
    left as a number instead of compared against a threshold. A frame with one row per reading."""
    with tempfile.TemporaryDirectory() as data_path:
        feature = DropsAndRisingsFeature(data_path, "detections.pickle", "rebounds.pickle", config)
        window_ah, window_temperature, rows = [], [], []
        for time, temperature, ah in zip(history["time"], history["Temperature"], history["ah"]):
            timestamp = pd.Timestamp(time)
            window_ah = utils.update_sliding_window(window_ah, ah, timestamp)
            window_temperature = utils.update_sliding_window(window_temperature, temperature, timestamp)
            rows.append(_statistic(feature, timestamp, window_ah, window_temperature))
    return pd.DataFrame(rows)


def _statistic(feature, timestamp, window_ah, window_temperature):
    sampled_ah = utils.minute_resampling(window_ah)
    front_level, end_mean, lag_minutes = utils.compute_front_end_measures(sampled_ah)
    row = {"time": timestamp, "drop": np.nan, "scale": np.nan, "normalised_drop": np.nan,
           "falling": False, "temperature_confirms": False}
    if front_level is None:
        return row
    # the scale over the span the drop is measured across, so the ratio below is the drop in scales
    scale = utils.level_change_scale(window_ah, lag_minutes, AH_CHANGE_FLOOR)
    drop = front_level - end_mean
    descending = len(window_ah) >= 2 and window_ah[-1]["value"] < window_ah[-2]["value"]
    reference = feature.reference_temperature(window_temperature, timestamp, False)
    row.update({
        "drop": drop,
        "scale": scale,
        "normalised_drop": drop / scale,
        "falling": bool(descending and (feature.is_falling_last10min(sampled_ah) or
                                        feature.is_falling_extreme_last2min(window_ah))),
        "temperature_confirms": bool(feature.temperature_confirms(window_temperature, timestamp, reference)),
    })
    return row


def alarms(statistics, factor, drop_min, confirmed_by_temperature=False, merge_minutes=30):
    """The times the trigger fires at ah_drop_std_factor = factor, with firings less than merge_minutes apart
    counted as one: the operator does not open a window that is open already."""
    fires = (statistics["normalised_drop"] > factor) & (statistics["drop"] >= drop_min) & statistics["falling"]
    if confirmed_by_temperature:
        fires = fires & statistics["temperature_confirms"]
    merged, last = [], None
    for time in statistics.loc[fires, "time"]:
        if last is None or time - last >= pd.Timedelta(merge_minutes, "min"):
            merged.append(time)
        last = time
    return merged


def sweep(statistics, airings, factors, drop_min, confirmed_by_temperature=False, catch_minutes=20, margin_minutes=30):
    """Report 3: for each candidate ah_drop_std_factor, what it catches and what it costs. A frame with one row
    per factor. An airing counts as caught by an alarm within catch_minutes of its start; an alarm counts as
    false outside every airing and its margin."""
    days = _days(statistics)
    rows = []
    for factor in factors:
        fired = alarms(statistics, factor, drop_min, confirmed_by_temperature)
        latencies, caught = [], 0
        for start in airings["start"]:
            within = [time for time in fired if start <= time <= start + pd.Timedelta(catch_minutes, "min")]
            if within:
                caught += 1
                latencies.append((min(within) - start).total_seconds() / 60)
        false = [time for time in fired if not _inside_an_airing(time, airings, margin_minutes)]
        rows.append({
            "factor": factor,
            "alarms": len(fired),
            "airings_caught": caught,
            "recall": caught / len(airings) if len(airings) else float("nan"),
            "median_latency_min": float(np.median(latencies)) if latencies else float("nan"),
            "false_alarms": len(false),
            "false_alarms_per_day": len(false) / days if days else float("nan"),
        })
    return pd.DataFrame(rows)


def _inside_an_airing(time, airings, margin_minutes):
    margin = pd.Timedelta(margin_minutes, "min")
    return any(start - margin <= time <= end + margin for start, end in zip(airings["start"], airings["end"]))


def _days(statistics):
    if len(statistics) < 2:
        return 0.0
    return (statistics["time"].iloc[-1] - statistics["time"].iloc[0]).total_seconds() / 86400


def recommend(report, target_recall, false_alarm_budget):
    """The largest factor that still catches target_recall of the airings, and the smallest that stays within
    false_alarm_budget alarms a day. Either is None when no candidate manages it."""
    catching = report[report["recall"] >= target_recall]
    affordable = report[report["false_alarms_per_day"] <= false_alarm_budget]
    return (float(catching["factor"].max()) if len(catching) else None,
            float(affordable["factor"].min()) if len(affordable) else None)


def _print(title, frame):
    print(f"\n{title}")
    print("-" * len(title))
    if len(frame) == 0:
        print("nothing found")
    else:
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate the humidity-drop trigger from a room's own history.")
    parser.add_argument("history", help="CSV with the columns time, Temperature and Humidity")
    parser.add_argument("--factors", default="0.5,1,1.5,2,3,4,6,8,12",
                        help="the candidate values of ah_drop_std_factor (default: %(default)s)")
    parser.add_argument("--drop-min", type=float, default=None,
                        help="ah_drop_min to hold while sweeping; the operator's default unless given. "
                             "Pass 0 to see what the normalised threshold decides on its own")
    parser.add_argument("--min-depth", type=float, default=0.8,
                        help="g/m³ a fall must reach to count as an airing in report 2 (default: %(default)s)")
    parser.add_argument("--target-recall", type=float, default=0.8,
                        help="share of the airings a recommended factor must catch (default: %(default)s)")
    parser.add_argument("--false-alarms-per-day", type=float, default=1.0,
                        help="what a recommended factor may cost outside the airings (default: %(default)s)")
    parser.add_argument("--out", help="write the per-reading statistics to this CSV")
    arguments = parser.parse_args(argv)

    config = CustomConfig({})
    drop_min = config.ah_drop_min if arguments.drop_min is None else arguments.drop_min
    factors = [float(value) for value in arguments.factors.split(",")]

    history = read_history(arguments.history, config.max_humidity_age_minutes)
    if len(history) < 2:
        print("the history holds fewer than two readings with both a temperature and a humidity", file=sys.stderr)
        return 1
    print(f"{len(history)} readings from {history['time'].iloc[0]} to {history['time'].iloc[-1]}")

    spans = scale_by_span(history)
    _print("1. how far the room's level moves by itself, over the spans a drop is measured across", spans)
    print("\nah_drop_std_factor multiplies this. `scale` has the whole history to measure it; "
          "`median_scale_per_hour` is what one hour sees of it, and one hour is all a deployed operator holds.")

    gaps = scale_by_gap(history["time"], history["ah"])
    _print("1b. how far the readings move between two messages", gaps)
    exponent = gap_exponent(gaps)
    if exponent is not None:
        print(f"\nacross these gap lengths the change grows with gap**{exponent:.2f}, where a random walk would "
              f"give 0.50. A sensor reporting on change makes the gap no measure of time for the room: several "
              f"readings arrive within the same second for one event, and a long gap means the room held still.")

    airings = find_airings(minute_grid(history), min_depth=arguments.min_depth)
    _print("2. airings found by looking forward", airings)

    statistics = replay(history, config)
    if arguments.out:
        statistics.to_csv(arguments.out, index=False)
        print(f"\nper-reading statistics written to {arguments.out}")

    report = sweep(statistics, airings, factors, drop_min)
    _print(f"3. ah_drop_std_factor, humidity drop alone, at ah_drop_min = {drop_min:g}", report)
    confirmed = sweep(statistics, airings, factors, drop_min, confirmed_by_temperature=True)
    _print("3b. the same, with the temperature confirming the drop", confirmed)

    catching, affordable = recommend(confirmed, arguments.target_recall, arguments.false_alarms_per_day)
    print(f"\nlargest factor catching {arguments.target_recall:.0%} of the airings: "
          f"{'none of the candidates' if catching is None else f'{catching:g}'}")
    print(f"smallest factor within {arguments.false_alarms_per_day:g} false alarms a day: "
          f"{'none of the candidates' if affordable is None else f'{affordable:g}'}")
    print("Both numbers rest on the airings of report 2. Read report 2 first: a handful of airings is an "
          "anecdote, and a factor calibrated on one is too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
