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
"""

import random
import unittest

import pandas as pd

from algo import utils

T0 = pd.Timestamp("2026-01-15T08:00:00Z")

FLOOR = 0.05


def window(values, start=T0, step_seconds=60):
    return [{"timestamp": start + pd.Timedelta(i * step_seconds, "s"), "value": v} for i, v in enumerate(values)]


def random_walk(steps, step_size, seed=1):
    """A series whose level moves by step_size per minute on average, the way a room's humidity drifts."""
    rnd = random.Random(seed)
    value, values = 10.0, []
    for _ in range(steps):
        value += rnd.gauss(0, step_size)
        values.append(value)
    return values


def republished(sliding_window, after_seconds=0.4):
    """The window as a device that publishes its whole state on every attribute change leaves it: every reading
    once more a fraction of a second later, carrying the same value."""
    repeated = []
    for entry in sliding_window:
        repeated.append(entry)
        repeated.append({"timestamp": entry["timestamp"] + pd.Timedelta(after_seconds, "s"), "value": entry["value"]})
    return repeated


class TestSlope(unittest.TestCase):
    def test_slope_of_a_linear_rise_is_its_rate_per_minute(self):
        rising = window([50 + 0.1 * i for i in range(10)])

        self.assertAlmostEqual(utils.compute_n_min_slope(rising, 10), 0.1)

    def test_slope_covers_only_the_last_n_minutes(self):
        flat_then_rising = window([50] * 20 + [50 + 0.2 * i for i in range(1, 11)])

        self.assertAlmostEqual(utils.compute_n_min_slope(flat_then_rising, 10), 0.2)


class TestMinuteResampling(unittest.TestCase):
    def test_interpolates_linearly_at_every_full_minute(self):
        readings = [{"timestamp": T0 + pd.Timedelta(30, "s"), "value": 10.0},
                    {"timestamp": T0 + pd.Timedelta(150, "s"), "value": 14.0}]

        grid = utils.minute_resampling(readings)

        self.assertEqual([e["timestamp"] for e in grid], [T0 + pd.Timedelta(1, "min"), T0 + pd.Timedelta(2, "min")])
        self.assertEqual([round(e["value"], 6) for e in grid], [11.0, 13.0])

    def test_a_later_reading_with_the_same_timestamp_replaces_the_earlier_one(self):
        readings = [{"timestamp": T0, "value": 10.0}, {"timestamp": T0, "value": 12.0},
                    {"timestamp": T0 + pd.Timedelta(2, "min"), "value": 12.0}]

        grid = utils.minute_resampling(readings)

        self.assertEqual([e["value"] for e in grid], [12.0, 12.0, 12.0])


class TestFrontEndMeasures(unittest.TestCase):
    def test_reports_the_minutes_between_the_midpoints_of_front_and_end(self):
        # 60 readings a minute apart: the front is the first 55, the end the last 5
        _, _, lag_minutes = utils.compute_front_end_measures(window([10.0] * 60))

        self.assertAlmostEqual(lag_minutes, 30.0)

    def test_reports_the_front_median_and_the_end_mean(self):
        # a spike in the front must not move the level it stands for
        front_level, end_mean, _ = utils.compute_front_end_measures(window([10.0] * 30 + [99.0] + [10.0] * 24 + [8.0] * 5))

        self.assertAlmostEqual(front_level, 10.0)
        self.assertAlmostEqual(end_mean, 8.0)

    def test_reports_nothing_while_the_window_is_too_short_to_show_a_change(self):
        self.assertEqual(utils.compute_front_end_measures(window([10.0] * 20)), (None, None, None))


class TestHeldLevel(unittest.TestCase):
    def test_a_reading_holds_until_the_next_one(self):
        readings = [{"timestamp": T0, "value": 10.0}, {"timestamp": T0 + pd.Timedelta(3, "min"), "value": 12.0}]
        every_minute = pd.date_range(T0, T0 + pd.Timedelta(4, "min"), freq="min")

        self.assertEqual(utils.held_level(readings, every_minute), [10.0, 10.0, 10.0, 12.0, 12.0])

    def test_the_last_of_several_readings_at_one_instant_is_the_one_in_force(self):
        readings = [{"timestamp": T0, "value": 10.0}, {"timestamp": T0, "value": 11.0}]

        self.assertEqual(utils.held_level(readings, [T0]), [11.0])


class TestLevelChangeScale(unittest.TestCase):
    def test_the_scale_of_a_drifting_room_grows_with_the_span(self):
        """A level that wanders is further from where it was the longer one waits, and for a random walk it is
        further by the square root of the wait. That growth is measured here rather than assumed: the gaps
        between a sensor's readings were the wrong place to look for it.

        Over two days of readings, because the estimate needs many spans' worth of window: the same walk over
        ten hours puts this ratio anywhere between 1.6 and 3.0.
        """
        drifting = window(random_walk(3000, step_size=0.2))

        over_10 = utils.level_change_scale(drifting, 10, FLOOR)
        over_40 = utils.level_change_scale(drifting, 40, FLOOR)

        self.assertAlmostEqual(over_10, 0.2 * 10 ** 0.5, delta=0.08)
        self.assertGreater(over_40 / over_10, 1.7)
        self.assertLess(over_40 / over_10, 2.4)

    def test_several_readings_of_one_message_do_not_change_the_scale(self):
        """zigbee2mqtt publishes the whole state on every attribute change, so one event arrives as several
        readings a fraction of a second apart. They are one level, not a step the room took in no time."""
        drifting = window(random_walk(200, step_size=0.2))

        self.assertAlmostEqual(utils.level_change_scale(republished(drifting), 20, FLOOR),
                               utils.level_change_scale(drifting, 20, FLOOR))

    def test_readings_far_apart_at_the_same_level_are_no_movement(self):
        # a sensor reporting on change stays silent while its value holds; that silence is a still room
        quiet = window([10.0] * 20, step_seconds=20 * 60)

        self.assertEqual(utils.level_change_scale(quiet, 30, FLOOR), FLOOR)

    def test_a_series_that_never_moves_falls_back_to_the_floor(self):
        self.assertEqual(utils.level_change_scale(window([10.0] * 60), 30, FLOOR), FLOOR)

    def test_a_single_large_jump_does_not_raise_the_scale(self):
        # within a tenth of the scale: the jump crosses 20 of the 580 changes and moves their order, which a
        # median absorbs and a standard deviation would not
        values = random_walk(600, step_size=0.2)
        quiet = window(values)
        with_an_event = window(values[:300] + [value - 5.0 for value in values[300:]])

        self.assertAlmostEqual(utils.level_change_scale(with_an_event, 20, FLOOR),
                               utils.level_change_scale(quiet, 20, FLOOR), delta=0.1)

    def test_a_window_shorter_than_twice_the_span_falls_back_to_the_floor(self):
        # one change over the span is a single number, and a single number is no scale
        drifting = window(random_walk(40, step_size=0.2))

        self.assertEqual(utils.level_change_scale(drifting, 30, FLOOR), FLOOR)


class TestLogs(unittest.TestCase):
    def test_a_log_keeps_only_its_newest_entries(self):
        log = []

        for i in range(utils.LOG_LIMIT + 5):
            utils.append_capped(log, i)

        self.assertEqual(log, list(range(5, utils.LOG_LIMIT + 5)))


if __name__ == "__main__":
    unittest.main()
