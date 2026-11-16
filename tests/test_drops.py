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

import math
import tempfile
import unittest

import pandas as pd

from algo import utils
from algo.drops_and_risings_feature import DropsAndRisingsFeature
from main import CustomConfig

T0 = pd.Timestamp("2026-01-15T08:00:00Z")


def an_hour_ending_in_a_drop(wander, drop):
    """An hour of absolute humidity a minute apart: a level swinging by `wander` g/m³ around 9.0 with a period
    of 20 minutes, and a fall of `drop` g/m³ over the last 5 minutes on top of it.

    A level that swings, not a reading that jitters: the drop is judged against how far this room's level moves
    over the half hour the drop is measured across, and a room whose readings alternate around a level that
    stands still is not a room where such a drop is ordinary.
    """
    values = [9.0 + wander * math.sin(2 * math.pi * minute / 20) for minute in range(60)]
    for minute in range(55, 60):
        values[minute] -= drop * (minute - 54) / 5
    return [{"timestamp": T0 + pd.Timedelta(minute, "min"), "value": value} for minute, value in enumerate(values)]


class TestUnusualDrops(unittest.TestCase):
    """The same drop is unusual in a quiet room and ordinary in a restless one, with the same configuration."""

    def setUp(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        self.feature = DropsAndRisingsFeature(data_dir.name, "detections.pickle", "rebounds.pickle", CustomConfig({}))

    def falls_unusually(self, readings):
        return self.feature.is_falling_unusually(readings, utils.minute_resampling(readings))

    def test_a_quiet_room_reports_a_drop_of_2_as_unusual(self):
        self.assertTrue(self.falls_unusually(an_hour_ending_in_a_drop(wander=0.02, drop=2.0)))

    def test_a_restless_room_does_not_report_the_same_drop(self):
        self.assertFalse(self.falls_unusually(an_hour_ending_in_a_drop(wander=1.5, drop=2.0)))

    def test_a_restless_room_reports_a_drop_that_stands_out_from_its_own_movement(self):
        self.assertTrue(self.falls_unusually(an_hour_ending_in_a_drop(wander=1.5, drop=6.0)))

    def test_a_rise_is_no_drop(self):
        rising = [{"timestamp": entry["timestamp"], "value": -entry["value"]}
                  for entry in an_hour_ending_in_a_drop(wander=0.02, drop=2.0)]

        self.assertFalse(self.falls_unusually(rising))

    def test_the_unusual_drop_grows_with_the_room_it_is_measured_in(self):
        quiet = an_hour_ending_in_a_drop(wander=0.02, drop=0.0)
        restless = an_hour_ending_in_a_drop(wander=1.5, drop=0.0)

        self.assertGreater(self.feature.unusual_drop(restless, 30), self.feature.unusual_drop(quiet, 30))


if __name__ == "__main__":
    unittest.main()
