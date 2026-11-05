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

import datetime
import random
import tempfile
import unittest

from algo.moving_mean_feature import MovingMeanFeature

T0 = datetime.datetime(2026, 1, 15, tzinfo=datetime.timezone.utc)


def new_feature(data_path, parameters=None):
    return MovingMeanFeature(data_path, "detections.pickle", "dismissed.pickle", "pending.pickle",
                             "waiting.pickle", model_parameters=parameters)


def feed_periods(feature, period_means, noise=0.5, seed=1):
    """One reading every 5 minutes for each 6 h period, then one more that closes the last period."""
    rnd = random.Random(seed)
    timestamp = T0
    for mean in period_means:
        for _ in range(72):
            feature.update_stats(timestamp, mean + rnd.gauss(0, noise), False)
            timestamp += datetime.timedelta(minutes=5)
    feature.update_stats(timestamp, period_means[-1], False)
    return timestamp


class MovingMeanTestCase(unittest.TestCase):
    def setUp(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        self.data_path = data_dir.name


class TestBaselineSpread(MovingMeanTestCase):
    def test_includes_the_daily_cycle_between_periods(self):
        feature = new_feature(self.data_path)

        feed_periods(feature, [45, 50, 55, 50])

        # within-period noise is 0.5; the period means alone spread by about 3
        self.assertGreater(feature.current_std, 2.5)


if __name__ == "__main__":
    unittest.main()
