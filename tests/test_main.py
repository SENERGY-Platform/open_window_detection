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
import tempfile
import unittest

from operator_lib.util import OperatorConfig, create_filter_handler

from main import CustomConfig, Operator
from tests._harness import MockKafkaProducer

OPERATOR_CONFIG = {
    "config": {},
    "inputTopics": [
        {
            "name": "climate_service",
            "filterType": "DeviceId",
            "filterValue": "device:1",
            "mappings": [
                {"dest": "Temperature", "source": "value.temperature"},
                {"dest": "Humidity", "source": "value.humidity"},
            ],
        }
    ],
}


class TestRunPath(unittest.TestCase):
    def test_run_accepts_the_arguments_the_operator_lib_passes(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        opr_config = OperatorConfig(OPERATOR_CONFIG)
        operator = Operator()
        operator.init(
            kafka_consumer=None,
            kafka_producer=MockKafkaProducer(),
            filter_handler=create_filter_handler(opr_config.inputTopics, "pipeline:1", Operator.selectors),
            output_topic="output",
            pipeline_id="pipeline:1",
            operator_id="operator:1",
            config=CustomConfig({"data_path": data_dir.name}),
        )
        message = {"device_id": "device:1", "service_id": "climate:service",
                   "value": {"temperature": 21.0, "humidity": 50.0}}

        _, results = operator._OperatorBase__call_run(
            message, "device:1", datetime.datetime(2026, 1, 15, 8, tzinfo=datetime.timezone.utc))

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["window_open"])


if __name__ == "__main__":
    unittest.main()
