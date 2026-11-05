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
from tests._harness import MockKafkaProducer, at, new_operator

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


class TestInvalidReadings(unittest.TestCase):
    def setUp(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        self.operator, _ = new_operator(data_dir.name)
        self.send({"Temperature": 21.0, "Humidity": 50.0}, 0)  # past the init phase reset

    def send(self, data, minute):
        return self.operator.run(data=data, selector=None, device_id="device:1", timestamp=at(minute))

    def test_ignores_a_humidity_reading_that_is_null(self):
        self.assertIsNone(self.send({"Humidity": None}, 1))

    def test_ignores_a_humidity_reading_that_is_not_a_number(self):
        self.assertIsNone(self.send({"Humidity": float("nan")}, 1))

    def test_ignores_a_humidity_reading_outside_0_to_100_percent(self):
        self.assertIsNone(self.send({"Humidity": 150.0}, 1))

    def test_ignores_a_temperature_reading_that_is_not_a_number(self):
        self.assertIsNone(self.send({"Temperature": "warm"}, 1))

    def test_keeps_reporting_after_an_invalid_reading(self):
        self.send({"Humidity": None}, 1)

        self.assertFalse(self.send({"Humidity": 50.0}, 2)["window_open"])


class TestInitPhase(unittest.TestCase):
    def test_first_init_message_of_a_fresh_deployment_names_a_real_end_time(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)

        _, producer = new_operator(data_dir.name, init_phase_length="2", init_phase_level="d")

        self.assertNotIn("NaT", producer.messages[0]["analytics"]["initial_phase"])


if __name__ == "__main__":
    unittest.main()
