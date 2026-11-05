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

import json

import pandas as pd

from main import CustomConfig, Operator

T0 = pd.Timestamp("2026-01-15T08:00:00Z")


class MockKafkaProducer:
    def __init__(self):
        self.messages = []

    def produce(self, topic, value, key, **kwargs):
        self.messages.append(json.loads(value))


def new_operator(data_path, **config):
    """An operator set up the way OperatorLib does it, keeping its state in data_path."""
    producer = MockKafkaProducer()
    operator = Operator()
    operator.init(
        kafka_consumer=None,
        kafka_producer=producer,
        filter_handler=None,
        output_topic="output",
        pipeline_id="pipeline:1",
        operator_id="operator:1",
        config=CustomConfig({"data_path": data_path, "init_phase_length": "0", "init_phase_level": "min", **config}),
    )
    return operator, producer


def at(minute, seconds=0):
    return (T0 + pd.Timedelta(minute, "min") + pd.Timedelta(seconds, "s")).to_pydatetime()


def feed(operator, readings):
    """Sends (minute, temperature, humidity) readings the way the device does, temperature first and humidity
    20 s later. Returns (minute, output) for every message that produced an output."""
    outputs = []
    for minute, temperature, humidity in readings:
        for data, timestamp in (({"Temperature": temperature}, at(minute)), ({"Humidity": humidity}, at(minute, 20))):
            result = operator.run(data=data, selector=None, device_id="device:1", timestamp=timestamp)
            if result:
                outputs.append((minute, result))
    return outputs


def transitions(outputs):
    """The minutes at which window_open changed, with the new state."""
    changes, state = [], False
    for minute, output in outputs:
        if output["window_open"] != state:
            changes.append((minute, output["window_open"]))
            state = output["window_open"]
    return changes
