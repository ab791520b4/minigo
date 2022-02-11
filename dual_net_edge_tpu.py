# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file contains an implementation of dual_net.py for Google's EdgeTPU.
It can only be used for inference and requires a specially quantized and
compiled model file.

For more information see https://coral.withgoogle.com
"""

import numpy as np
import features as features_lib
import go
from pycoral.utils import edgetpu
from pycoral.adapters import common


def extract_agz_features(position):
    return features_lib.extract_features(position, features_lib.AGZ_FEATURES)


class DualNetworkEdgeTpu():
    """DualNetwork implementation for Google's EdgeTPU."""

    def __init__(self, save_file):
        self.interpreter = edgetpu.make_interpreter(save_file)
        self.interpreter.allocate_tensors()
        self.board_size = go.N
        self.output_policy_size = self.board_size**2 + 1

        if len(self.interpreter.get_input_details()) != 1:
            raise RuntimeError(
                'Invalid number of input tensors {}. Expected: {}'.format(
                    len(self.interpreter.get_input_details()), 1))

        input_tensor_shape = self.interpreter.get_input_details()[0]['shape']

        expected_input_shape = [1, self.board_size, self.board_size, 17]
        if not np.array_equal(input_tensor_shape, expected_input_shape):
            raise RuntimeError(
                'Invalid input tensor shape {}. Expected: {}'.format(
                    input_tensor_shape, expected_input_shape))

        output_tensors_sizes = [output_tensor_detail['shape'].prod() for output_tensor_detail in self.interpreter.get_output_details()]
        expected_output_tensor_sizes = [self.output_policy_size, 1]
        if not np.array_equal(output_tensors_sizes,
                              expected_output_tensor_sizes):
            raise RuntimeError(
                'Invalid output tensor sizes {}. Expected: {}'.format(
                    output_tensors_sizes, expected_output_tensor_sizes))

    def run(self, position):
        """Runs inference on a single position."""
        probs, values = self.run_many([position])
        return probs[0], values[0]

    def run_many(self, positions):
        """Runs inference on a list of position."""
        processed = map(extract_agz_features, positions)
        probabilities = []
        values = []
        for state in processed:
            assert state.shape == (self.board_size, self.board_size,
                                   17), str(state.shape)
            edgetpu.run_inference(self.interpreter, state.flatten())
            output_details = self.interpreter.get_output_details()

            # dequantize output tensor #0 (policy output)
            output_data = self.interpreter.tensor(output_details[0]['index'])().flatten()
            scale, zero_point = output_details[0]['quantization']
            policy_output = scale * (output_data.astype(np.int64) - zero_point)

            # dequantize output tensor #1 (value output)
            output_data = self.interpreter.tensor(output_details[1]['index'])().flatten()
            scale, zero_point = output_details[1]['quantization']
            value_output = scale * (output_data.astype(np.int64) - zero_point)

            probabilities.append(policy_output)
            values.append(value_output)
        return probabilities, values
