import onnx
import numpy as np
from onnx.backend.test.case.node import expect
from onnx import helper


x = (
    np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    .astype(np.uint8)
    .reshape((1, 1, 3, 3))
)
x_zero_point = np.uint8(1)
w = np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2))

y = (
    np.array([1, 3, 5, 3, 5, 12, 16, 9, 11, 24, 28, 15, 7, 15, 17, 9])
    .astype(np.int32)
    .reshape((1, 1, 4, 4))
)

# ConvInteger with padding
convinteger_node_with_padding = onnx.helper.make_node(
    "ConvInteger",
    inputs=["x", "w", "x_zero_point"],
    outputs=["y"],
    pads=[1, 1, 1, 1],
)

expect(
    convinteger_node_with_padding,
    inputs=[x, w, x_zero_point],
    outputs=[y],
    name="test_convinteger_with_padding",
)