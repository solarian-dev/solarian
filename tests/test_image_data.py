"""The observation JPEG is the picture a browser is shown, and what else
crosses the wire."""

import cv2
import numpy as np

from solarian.core.data import ImageData, MaskData, PosData, set_image_jpeg_quality


def test_a_received_jpeg_round_trips_and_stays_with_the_pixels():
    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[..., 0] = 255
    set_image_jpeg_quality(90)
    try:
        sent = ImageData(value=red)
        sent.compress()
        jpeg = sent.payload

        # OpenCV decodes to BGR, so a browser-correct red sits in channel 2.
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert int(decoded[0, 0, 2]) > 250
        assert int(decoded[0, 0, 0]) < 5

        received = ImageData(shape=(16, 16, 3)).duplicate(
            {"value": jpeg, "timestamp": 1.0}
        )
        assert received._jpeg == jpeg
        assert int(received.value[0, 0, 0]) > 250, "the array stays RGB"
        assert int(received.value[0, 0, 2]) < 5

        received.value = red
        assert received._jpeg is None, "replacing the pixels retires the JPEG"
    finally:
        set_image_jpeg_quality(None)


def test_a_number_crosses_the_wire_as_itself():
    sent = PosData(value=1.5)
    sent.compress()
    assert sent.to_dict()["value"] == 1.5
    received = PosData().duplicate(sent.to_dict())
    assert received.value == 1.5


def test_an_empty_mask_decodes_to_zeros_and_a_target_to_itself():
    for raw in (np.zeros((8, 12), np.uint8), np.eye(8, 12, dtype=np.uint8)):
        for _ in range(2):  # the second empty encode reuses the first
            sent = MaskData(value=raw)
            sent.compress()
            received = MaskData().duplicate(sent.to_dict())
            np.testing.assert_array_equal(received.value, raw)
