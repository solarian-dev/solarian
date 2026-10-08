"""JSON wire encoding shared by the HTTP policy and service clients/servers."""

import base64
from typing import Any, Dict

import numpy as np


def encode_arrays(values: Dict[str, Any]) -> Dict[str, Any]:
    """Encode a dict for the wire; numpy arrays go as raw bytes.

    Nested JSON lists would be ~5x the array's own size in decimal text, and
    writing them costs the sender more CPU than the model costs the receiver:
    a 224x224x3 float32 image measured 3.05 MB and 142 ms on a Pi 5, against
    176 ms for the diffusion policy it was being sent to. ``dtype`` carries
    byte order, so a little-endian sender and a big-endian receiver agree.
    """
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            out[key] = {
                "__ndarray__": base64.b64encode(value.tobytes()).decode("ascii"),
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
        else:
            out[key] = value
    return out


def decode_arrays(values: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a JSON-decoded dict back into numpy arrays.

    Accepts both the raw-buffer encoding :func:`encode_arrays` sends and plain
    nested lists, so anything hand-rolling a request against these endpoints —
    curl, a notebook, another language — still works.
    """
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            out[key] = value
        elif isinstance(value, dict) and "__ndarray__" in value:
            # Through a bytearray because a buffer over immutable bytes gives
            # a read-only array, and torch warns on those.
            buffer = bytearray(base64.b64decode(value["__ndarray__"]))
            out[key] = np.frombuffer(
                buffer, dtype=np.dtype(value["dtype"])
            ).reshape(value["shape"])
        elif isinstance(value, list):
            out[key] = np.array(value)
        else:
            out[key] = value
    return out


def to_serializable(value: Any) -> Any:
    """Recursively turn numpy types into JSON-serialisable Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value
