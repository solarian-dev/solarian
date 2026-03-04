from __future__ import annotations
from typing import Any, Optional, Tuple, Type, TYPE_CHECKING
from enum import Enum
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator, Field
import time
import pickle
import zlib


class DataComputer(int, Enum):
    SERVER = 0
    ROBOT = 1
    CONTROLLER = 2


class Data(BaseModel):
    """
    Invariant:
      - if is_compressed == False: payload is the real python object
      - if is_compressed == True : payload is bytes (compressed)

    Behavior:
      - Accessing .value auto-uncompresses (mutates the instance) if needed.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,  # allow using "payload=" while still dumping as "value"
    )

    dtype: Type = int
    shape: Tuple[int, ...] | None = None
    path: tuple[str, str, str] | None = None
    timestamp: float = Field(default_factory=time.time)

    is_compressed: bool = False
    compression_codec: str | None = None

    # Internal storage; serialized/deserialized under key "value"
    payload: Any | None = Field(default=None, alias="value")
    name: str | None = None
    # -----------------------
    # Public API
    # -----------------------
    @property
    def value(self) -> Any | None:
        """
        Auto-uncompress on access (SIDE EFFECT).
        """
        return self.payload

    @value.setter
    def value(self, v: Any | None) -> None:
        """
        Setting value sets uncompressed state.
        """
        self.payload = v
        if type(v) == bytes:
            self.is_compressed = True
            self.uncompress()
        else:
            self.is_compressed = False
            self.compression_codec = None

    def compress(self) -> None:
        if self.is_compressed:
            return
        b = self._compress_impl(self.payload)
        self.payload = b
        self.is_compressed = True
        self.compression_codec = self._default_codec()

    def uncompress(self) -> None:
        if not self.is_compressed:
            return

        b = self.payload
        if b is None:
            raw = None
        else:
            if not isinstance(b, (bytes, bytearray, memoryview)):
                raise TypeError(
                    f"{self.__class__.__name__} is_compressed=True but value is {type(b)!r}, expected bytes-like."
                )
            raw = self._uncompress_impl(bytes(b))

        self.payload = raw
        self.is_compressed = False
        self.compression_codec = None

    def to_dict(self):
        # With auto-uncompress semantics, this returns the uncompressed value
        return {'value': self.value, 'timestamp': self.timestamp}

    @classmethod
    def zeros(cls):
        return cls()

    @model_validator(mode="after")
    def _check_compression_invariant(self):
        if self.is_compressed:
            if self.payload is not None and not isinstance(self.payload, (bytes, bytearray, memoryview)):
                raise TypeError("is_compressed=True requires payload to be bytes-like")
        else:
            # optional: forbid raw bytes when uncompressed (unless you want to allow it)
            # if isinstance(self.payload, (bytes, bytearray, memoryview)):
            #     raise TypeError("is_compressed=False but payload is bytes-like")
            pass
        return self

    def __set_name__(self, owner, name):
        object.__setattr__(self, 'name', name)

    def __get__(self, instance, owner):
        if instance is not None:
            object.__setattr__(self, 'path', (instance.parent.name, instance.name, self.name))
        return self

    def get_path(self):
        # returns a tuple of the form (module_name, component_name, data_name)
        return self.path
        

    # def duplicate(self, value=None):
    #     d = self.model_copy()
    #     if value is not None:
    #         d.value = value
    #     return d

    # receives a dictionary of data and returns a new Data instance
    def duplicate(self, data: dict = {}):
        d = self.model_copy()
        # set values 
        for name, value in data.items():
            setattr(d, name, value)
        return d

    # -----------------------
    # Overridable hooks
    # -----------------------
    def _default_codec(self) -> str:
        return "zlib+pickle"

    def _compress_impl(self, raw: Any) -> bytes:
        payload = pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL)
        return zlib.compress(payload, level=6)

    def _uncompress_impl(self, b: bytes) -> Any:
        payload = zlib.decompress(b)
        return pickle.loads(payload)





class PosData(Data):
    dtype: Type = float
    payload: float = Field(default=0.0, alias="value")

    if TYPE_CHECKING:
        @property
        def value(self) -> float: ...
        @value.setter
        def value(self, v: Any | None) -> None: ...


class SpeedData(Data):
    dtype: Type = float
    payload: float = Field(default=0.0, alias="value")


class TorqueEnableData(Data):
    dtype: Type = bool
    payload: bool = Field(default=False, alias="value")


class ImageData(Data):
    dtype: Type = np.uint8
    shape: Optional[Tuple[int, int, int]] = None
    payload: Optional[np.ndarray] = Field(default=None, alias="value")

    def model_post_init(self, __context: Any) -> None:
        # Only initialize if not compressed and no value provided
        if not self.is_compressed and self.payload is None and self.shape is not None:
            self.payload = np.zeros(self.shape, dtype=self.dtype)

    def _default_codec(self) -> str:
        return "jpeg"

    def _compress_impl(self, raw: Any) -> bytes:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("ImageData.compress() requires opencv-python (cv2).") from e

        if raw is None:
            return b""
        if not isinstance(raw, np.ndarray):
            raise TypeError(f"ImageData expects np.ndarray, got {type(raw)!r}")

        ok, buf = cv2.imencode(".jpg", raw)
        if not ok:
            raise RuntimeError("cv2.imencode('.jpg', image) failed")
        return buf.tobytes()

    def _uncompress_impl(self, b: bytes) -> Any:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("ImageData.uncompress() requires opencv-python (cv2).") from e

        if not b:
            if self.shape is None:
                return None
            return np.zeros(self.shape, dtype=self.dtype)

        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imdecode() failed")
        return img


class BoundingBoxData(Data):
    dtype: Type = np.uint8
    shape: Optional[Tuple[int, ...]] = (4,)
    payload: Tuple[int, int, int, int] = Field(default=(0, 0, 0, 0), alias="value")


class TimestampData(Data):
    dtype: Type = float
    payload: float = Field(default_factory=time.time, alias="value")