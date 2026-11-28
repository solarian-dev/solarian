from __future__ import annotations
from typing import Any, ClassVar, Optional, Tuple, Type, TYPE_CHECKING, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
import time
import pickle
import zlib


# Optional JPEG quality for ImageData.compress (OpenCV default is 95 if unset).
_image_jpeg_quality: int | None = None


def set_image_jpeg_quality(quality: int | None) -> None:
    """Set process-wide JPEG quality used by :class:`ImageData` compression."""
    global _image_jpeg_quality
    if quality is None:
        _image_jpeg_quality = None
    else:
        _image_jpeg_quality = int(max(1, min(100, quality)))


def get_image_jpeg_quality() -> int | None:
    return _image_jpeg_quality


def encode_jpeg(image: np.ndarray) -> bytes:
    """Browser JPEG of an RGB array, at :func:`get_image_jpeg_quality`.

    OpenCV encodes BGR. The swap here is undone by
    :meth:`ImageData._uncompress_impl`, so a round trip returns the same
    array, and the file itself is the one a browser can show.
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("JPEG encoding requires opencv-python (cv2).") from e

    if image.ndim == 3 and image.shape[2] == 3:
        # cvtColor, not a reversed view copied contiguous: 0.4 ms against 7 ms
        # for a 1014x760 frame on a Pi 5.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = np.ascontiguousarray(image)
    quality = get_image_jpeg_quality()
    if quality is None:
        ok, buf = cv2.imencode(".jpg", image)
    else:
        ok, buf = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', image) failed")
    return buf.tobytes()


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

    # Optional GUI / warning bands. A float is one-sided (value >= band);
    # (lo, hi) is two-sided (value <= lo or value >= hi). ``high`` wins.
    mid: ClassVar[Union[float, Tuple[float, float], None]] = None
    high: ClassVar[Union[float, Tuple[float, float], None]] = None

    # Physical unit, when the *kind* of data fixes it. Left None where one
    # class spans several units: a PosData is millimetres on a gripper and
    # metres on a Cartesian axis, so there the component declares it.
    unit: ClassVar[str | None] = None

    # The component class that declared this descriptor, filled by
    # __set_name__. A ClassVar rather than a field so it never reaches the
    # wire, and None for a Data built standalone rather than as an attribute.
    component_class: ClassVar[type | None] = None

    # Seconds an observation reading of this kind stays current, for
    # quantities that change far slower than a control tick and cost a bus
    # round trip to read. None reads it on every tick.
    refresh_period: ClassVar[float | None] = None

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
            self._drop_encoded()
            self.is_compressed = True
            self.uncompress()
        else:
            self.is_compressed = False
            self.compression_codec = None
            self._drop_encoded()

    def _drop_encoded(self) -> None:
        return

    def compress(self) -> None:
        # A plain number crosses the wire as itself: msgpack packs a float in
        # 9 bytes, pickle+zlib in about 20, and the receiver only decodes bytes.
        if self.is_compressed or type(self.payload) in (float, int, bool):
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
        # Unlike ``path``, which every access rewrites and so is only valid if
        # read straight away, this is fixed at class creation. It is what lets a
        # field ask the component for a unit the Data class cannot know.
        object.__setattr__(self, 'component_class', owner)

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

    def duplicate(self, data: dict = {}):
        d = self.model_copy()
        for name, value in data.items():
            setattr(d, name, value)
        if d.payload is not None and not d.is_compressed:
            d.payload = d._coerce_payload(d.payload)
        return d

    def _coerce_payload(self, value):
        """Coerce *value* to the type declared by ``dtype`` / ``shape``."""
        if self.shape is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"{type(self).__name__} expects ndarray with shape "
                    f"{self.shape}, got {type(value).__name__}"
                )
            return value
        # Scalar: convert numpy scalars / compatible types to dtype
        if isinstance(value, self.dtype):
            return value
        try:
            return self.dtype(value.item() if isinstance(value, np.generic) else value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise TypeError(
                f"{type(self).__name__} expects {self.dtype.__name__}, "
                f"got {type(value).__name__} ({value!r})"
            ) from exc

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





#: Shapes that bound components report for their ``Data``, by
#: ``(module, component, data)`` path. A descriptor belongs to its component
#: class and so is shared by every instance: two cameras of one class at
#: different resolutions can only be told apart by path.
_bound_shapes: dict[tuple[str, str, str], tuple[int, ...]] = {}


def bind_shape(path: tuple[str, str, str], shape: tuple[int, ...]) -> None:
    _bound_shapes[path] = tuple(shape)


def shape_at(path: tuple[str, str, str] | None, default: tuple[int, ...] | None) -> tuple[int, ...] | None:
    """The shape bound at *path*, else *default* (the descriptor's own)."""
    return _bound_shapes.get(path, default)


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
    # The JPEG this array was decoded from. Retired when the pixels are
    # replaced: a later encode would be a different picture.
    _jpeg: bytes | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        # Only initialize if not compressed and no value provided
        if not self.is_compressed and self.payload is None and self.shape is not None:
            self.payload = np.zeros(self.shape, dtype=self.dtype)

    def _default_codec(self) -> str:
        return "jpeg"

    def _drop_encoded(self) -> None:
        self._jpeg = None

    def uncompress(self) -> None:
        encoded = self.payload if self.is_compressed else None
        super().uncompress()
        if isinstance(encoded, (bytes, bytearray, memoryview)) and encoded:
            self._jpeg = bytes(encoded)

    def _compress_impl(self, raw: Any) -> bytes:
        if raw is None:
            return b""
        if not isinstance(raw, np.ndarray):
            raise TypeError(f"ImageData expects np.ndarray, got {type(raw)!r}")
        return encode_jpeg(raw)

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
        # encode_jpeg swapped RGB to BGR for OpenCV. Swap back.
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class BoundingBoxData(Data):
    """Pixel ``(x0, y0, x1, y1)`` in the source image. All zeros = no box."""

    dtype: Type = np.int32
    shape: Optional[Tuple[int, ...]] = (4,)
    payload: Optional[np.ndarray] = Field(default=None, alias="value")

    def model_post_init(self, __context: Any) -> None:
        if not self.is_compressed and self.payload is None:
            self.payload = np.zeros(self.shape, dtype=self.dtype)

    def _coerce_payload(self, value):
        return np.asarray(value, dtype=self.dtype).reshape(self.shape)

    def overlay_box(self) -> np.ndarray:
        """``(x0, y0, x1, y1)`` the UI draws."""
        return np.asarray(self.value)[:4]

    def overlay_text(self) -> str | None:
        """What the UI writes by the box; a subclass carrying more than the
        corners says what they mean here. None writes nothing."""
        return None


class PointData(Data):
    """Pixel ``(x, y, visibility %)`` in the source image. All zeros = no point.

    The UI draws it as a ``box_px`` square centred on the point.
    """

    dtype: Type = np.float32
    shape: Optional[Tuple[int, ...]] = (3,)
    payload: Optional[np.ndarray] = Field(default=None, alias="value")
    #: Side of the square the UI draws, in source pixels.
    box_px: int = 24

    def model_post_init(self, __context: Any) -> None:
        if not self.is_compressed and self.payload is None:
            self.payload = np.zeros(self.shape, dtype=self.dtype)

    def _coerce_payload(self, value):
        return np.asarray(value, dtype=self.dtype).reshape(self.shape)

    def overlay_box(self) -> np.ndarray:
        """The ``box_px`` square centred on the point; all zeros for no point."""
        value = np.asarray(self.value, dtype=np.float32)
        if not value.any():
            return np.zeros(4, dtype=np.int32)
        x, y = value[:2]
        r = self.box_px / 2
        return np.rint([x - r, y - r, x + r, y + r]).astype(np.int32)

    def overlay_text(self) -> str | None:
        value = np.asarray(self.value)
        if not value.any():
            return None
        visibility = int(round(float(value[2])))
        return f"point {visibility}%" if visibility >= 50 else f"occluded {visibility}%"


_EMPTY_MASK_PNG: dict[tuple[int, ...], bytes] = {}


class MaskData(Data):
    """``(H, W)`` uint8 0/1 segmentation of the source image. All zeros = no target."""

    dtype: Type = np.uint8
    shape: Optional[Tuple[int, int]] = None
    payload: Optional[np.ndarray] = Field(default=None, alias="value")

    def model_post_init(self, __context: Any) -> None:
        if not self.is_compressed and self.payload is None and self.shape is not None:
            self.payload = np.zeros(self.shape, dtype=self.dtype)

    def _coerce_payload(self, value):
        return np.asarray(value, dtype=self.dtype)

    def _default_codec(self) -> str:
        return "png"

    def _compress_impl(self, raw: Any) -> bytes:
        import cv2

        if raw is None:
            return b""
        mask = np.asarray(raw, dtype=np.uint8)
        # Until a target is picked the mask is all zeros, at camera rate; the
        # check costs a thirtieth of the encode on a Pi.
        empty = not mask.any()
        if empty and mask.shape in _EMPTY_MASK_PNG:
            return _EMPTY_MASK_PNG[mask.shape]
        # Lossless and tiny for a binary image; JPEG would blur the edge.
        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError("cv2.imencode('.png', mask) failed")
        encoded = buf.tobytes()
        if empty:
            _EMPTY_MASK_PNG[mask.shape] = encoded
        return encoded

    def _uncompress_impl(self, b: bytes) -> Any:
        import cv2

        if not b:
            return None if self.shape is None else np.zeros(self.shape, dtype=self.dtype)
        mask = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError("cv2.imdecode() failed for a mask")
        return mask


class TimestampData(Data):
    dtype: Type = float
    payload: float = Field(default_factory=time.time, alias="value")
    unit: ClassVar[str | None] = "s"


class TemperatureData(Data):
    """Servo / sensor temperature in degrees Celsius."""

    dtype: Type = float
    payload: float = Field(default=0.0, alias="value")
    unit: ClassVar[str | None] = "degC"
    # A motor heats over tens of seconds.
    refresh_period: ClassVar[float | None] = 1.0
    # Feetech over-temp cutout is 70 °C; warn well before that.
    mid: ClassVar[Union[float, Tuple[float, float], None]] = 50.0
    high: ClassVar[Union[float, Tuple[float, float], None]] = 60.0

    if TYPE_CHECKING:
        @property
        def value(self) -> float: ...
        @value.setter
        def value(self, v: Any | None) -> None: ...


ThresholdBand = Union[float, Tuple[float, float], None]


def threshold_exceeded(value: float, band: ThresholdBand) -> bool:
    """Return True if ``value`` is outside ``band``.

    A float is one-sided (``value >= band``). A ``(lo, hi)`` tuple is
    two-sided (``value <= lo or value >= hi``). ``None`` never exceeds.
    """
    if band is None:
        return False
    if isinstance(band, (tuple, list)) and len(band) == 2:
        lo, hi = float(band[0]), float(band[1])
        return value <= lo or value >= hi
    return value >= float(band)


def threshold_level(value: float, mid: ThresholdBand, high: ThresholdBand) -> str | None:
    """Return ``"high"``, ``"mid"``, or ``None``. ``high`` wins if both match."""
    if threshold_exceeded(value, high):
        return "high"
    if threshold_exceeded(value, mid):
        return "mid"
    return None