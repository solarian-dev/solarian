"""The episode table's columns, derived from the project's own schema.

Nothing here is hand-written per robot. ``Observation.get_module_obs_spec()``
and ``Action.get_module_actions_spec()`` already describe every signal a
project declared, down to the ``Data`` subclass behind it, so the columns and
their dtypes fall out of walking them. A project that adds a joint gets a
column without touching this file.

Image fields are split out rather than tabulated: they go to mp4, and the
table carries only the frame index and capture stamp that point into it. Other
vector-valued fields are left out entirely — a column holds a scalar, and the
``data_class`` recorded beside each one is what keeps a servo temperature
distinguishable from a joint angle, which a flat float vector cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from solarian.core.action import Action
from solarian.core.data import ImageData, TimestampData
from solarian.core.observation import Observation

#: Seconds since the episode's start, and the robot's synced wall clock.
TIME_COLUMNS = ("t", "wall_ts")

#: Parquet column dtype -> the numpy dtype its row buffer is allocated with.
NUMPY_BY_DTYPE = {
    "float64": np.float64,
    "float32": np.float32,
    "int32": np.int32,
    "bool": np.bool_,
}


@dataclass(frozen=True)
class CameraKey:
    """One captured image field, i.e. one mp4 per episode.

    ``sensor`` is the component behind the field, which is how the robot finds
    the camera's resolution and colour mode to size an encoder.
    """

    module: str
    field: str
    sensor: str

    @property
    def name(self) -> str:
        return f"{self.module}.{self.field}"

    @property
    def frame_index_column(self) -> str:
        return f"cam.{self.name}.frame_index"

    @property
    def capture_ts_column(self) -> str:
        return f"cam.{self.name}.capture_ts"


Band = float | tuple[float, float] | None


@dataclass(frozen=True)
class Column:
    """One scalar column of ``episode.parquet``."""

    name: str
    dtype: str
    data_class: str
    unit: str | None = None
    #: The component behind this column, so a consumer can tell a Cartesian
    #: axis from a joint without parsing the column name.
    component: str | None = None
    #: Warning bands as declared: a number is one-sided (``value >= band``), a
    #: pair is two-sided (``value <= lo or value >= hi``). Flattening a pair to
    #: its upper element would read as a one-sided band and lose the lower one.
    mid: Band = None
    high: Band = None

    def to_json(self) -> dict:
        # ``unit`` is always emitted, null included: a column nobody gave a unit
        # is a gap in the project's schema, and hiding the key hides the gap.
        out: dict = {"data_class": self.data_class, "dtype": self.dtype, "unit": self.unit}
        if self.component is not None:
            out["component"] = self.component
        if self.mid is not None:
            out["mid"] = _band_json(self.mid)
        if self.high is not None:
            out["high"] = _band_json(self.high)
        return out


@dataclass(frozen=True)
class RecordingSchema:
    """Every column of an episode table, plus the cameras it points at."""

    columns: tuple[Column, ...]
    cameras: tuple[CameraKey, ...]
    #: Observation modules in declaration order with the field names tabulated
    #: for each. Image fields are absent; they are in ``cameras``.
    obs_fields: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())
    act_fields: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def camera(self, name: str) -> CameraKey | None:
        return next((c for c in self.cameras if c.name == name), None)

    def numeric_columns(self) -> tuple[str, ...]:
        """Columns worth accumulating statistics over at seal time.

        Timestamps are excluded: the min and max of a Unix epoch is the
        episode's own start and end, which ``episode.json`` already states.
        """
        return tuple(
            c.name
            for c in self.columns
            if c.dtype == "float32" and not c.name.endswith(".ts")
        )

    def to_json(self) -> dict:
        return {c.name: c.to_json() for c in self.columns}


def _column_dtype(spec) -> str | None:
    """Parquet dtype for a field spec, or ``None`` if it is not tabulated.

    Everything numeric lands in float32: these are sensor readings, and the
    extra mantissa of float64 would store noise. Timestamps are the exception
    and are declared explicitly, because float32 cannot resolve a millisecond
    inside a Unix epoch. Arrays (boxes, pictures, masks) are not tabulated;
    a picture or mask may declare no shape, so its dtype is what marks it.
    """
    if spec.shape is not None or spec.dtype not in (bool, int, float):
        return None
    return "bool" if spec.dtype is bool else "float32"


def _band_json(band: Band):
    """A band as it was declared: a number, or a two-element list."""
    if isinstance(band, (tuple, list)):
        return [float(band[0]), float(band[1])]
    return float(band)


def _threshold(band) -> Band:
    """Normalise a ``Data`` threshold band, keeping both sides of a pair.

    ``SensorField`` declares ``mid`` / ``high``; ``CalculatedField`` does not,
    so the attribute is genuinely optional here.
    """
    if band is None:
        return None
    if isinstance(band, (tuple, list)):
        return (float(band[0]), float(band[1]))
    return float(band)


def build_schema() -> RecordingSchema:
    """Walk the active ``Observation`` and ``Action`` subclasses."""
    columns: list[Column] = [
        Column(
            name=name,
            dtype="float64",
            data_class="TimestampData",
            unit=TimestampData.unit,
        )
        for name in TIME_COLUMNS
    ]
    cameras: list[CameraKey] = []
    obs_fields: list[tuple[str, tuple[str, ...]]] = []
    act_fields: list[tuple[str, tuple[str, ...]]] = []

    for module_name, module_obs in Observation.get_module_obs_spec().items():
        module_cls = type(module_obs)
        sensor_specs = module_cls.get_sensor_fields_spec()
        specs = {**sensor_specs, **module_cls.get_calculated_fields_spec()}

        # One bus read fills every field of a module, so one capture stamp per
        # module is the honest granularity; per field would imply a precision
        # the driver does not have.
        columns.append(
            Column(
                name=f"obs.{module_name}.ts",
                dtype="float64",
                data_class="TimestampData",
                unit=TimestampData.unit,
            )
        )

        tabulated: list[str] = []
        for field_name, spec in specs.items():
            # A calculated ImageData is a derived picture, not a capture: no
            # sensor produced it and no encoder can be sized for it, so it
            # falls through and is simply not recorded.
            if spec.data_class is ImageData and field_name in sensor_specs:
                camera = CameraKey(
                    module=module_name, field=field_name, sensor=spec.sensor_name
                )
                cameras.append(camera)
                columns.append(
                    Column(
                        name=camera.frame_index_column,
                        dtype="int32",
                        data_class="ImageData",
                    )
                )
                columns.append(
                    Column(
                        name=camera.capture_ts_column,
                        dtype="float64",
                        data_class="ImageData",
                        unit=TimestampData.unit,
                        component=spec.sensor_name,
                    )
                )
                continue

            dtype = _column_dtype(spec)
            if dtype is None:
                continue
            tabulated.append(field_name)
            columns.append(
                Column(
                    name=f"obs.{module_name}.{field_name}",
                    dtype=dtype,
                    data_class=spec.data_class.__name__,
                    unit=spec.unit,
                    # A CalculatedField has no component: a processor made it.
                    component=getattr(spec, "sensor_name", None),
                    mid=_threshold(getattr(spec, "mid", None)),
                    high=_threshold(getattr(spec, "high", None)),
                )
            )
        obs_fields.append((module_name, tuple(tabulated)))

    # Unlike Observation's, Action's accessor does not resolve the active
    # subclass for you, so the base class would report no modules at all.
    action_cls = Action._resolve_class()
    for module_name, module_action in action_cls.get_module_actions_spec().items():
        specs = type(module_action).get_actuator_fields_spec()
        tabulated = []
        for field_name, spec in specs.items():
            dtype = _column_dtype(spec)
            if dtype is None:
                continue
            tabulated.append(field_name)
            columns.append(
                Column(
                    name=f"act.{module_name}.{field_name}",
                    dtype=dtype,
                    data_class=spec.data_class.__name__,
                    unit=spec.unit,
                    component=spec.actuator_name,
                    mid=_threshold(spec.mid),
                    high=_threshold(spec.high),
                )
            )
        act_fields.append((module_name, tuple(tabulated)))

    return RecordingSchema(
        columns=tuple(columns),
        cameras=tuple(cameras),
        obs_fields=tuple(obs_fields),
        act_fields=tuple(act_fields),
    )
