"""Processor input resolution must survive multiprocessing spawn pickle."""

from __future__ import annotations

import pickle

from solarian.core.data import PosData
from solarian.core.execution import ExecutionPlace
from solarian.core.observation import CalculatedField, ModuleObs, SensorField
from solarian.perception.processor import Processor, field_name_for_input


class _DoubleProcessor(Processor):
    def process(self, inputs):
        return float(inputs[0].value) * 2


def _make_obs_cls():
    src = PosData(path=("mod", "grip", "pos"))

    class _Obs(ModuleObs):
        gripper = SensorField(src)
        doubled = CalculatedField(
            inputs=[src],
            processor=_DoubleProcessor,
            execution_place=ExecutionPlace.ROBOT,
            output=PosData(),
        )

    return _Obs, src


def test_field_name_for_input_matches_path_not_id():
    prev = ModuleObs._active_subclass
    try:
        obs_cls, src = _make_obs_cls()
        copied = pickle.loads(pickle.dumps(src))
        assert id(copied) != id(src)
        name = field_name_for_input(
            copied,
            obs_cls.get_sensor_fields_spec(),
            obs_cls.get_calculated_fields_spec(),
        )
        assert name == "gripper"
    finally:
        ModuleObs._active_subclass = prev


def test_processor_resolves_inputs_after_pickle():
    prev = ModuleObs._active_subclass
    try:
        obs_cls, _src = _make_obs_cls()
        proc = obs_cls.create_processors(ExecutionPlace.ROBOT)["doubled"]
        proc = pickle.loads(pickle.dumps(proc))
        assert id(proc.inputs[0]) != id(
            obs_cls.get_sensor_fields_spec()["gripper"].source
        )

        obs = obs_cls.zeros()
        obs["gripper"].value = 10.0
        result = proc(obs)
        assert float(result.value) == 20.0
    finally:
        ModuleObs._active_subclass = prev
