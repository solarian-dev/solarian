import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solarian.core.data import Data


class Processor:
    """
    Base class for observation/action processors.

    Subclass and implement ``process()`` for custom computation.

    Created by the component that owns the computation (robot ``Module``,
    ``RelayServer``, ``Controller``) via ``create_processors()``, which
    passes the ``inputs`` (schema Data refs) and ``output`` (Data template)
    from the ``CalculatedField`` declaration.

    Calling the processor with a ``ModuleObs`` resolves inputs, runs
    ``process()``, and returns a new ``Data`` instance::

        result = processor(module_obs)   # returns Data
        module_obs[field_name] = result
    """

    def __init__(self, inputs: list["Data"], output: "Data", conf: str | None = None):
        self.inputs = inputs
        self.output = output

        if conf is not None:
            from omegaconf import OmegaConf
            self.cfg = OmegaConf.load(conf)
        else:
            self.cfg = None

        self._input_field_names: list[str] | None = None

    def __call__(self, obs) -> "Data":
        if self._input_field_names is None:
            self._resolve_input_fields(obs)
            assert self._input_field_names is not None
        input_data = [obs[name] for name in self._input_field_names]
        value = self.process(input_data)
        result = self.output.duplicate()
        result.value = value
        result.timestamp = time.time()
        return result

    def process(self, inputs: list["Data"]):
        """Override in subclasses with the actual computation.

        Args:
            inputs: Resolved Data instances from the observation.

        Returns:
            The computed value (will be set on the output Data).
        """
        raise NotImplementedError

    def on_event(self, event_name: str, data=None):
        """Handle an external event (e.g., user click, stop tracking)."""
        pass

    def _resolve_input_fields(self, obs):
        """Build mapping from schema Data refs to field names (once)."""
        actual_cls = type(obs)
        sensor_fields = actual_cls.get_sensor_fields_spec()
        calculated_fields = actual_cls.get_calculated_fields_spec()

        source_to_field: dict[int, str] = {}
        for fname, sf in sensor_fields.items():
            source_to_field[id(sf.source)] = fname
        for fname, cf in calculated_fields.items():
            source_to_field[id(cf.output)] = fname

        self._input_field_names = [
            source_to_field[id(inp)] for inp in self.inputs
        ]
