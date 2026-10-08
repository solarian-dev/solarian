import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solarian.core.data import Data


def field_name_for_input(inp, sensor_fields, calculated_fields) -> str | None:
    """Map a schema ``Data`` ref to its observation field name.

    ``id()`` is not stable across multiprocessing spawn: the parent pickles
    processor ``inputs``, the child unpickles copies, and
    ``get_sensor_fields_spec()`` still points at the re-imported class-level
    objects. ``Data.path`` is pickled with the object, so it is the identity
    that survives. ``id()`` remains a same-process fallback for
    ``CalculatedField.output`` values that have no path.
    """
    path = getattr(inp, "path", None)
    if path is not None:
        path_t = tuple(path)
        for fname, sf in sensor_fields.items():
            src_path = getattr(sf.source, "path", None)
            if src_path is not None and tuple(src_path) == path_t:
                return fname
        for fname, cf in calculated_fields.items():
            out_path = getattr(cf.output, "path", None)
            if out_path is not None and tuple(out_path) == path_t:
                return fname
    inp_id = id(inp)
    for fname, sf in sensor_fields.items():
        if id(sf.source) == inp_id:
            return fname
    for fname, cf in calculated_fields.items():
        if id(cf.output) == inp_id:
            return fname
    return None


def load_processor_conf(conf: str):
    """A processor's yaml with ``${deployment.*}`` resolved for this host.

    Every machine that builds a processor reads the same project conf and
    deployment, so they all agree on a value such as a service URL or the
    execution place of a field.
    """
    from omegaconf import OmegaConf
    from solarian.utils.configs import resolve_with_deployment

    return OmegaConf.create(resolve_with_deployment(OmegaConf.load(conf)))


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
        self.cfg = load_processor_conf(conf) if conf is not None else None

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

    def start(self, params: dict) -> None:
        """Begin (or restart) computing, e.g. track the object at ``params["point"]``.

        Sent by ``START_SERVICE``. Processors that always compute ignore it.
        """

    def stop(self) -> None:
        """Stop computing; the field goes back to its empty value."""

    def _resolve_input_fields(self, obs):
        """Build mapping from schema Data refs to field names (once)."""
        actual_cls = type(obs)
        sensor_fields = actual_cls.get_sensor_fields_spec()
        calculated_fields = actual_cls.get_calculated_fields_spec()

        names: list[str] = []
        for inp in self.inputs:
            fname = field_name_for_input(inp, sensor_fields, calculated_fields)
            if fname is None:
                raise KeyError(
                    f"Processor input {inp!r} (path={getattr(inp, 'path', None)}) "
                    f"does not match any SensorField.source or "
                    f"CalculatedField.output on {actual_cls.__name__}"
                )
            names.append(fname)
        self._input_field_names = names
