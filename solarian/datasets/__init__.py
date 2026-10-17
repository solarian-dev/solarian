"""Dataset operations: curate and enrich recorded sessions as immutable versions.

``solarian dataset ops`` lists what is installed; ``solarian dataset run`` runs
one. Each op is a :class:`DatasetOp` (see :mod:`solarian.datasets.op`) and
every output is a version of its inputs (:mod:`solarian.core.recording.archive`).
A :class:`ComputedDataset` groups the versions built over one source, and an
:class:`OpSession` redoes and reviews an open version's episodes one at a time.
"""

from solarian.datasets.op import (
    Cancelled,
    DatasetOp,
    Episode,
    EpisodeOut,
    OpContext,
    OpKind,
    Resources,
    discover_ops,
)
from solarian.datasets.runner import OutputSummary, run_op
from solarian.datasets.computed import ComputedDataset
from solarian.datasets.op_session import OpSession

__all__ = [
    "Cancelled",
    "ComputedDataset",
    "DatasetOp",
    "Episode",
    "EpisodeOut",
    "OpContext",
    "OpKind",
    "OpSession",
    "OutputSummary",
    "Resources",
    "discover_ops",
    "run_op",
]
