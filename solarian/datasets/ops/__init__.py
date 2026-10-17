"""The dataset ops solarian ships; plugins add theirs under ``solarian.dataset_ops``."""

from solarian.datasets.ops.delete_episodes import DeleteEpisodesOp
from solarian.datasets.ops.merge import MergeOp
from solarian.datasets.ops.mirror import MirrorOp
from solarian.datasets.ops.split_pick_place import SplitPickPlaceOp

BUILTIN_OPS = (DeleteEpisodesOp, SplitPickPlaceOp, MirrorOp, MergeOp)
