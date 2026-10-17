from __future__ import annotations

from typing import Iterable

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, Field

from solarian.core.recording.archive import TABLE_FILE, table_bytes
from solarian.datasets.op import DatasetOp, Episode, EpisodeOut, OpContext, OpKind

#: session.json entries that only warn when inputs disagree; the first input's wins.
SOFT_KEYS = ("robot", "fps", "video", "task")


class MergeParams(BaseModel):
    strict_columns: bool = Field(default=False, description="Fail when the inputs' columns differ, instead of keeping the common ones.")
    strict_dtypes: bool = Field(default=False, description="Fail when a column's dtype differs, instead of promoting it.")


def merge_schema(metas: list[dict], params: MergeParams, log) -> dict:
    """``{column: spec}`` common to every input, in the first input's order, dtypes promoted."""
    columns = [set(m["schema"]) for m in metas]
    common = set.intersection(*columns)
    for i, cols in enumerate(columns):
        if cols - common:
            if params.strict_columns:
                raise ValueError(f"input #{i} has columns the others lack: {sorted(cols - common)}")
            log(f"WARNING input #{i}: dropping columns not in every input: {sorted(cols - common)}")
    if not common:
        raise ValueError("The inputs have no table columns in common")
    schema = {}
    for name in metas[0]["schema"]:
        if name not in common:
            continue
        dtypes = [m["schema"][name]["dtype"] for m in metas]
        spec = dict(metas[0]["schema"][name])
        if len(set(dtypes)) > 1:
            if params.strict_dtypes:
                raise ValueError(f"{name}: dtype differs across inputs: {dtypes}")
            spec["dtype"] = np.result_type(*dtypes).name
            log(f"WARNING {name}: dtype differs across inputs ({dtypes}); promoting to {spec['dtype']}")
        schema[name] = spec
    return schema


def _fields(fields: dict, prefix: str, schema: dict) -> dict:
    """``obs_fields`` / ``act_fields`` restricted to the merged columns."""
    out = {}
    for module, names in fields.items():
        kept = [f for f in names if f"{prefix}.{module}.{f}" in schema]
        if kept:
            out[module] = kept
    return out


def _conform(table: pa.Table, schema: dict) -> pa.Table:
    table = table.select(list(schema))
    for i, name in enumerate(table.column_names):
        dtype = pa.from_numpy_dtype(np.dtype(schema[name]["dtype"]))
        if table.schema.field(name).type != dtype:
            table = table.set_column(i, name, table.column(name).cast(dtype))
    return table


class MergeOp(DatasetOp):
    name = "merge"
    title = "Merge datasets"
    description = """
    Concatenate the episodes of two or more sessions, in input order. Cameras
    must match. Tables are merged on the columns every input has (dtypes
    promoted); only the tables that change are rewritten, everything else is
    read from the inputs.
    """
    kind = OpKind.COMBINE
    Params = MergeParams
    min_inputs = 2
    max_inputs = None

    def setup(self, params: MergeParams, ctx: OpContext) -> None:
        metas = [s.meta for s in ctx.inputs]
        cameras = {n: (c["width"], c["height"]) for n, c in metas[0]["cameras"].items()}
        for i, meta in enumerate(metas[1:], start=1):
            other = {n: (c["width"], c["height"]) for n, c in meta["cameras"].items()}
            if other != cameras:
                raise ValueError(f"input #{i} has cameras {other}, input #0 {cameras}")
            for key in SOFT_KEYS:
                if meta.get(key) != metas[0].get(key):
                    ctx.log(f"WARNING session.json {key!r} differs in input #{i}; using input #0's")
        self.schema = merge_schema(metas, params, ctx.log)

    def session_meta(self, metas: list[dict], params: MergeParams) -> dict:
        ref = metas[0]
        return {**ref, "schema": self.schema,
                "obs_fields": _fields(ref.get("obs_fields", {}), "obs", self.schema),
                "act_fields": _fields(ref.get("act_fields", {}), "act", self.schema)}

    def combine(self, params: MergeParams, ctx: OpContext) -> Iterable[EpisodeOut]:
        for session in ctx.inputs:
            declared = session.meta["schema"]
            conforms = list(declared) == list(self.schema) and all(
                declared[k]["dtype"] == self.schema[k]["dtype"] for k in self.schema)
            for index in session.episodes():
                episode = Episode(session, index)
                if conforms:
                    yield episode.derive()
                    continue
                stats = {k: v for k, v in episode.meta.get("stats", {}).items() if k in self.schema}
                yield episode.derive({TABLE_FILE: table_bytes(_conform(episode.table(), self.schema))}, stats=stats)
