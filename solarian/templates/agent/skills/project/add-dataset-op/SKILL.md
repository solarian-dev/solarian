---
name: add-dataset-op
description: Write a Solarian dataset op (filter, map, expand or combine recorded episodes - drop bad episodes, add a tracked bbox, split pick from place, merge sessions) in this project's datasets/ folder and run it on recordings with `solarian dataset`. Use when the user wants to clean, enrich or reshape recorded data before training.
---

# Add a dataset op

Docs: `solarian docs show datasets`. List what exists first, a built-in may do it already:

```bash
solarian dataset ops                  # built-ins: delete-episodes, split-pick-place, mirror, merge, plus plugin ops
solarian dataset describe <op>
```

## 1. The file

One op per file in `{{ package }}/datasets/<op_name>.py`. No registration: the file stem, with `_` as `-`, is the op name, and `solarian dataset ops` lists it as `project:<file>`.

```python
# {{ package }}/datasets/drop_short_episodes.py
from pydantic import BaseModel, Field

from solarian.datasets import DatasetOp, Episode, OpKind


class DropShortEpisodesParams(BaseModel):
    min_frames: int = Field(default=30, description="Episodes with fewer rows are dropped.")


class DropShortEpisodesOp(DatasetOp):
    name = "drop-short-episodes"
    title = "Drop short episodes"
    description = "Removes episodes shorter than min_frames."
    kind = OpKind.FILTER
    Params = DropShortEpisodesParams

    def keep(self, episode: Episode, params: DropShortEpisodesParams) -> bool:
        return episode.n_rows >= params.min_frames
```

The method depends on `kind`:

| kind | method | returns |
| --- | --- | --- |
| `FILTER` | `keep(episode, params)` (no ctx) | `bool` |
| `MAP` | `map(episode, params, ctx)` | `episode.derive(files, **meta)`, or `None` to drop it |
| `EXPAND` | `expand(episode, params, ctx)` | yields `(output_name, EpisodeOut)`; declare `outputs = (...)` |
| `COMBINE` | `combine(params, ctx)` | the merged episodes |

Optional: `setup(params, ctx)` (load models once), `resources = Resources(gpu=True)`, `produces`. Imports come from `solarian.datasets` (`DatasetOp`, `OpKind`, `Episode`, `EpisodeOut`, `OpContext`, `Resources`). An op useful beyond this project goes in a plugin: `solarian create-plugin op <name>`.

## 2. Run it

```bash
solarian dataset ops                                          # it is listed, or its import error is
solarian dataset run drop-short-episodes recordings/robot/<session> --param min_frames=50
```

`--param key=value` values are YAML and keys can be dotted; `--params-json` takes JSON or `@file`. `run` works on raw sessions. For versioned, reviewable edits, make a computed dataset and apply ops to it:

```bash
solarian dataset new recordings/robot/<session> datasets/pick
solarian dataset apply datasets/pick drop-short-episodes --param min_frames=50
solarian dataset log datasets/pick
```

Do not use `solarian add-op` for now: the template it copies is missing from this Solarian version, so write the file as above.
