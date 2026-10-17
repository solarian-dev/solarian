from __future__ import annotations

from pydantic import BaseModel, Field

from solarian.datasets.op import DatasetOp, Episode, OpContext, OpKind


class DeleteEpisodesParams(BaseModel):
    episodes: list[int] = Field(default_factory=list, description="Indices of the episodes to drop.")
    missing_targets: str | None = Field(
        default=None,
        description="Also drop every episode without targets for this camera, e.g. camera0.rgb.",
    )


class DeleteEpisodesOp(DatasetOp):
    name = "delete-episodes"
    title = "Delete episodes"
    description = """
    Drop episodes, picked by index (for instance after watching them) or because
    they lack targets for a camera. The survivors are renumbered from 0 in their
    order; nothing but their episode.json is stored again.
    """
    kind = OpKind.FILTER
    Params = DeleteEpisodesParams

    def setup(self, params: DeleteEpisodesParams, ctx: OpContext) -> None:
        unknown = sorted(set(params.episodes) - set(ctx.session.episodes()))
        if unknown:
            raise ValueError(f"Not episodes of {ctx.session.path.name}: {unknown}")
        if params.missing_targets is not None and params.missing_targets not in ctx.session.cameras:
            raise ValueError(f"{params.missing_targets!r} is not a camera: {ctx.session.cameras}")

    def keep(self, episode: Episode, params: DeleteEpisodesParams) -> bool:
        if episode.index in params.episodes:
            return False
        return params.missing_targets is None or episode.has_targets(params.missing_targets)
