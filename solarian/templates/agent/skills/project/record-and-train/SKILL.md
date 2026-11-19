---
name: record-and-train
description: Record teleoperated demonstrations with `solarian record`, curate them, export or train a policy (UMI, diffusion, ACT) with `solarian train` and a conf/training yaml, then serve the checkpoint back to a skill on the robot. Use when the user wants the robot to learn a task from data.
---

# Record and train

Docs: `solarian docs show recording`, `datasets`, `training`, `policies`, and the tutorial `tutorials/training` (`collect`, `datasets`, `train`, `run`).

## 1. Record

`record` does not start the robot or relay; run them first.

```bash
solarian run-server                       # background
solarian run-robot                        # add --sim to rehearse
solarian record num_episodes=50 episode_time_s=20
```

- Config: `conf/commands/record.yaml`. It extends `controller.yaml` (`defaults: [/controller@_global_, _self_]`, keep `_self_` last), so `recording.robot` and `recording.controller` there choose which side records.
- While recording, `r` ends an episode early and delete re-records it.
- Sessions land in `recordings/robot/<NNNN>_<hash>_<name>/` (and `recordings/controller/...`): `session.json`, `manifest.jsonl`, `episodes/NNNNNN/{episode.parquet, video/<module>.<field>.mp4, ...}`.

## 2. Curate

Drop, split or enrich episodes with dataset ops; see the `add-dataset-op` skill and `solarian dataset ops`.

## 3. Export (when the trainer wants another format)

```bash
solarian export-dataset recordings/robot/<session>        # LeRobot v2.1, into <session>/export/lerobot
```

## 4. Train

Trainers are plugins registered under `solarian.trainers` (e.g. `solarian install policy-diffusion`, or the UMI trainer). The run is a yaml:

```yaml
# conf/training/pick.yaml
trainer: umi                 # the solarian.trainers entry-point name
job_name: pick
output_dir: null             # null: models/<job_name>
modal: null                  # or Modal settings to train on a cloud GPU
# ... the trainer's own keys; see its docs or its seeded conf/training/*.yaml
```

```bash
solarian train --conf conf/training/pick          # --conf is required
```

On Solarian Base (cloud GPUs): `solarian base train pick` (from the `solarian-base` plugin; see `solarian base --help`).

## 5. Run the checkpoint

1. Point the policy skill's config at it: in `conf/skills/<skill>.yaml`, `server.checkpoint: ${oc.select:deployment.<skill>_policy_checkpoint,null}`, with the path in the deployment profile.
2. `solarian run-service <skill>` (or `-c <checkpoint>`).
3. In `{{ package }}/controller.py`, add `Tool(<PolicySkill>, {"config": "<skill>.yaml", ...}, agent_enabled=True)` to `tools`.
4. Try it in `--sim` first, then on the robot when the user asks.
