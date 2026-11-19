---
name: deploy
description: Run this Solarian project across machines - a deployment profile in conf/deployment with the relay address, ports and device paths, `-d` to select it, CurveZMQ keys from `solarian keygen`, per-role installs, and Solarian Base for cloud deploys, training and GPU endpoints. Use when the robot, relay and controller should run on different hosts or in the cloud.
---

# Deploy

Docs: `solarian docs show project-configs` (Deployments), `servers`, `installation`, and `solarian-base/quickstart` for the cloud.

## 1. A profile per topology

One file describes where everything runs; every machine uses the same profile.

```yaml
# conf/deployment/pi.yaml
extends: localhost                 # optional: merge over another profile
connection:
  ip: 192.168.1.20                 # the relay, as every node reaches it
  action_sender_port: 5555
  action_receiver_port: 5556
  state_sender_port: 5557
  state_receiver_port: 5558
  clock_sync_port: 5559
actuator_port_arm: /dev/ttyUSB0    # device paths as seen from the robot host
camera_port_wrist_camera: '0'
```

The rest of `conf/` reads these as `${deployment.<key>}`; keep host-specific values (IPs, ports, device paths, server URLs, checkpoints, tokens) out of every other file. Deployment values cannot be overridden on the command line; edit the profile or switch to another.

## 2. Select it

`-d pi` on each command, or `export SOLARIAN_DEPLOYMENT=pi` once per machine. `localhost` is the default; an unknown name fails and lists the profiles.

## 3. Install per role and run

```bash
pip install -e ".[robot]"           # robot host; [server], [controller], or [all] on one machine
solarian run-server -d pi           # relay host
solarian run-robot -d pi            # robot host
solarian run-controller -d pi       # operator machine
solarian sync-clock -d pi           # optional: align clocks with the relay
```

## 4. Encrypt the link

`solarian keygen` prints, without writing anything, `curve_allowed_client_keys` for `conf/server.yaml` and `connection:` key blocks for the relay, robot and controller. Paste them into the profiles. A relay on a public address must set `curve_allowed_client_keys`.

## 5. Solarian Base (cloud)

Needs the `solarian-base` plugin (`pip install -e ".[base]"`). Every action takes `--json`.

```bash
solarian base whoami --json         # if not logged in, ask the user to run `solarian base login`
solarian base link                  # link this folder to a Base project (--create to make one)
solarian base deploy                # upload the project
solarian base env add KEY VALUE     # secrets for cloud jobs
solarian base train <name>          # a conf/training run on a cloud GPU
solarian base endpoints start <service>
```

Base writes `conf/deployment/solarian_base.yaml`, which extends a robot profile with endpoint URLs and a token. Never create or paste Base tokens yourself.
