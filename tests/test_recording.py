"""Episode recording: the schema walk, the frame grid, and a sealed episode."""

from __future__ import annotations

import json
import threading
import time
from multiprocessing.managers import SharedMemoryManager
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from solarian.core.action import Action, ActionType, ActuatorField, ModuleAction
from solarian.core.clock_sync import synced_time
from solarian.core.controller import Controller
from solarian.core.controller_gui import ControllerGUI
from solarian.core.data import ImageData, PosData, TemperatureData
from solarian.core.event import Event, EventType
from solarian.core.observation import ModuleObs, Observation, SensorField
from solarian.core.sensor_bus import SensorBus
from solarian.core.recording.export import export_lerobot
from solarian.core.recording.recorder import SessionRecorder
from solarian.core.recording.schema import build_schema
from solarian.core.recording.session import CameraMeta, read_manifest
from solarian.core.recording.timeline import frame_index_for, repeats_for
from solarian.core.recording.video_writer import VideoWriter

ARCHIVE_SHAPE = (48, 64, 3)
FPS = 10.0


@pytest.fixture
def schema_classes():
    """A two-module robot: one arm with a temperature, one camera.

    Defining these inside a fixture and restoring the registries afterwards is
    what keeps the rest of the suite's active ``Observation`` subclass intact —
    ``__init_subclass__`` makes class definition a global side effect.
    """
    previous = (
        Observation._active_subclass,
        ModuleObs._active_subclass,
        Action._active_subclass,
        ModuleAction._active_subclass,
    )

    class _ArmObs(ModuleObs):
        shoulder = SensorField(PosData(path=("arm", "shoulder", "pos")))
        shoulder_temp = SensorField(TemperatureData(path=("arm", "shoulder", "temp")))

    class _CamObs(ModuleObs):
        image = SensorField(
            ImageData(path=("cam", "sensor", "rgb"), shape=ARCHIVE_SHAPE)
        )

    class _Obs(Observation):
        arm = _ArmObs()
        cam = _CamObs()

    class _ArmAction(ModuleAction):
        shoulder = ActuatorField(PosData(path=("arm", "shoulder", "pos")))

    class _Action(Action):
        arm = _ArmAction()

    yield _Obs, _Action

    (
        Observation._active_subclass,
        ModuleObs._active_subclass,
        Action._active_subclass,
        ModuleAction._active_subclass,
    ) = previous


def _observation(obs_cls, *, t: float, shoulder: float, with_image: bool):
    data = {
        "arm": {
            "shoulder": {"value": shoulder, "timestamp": t},
            "shoulder_temp": {"value": 41.0, "timestamp": t},
        }
    }
    if with_image:
        data["cam"] = {
            "image": {
                "value": np.zeros(ARCHIVE_SHAPE, dtype=np.uint8),
                "timestamp": t,
            }
        }
    return obs_cls(data)


# --- the frame grid --------------------------------------------------------


def test_a_timestamp_on_a_slot_boundary_lands_in_that_slot():
    """Float error must not push an exactly-on-time frame into the slot before."""
    assert frame_index_for(100.0 + 3 / 30.0, 100.0, 30.0) == 3


def test_a_frame_before_the_episode_started_is_not_encoded():
    n_repeats, cursor = repeats_for(99.9, 100.0, 30.0, next_index=0)

    assert (n_repeats, cursor) == (0, 0)


def test_a_late_frame_fills_every_slot_it_skipped():
    """Repeating is what keeps the mp4 in step with wall clock."""
    n_repeats, cursor = repeats_for(100.0 + 2 / 30.0, 100.0, 30.0, next_index=0)

    assert (n_repeats, cursor) == (3, 3)


def test_a_second_frame_inside_one_slot_is_dropped():
    n_repeats, cursor = repeats_for(100.001, 100.0, 30.0, next_index=1)

    assert (n_repeats, cursor) == (0, 1)


def test_recorder_and_encoder_agree_on_the_slot_for_the_same_capture():
    """The invariant the whole design rests on: no index crosses a process."""
    start, capture = 100.0, 100.0 + 7.4 / 30.0

    _, cursor = repeats_for(capture, start, 30.0, next_index=0)

    assert cursor - 1 == frame_index_for(capture, start, 30.0)


# --- the schema walk -------------------------------------------------------


def test_schema_is_derived_from_the_project_not_hand_written(schema_classes):
    schema = build_schema()

    assert set(schema.column_names) == {
        "t",
        "wall_ts",
        "obs.arm.ts",
        "obs.arm.shoulder",
        "obs.arm.shoulder_temp",
        "obs.cam.ts",
        "cam.cam.image.frame_index",
        "cam.cam.image.capture_ts",
        "act.arm.shoulder",
    }


def test_an_image_field_becomes_a_camera_rather_than_a_column(schema_classes):
    schema = build_schema()

    assert [c.name for c in schema.cameras] == ["cam.image"]


def test_threshold_bands_survive_into_the_schema_document(schema_classes):
    """Flattened to a float vector, a servo temperature reads as a joint angle."""
    document = build_schema().to_json()

    assert document["obs.arm.shoulder_temp"]["data_class"] == "TemperatureData"
    assert document["obs.arm.shoulder_temp"]["high"] == 60.0
    assert "high" not in document["obs.arm.shoulder"]


def test_timestamps_stay_out_of_the_statistics(schema_classes):
    """float32 cannot resolve a millisecond inside a Unix epoch."""
    assert build_schema().numeric_columns() == (
        "obs.arm.shoulder",
        "obs.arm.shoulder_temp",
        "act.arm.shoulder",
    )


# --- a sealed episode ------------------------------------------------------


@pytest.fixture
def clock(monkeypatch):
    """Drive the recorder's clock from the test.

    ``t`` and the episode's duration come from the wall clock, so a loop that
    writes twenty rows as fast as it can produces an episode lasting
    microseconds. Holding the clock is what lets a test say "rows 10 ms apart"
    and mean it.
    """
    state = {"now": 1_000_000.0}
    monkeypatch.setattr(
        "solarian.core.recording.recorder.synced_time", lambda: state["now"]
    )
    return state


@pytest.fixture
def recorder(tmp_path, schema_classes, clock):
    with SharedMemoryManager() as shm_manager:
        writer = VideoWriter(
            name="cam.image", fps=FPS, shape=ARCHIVE_SHAPE, preset="ultrafast"
        )
        sink = writer.open(shm_manager)
        session = SessionRecorder(
            root=tmp_path,
            session="testing",
            fps=FPS,
            writers={"cam.image": writer},
            robot_name="TestRobot",
            cameras={
                "cam.image": CameraMeta(
                    width=ARCHIVE_SHAPE[1],
                    height=ARCHIVE_SHAPE[0],
                    source_fps=FPS,
                )
            },
            video_options={"codec": "libx264", "crf": 20},
            task="put the cup down",
        )
        yield session, sink
        session.close()


def _record_one_episode(session, sink, obs_cls, action_cls, clock, *, n_rows=20):
    session.start_episode()
    start = session._start_ts
    frame = np.zeros(ARCHIVE_SHAPE, dtype=np.uint8)
    for i in range(n_rows):
        t = clock["now"] = start + i * 0.01
        # One capture every five rows, so the table is faster than the video
        # exactly as a real arm is faster than a camera.
        with_image = i % 5 == 0
        if with_image:
            sink.write_frame(frame, t)
        session.note_action(
            action_cls(
                action_type=ActionType.MODULE_ACTION,
                data={
                    "arm": {
                        "action_type": int(ActionType.SERVO_J),
                        "data": {"shoulder": {"value": float(i), "timestamp": t}},
                    }
                },
            )
        )
        session.write_row(_observation(obs_cls, t=t, shoulder=float(i),
                                       with_image=with_image))
    return session.end_episode()


def test_a_sealed_episode_has_a_table_a_video_and_a_ledger_line(
    recorder, schema_classes, clock
):
    session, sink = recorder
    obs_cls, action_cls = schema_classes

    meta = _record_one_episode(session, sink, obs_cls, action_cls, clock)

    episode_dir = session.paths.episode_dir(0)
    assert (episode_dir / "episode.parquet").is_file()
    assert (episode_dir / "episode.json").is_file()
    assert session.paths.video_path(0, "cam.image").is_file()
    assert [e["episode"] for e in read_manifest(session.paths)] == [0]
    assert meta.n_rows == 20


def test_every_module_update_becomes_a_row(recorder, schema_classes, clock):
    """The table samples at the union of every rate, not at one chosen grid."""
    session, sink = recorder
    obs_cls, action_cls = schema_classes

    _record_one_episode(session, sink, obs_cls, action_cls, clock)

    table = pq.read_table(session.paths.episode_dir(0) / "episode.parquet")
    assert table.num_rows == 20
    assert table.column("obs.arm.shoulder").to_pylist()[:3] == [0.0, 1.0, 2.0]


def test_a_signal_that_did_not_advance_carries_its_last_value_forward(
    recorder, schema_classes, clock
):
    """A row with holes is not a sample of the robot's state."""
    session, sink = recorder
    obs_cls, action_cls = schema_classes

    _record_one_episode(session, sink, obs_cls, action_cls, clock)

    table = pq.read_table(session.paths.episode_dir(0) / "episode.parquet")
    frame_indices = table.column("cam.cam.image.frame_index").to_pylist()
    # Rows run at 100 Hz against a 10 fps grid, so ten consecutive rows point
    # at the same frame even though only every fifth row carried an image.
    assert frame_indices[:10] == [0] * 10
    assert frame_indices[10:] == [1] * 10


def test_the_action_recorded_is_the_one_the_robot_received(recorder, schema_classes, clock):
    session, sink = recorder
    obs_cls, action_cls = schema_classes

    _record_one_episode(session, sink, obs_cls, action_cls, clock)

    table = pq.read_table(session.paths.episode_dir(0) / "episode.parquet")
    assert table.column("act.arm.shoulder").to_pylist()[:3] == [0.0, 1.0, 2.0]


def test_episode_metadata_reports_quality_without_opening_the_data(
    recorder, schema_classes, clock
):
    session, sink = recorder
    obs_cls, action_cls = schema_classes

    _record_one_episode(session, sink, obs_cls, action_cls, clock)

    document = json.loads(
        (session.paths.episode_dir(0) / "episode.json").read_text()
    )
    assert document["status"] == "ok"
    assert document["task"] == "put the cup down"
    assert document["n_frames"]["cam.image"] > 0
    assert document["stats"]["obs.arm.shoulder"]["max"] == 19.0
    assert document["stats"]["obs.arm.shoulder"]["count"] == 20


def test_the_archive_decodes_to_the_frame_count_the_metadata_claims(
    recorder, schema_classes, clock
):
    """Frame indices in the table are only meaningful if the mp4 really has them."""
    import av

    session, sink = recorder
    obs_cls, action_cls = schema_classes

    meta = _record_one_episode(session, sink, obs_cls, action_cls, clock)

    with av.open(str(session.paths.video_path(0, "cam.image"))) as container:
        decoded = sum(1 for _ in container.decode(video=0))
    assert decoded == meta.n_frames["cam.image"]


def test_an_episode_seals_while_the_camera_is_still_feeding_the_encoder(
    recorder, schema_classes, clock
):
    """The camera never stops, so the encoder cannot wait for an empty queue.

    A bus pushes captures whether or not an episode is open. If the encoder
    stopped only when it ran out of frames, an encoder even slightly slower
    than the camera would never stop at all.
    """
    session, sink = recorder
    obs_cls, action_cls = schema_classes
    frame = np.zeros(ARCHIVE_SHAPE, dtype=np.uint8)

    _record_one_episode(session, sink, obs_cls, action_cls, clock, n_rows=5)
    session.start_episode()
    feeding = True

    def keep_capturing():
        while feeding:
            clock["now"] += 1.0 / FPS
            sink.write_frame(frame, clock["now"])
            time.sleep(0.002)

    pump = threading.Thread(target=keep_capturing, daemon=True)
    pump.start()
    try:
        time.sleep(0.2)
        started = time.monotonic()
        meta = session.end_episode()
        sealed_in = time.monotonic() - started
    finally:
        feeding = False
        pump.join(timeout=5.0)

    assert meta is not None and meta.status == "ok"
    # Waiting for the queue to run dry would have cost the whole seal timeout
    # and left the file without its tail.
    assert sealed_in < 5.0
    # Frames captured after the episode ended belong to no episode, so the
    # grid stops at the stop instant rather than running on with the camera.
    assert meta.n_frames["cam.image"] <= round((meta.end_ts - meta.start_ts) * FPS) + 1


def test_the_archive_keeps_its_index_at_the_front(recorder, schema_classes, clock):
    """A plain mp4 indexes itself in a trailer, so one that never closed is not
    a short video but no video at all. Fragmenting it makes a killed episode
    decode up to wherever it stopped."""
    session, sink = recorder
    obs_cls, action_cls = schema_classes

    _record_one_episode(session, sink, obs_cls, action_cls, clock)

    data = session.paths.video_path(0, "cam.image").read_bytes()
    boxes = []
    offset = 0
    while offset < len(data) - 8:
        size = int.from_bytes(data[offset : offset + 4], "big")
        boxes.append(data[offset + 4 : offset + 8].decode("latin-1"))
        if size == 0:
            break
        offset += size
    assert "moof" in boxes
    assert boxes.index("moov") < boxes.index("mdat")


def test_a_video_that_never_sealed_is_not_labelled_a_good_episode(
    recorder, schema_classes, clock, monkeypatch
):
    """An unsealed mp4 is missing its tail; the metadata has to say so."""
    session, sink = recorder
    obs_cls, action_cls = schema_classes
    monkeypatch.setattr(
        session.writers["cam.image"], "end_episode", lambda stop_time: None
    )

    meta = _record_one_episode(session, sink, obs_cls, action_cls, clock, n_rows=5)

    assert meta.status == "video_failed"
    assert meta.n_frames["cam.image"] == 0
    assert read_manifest(session.paths)[0]["status"] == "video_failed"


def test_session_metadata_is_written_once_at_session_start(recorder):
    session, _ = recorder

    document = json.loads(session.paths.session_json.read_text())

    assert document["robot"] == "TestRobot"
    assert document["fps"] == FPS
    assert "obs.arm.shoulder" in document["schema"]


def test_a_dropped_episode_leaves_nothing_behind(recorder, schema_classes):
    """Re-recording a bad take must not shift every later episode's index."""
    session, sink = recorder
    obs_cls, action_cls = schema_classes

    session.start_episode()
    session.write_row(_observation(obs_cls, t=session._start_ts, shoulder=1.0,
                                   with_image=True))
    session.drop_episode()

    assert not session.paths.episode_dir(0).exists()
    assert read_manifest(session.paths) == []
    assert session.episode_count == 0


def test_a_second_episode_records_after_the_first_is_sealed(
    recorder, schema_classes, clock
):
    """No pause between takes: sealing one episode sets up the next."""
    session, sink = recorder
    obs_cls, action_cls = schema_classes

    _record_one_episode(session, sink, obs_cls, action_cls, clock, n_rows=5)
    _record_one_episode(session, sink, obs_cls, action_cls, clock, n_rows=5)

    assert [e["episode"] for e in read_manifest(session.paths)] == [0, 1]
    assert session.paths.video_path(1, "cam.image").is_file()


def test_naming_a_session_moves_later_episodes_into_its_own_directory(
    recorder, schema_classes, clock
):
    """One `solarian record` run is one directory, named by the operator."""
    session, sink = recorder
    obs_cls, action_cls = schema_classes
    _record_one_episode(session, sink, obs_cls, action_cls, clock, n_rows=5)

    session.open_session("cup_stacking")
    _record_one_episode(session, sink, obs_cls, action_cls, clock, n_rows=5)

    assert session.paths.root.name.startswith("cup_stacking__")
    assert session.episode_count == 1
    assert [e["episode"] for e in read_manifest(session.paths)] == [0]


def test_reasking_for_the_open_session_does_not_roll_the_directory(recorder):
    """RECORD_START is retried until the robot answers, and carries the name."""
    session, _ = recorder
    before = session.paths.root

    session.open_session("testing")

    assert session.paths.root == before


def test_rows_written_outside_an_episode_are_discarded(recorder, schema_classes):
    session, sink = recorder
    obs_cls, _ = schema_classes

    session.write_row(_observation(obs_cls, t=1.0, shoulder=1.0, with_image=True))

    assert session.end_episode() is None
    assert read_manifest(session.paths) == []


# --- export -----------------------------------------------------------------


@pytest.fixture
def exported(tmp_path, recorder, schema_classes, clock):
    """Two recorded episodes, converted to LeRobot v2.1."""
    session, sink = recorder
    obs_cls, action_cls = schema_classes
    _record_one_episode(session, sink, obs_cls, action_cls, clock, n_rows=20)
    _record_one_episode(session, sink, obs_cls, action_cls, clock, n_rows=20)

    out = tmp_path / "lerobot"
    stats = export_lerobot(session.paths.root, out, remap=False)
    return out, stats


def test_export_writes_the_layout_a_lerobot_reader_globs_for(exported):
    out, stats = exported

    assert (out / "meta" / "info.json").is_file()
    assert (out / "meta" / "episodes.jsonl").is_file()
    assert (out / "meta" / "tasks.jsonl").is_file()
    assert (out / "data" / "chunk-000" / "episode_000001.parquet").is_file()
    assert (
        out
        / "videos"
        / "chunk-000"
        / "observation.images.cam.image"
        / "episode_000001.mp4"
    ).is_file()
    assert stats.n_episodes == 2


def test_export_declares_the_codebase_version_its_readers_check(exported):
    out, _ = exported

    info = json.loads((out / "meta" / "info.json").read_text())

    assert info["codebase_version"] == "v2.1"
    assert info["fps"] == FPS
    assert info["robot_type"] == "TestRobot"
    assert info["features"]["observation.state"]["names"] == [
        "obs.arm.shoulder",
        "obs.arm.shoulder_temp",
    ]
    assert info["features"]["action"]["names"] == ["act.arm.shoulder"]


def test_export_resamples_the_native_rate_table_onto_one_grid(exported):
    """The session runs at whatever each module does; LeRobot wants row i at i/fps."""
    out, _ = exported

    table = pq.read_table(out / "data" / "chunk-000" / "episode_000000.parquet")

    # 20 rows at 100 Hz is 0.19 s, which is two slots of a 10 fps grid.
    assert table.num_rows == 2
    assert table.column("timestamp").to_pylist() == [0.0, pytest.approx(0.1)]
    assert table.column("frame_index").to_pylist() == [0, 1]
    assert table.column("index").to_pylist() == [0, 1]
    assert table.column("episode_index").to_pylist() == [0, 0]


def test_export_holds_the_last_sample_rather_than_inventing_one(exported):
    """Sample-and-hold is the only resampling that cannot fabricate a reading."""
    out, _ = exported

    table = pq.read_table(out / "data" / "chunk-000" / "episode_000000.parquet")
    state = [list(row) for row in table.column("observation.state").to_pylist()]

    # Row 10 is the last one at or before t=0.1 with rows every 0.01 s.
    assert [row[0] for row in state] == [0.0, 10.0]


def test_the_global_index_runs_across_episodes(exported):
    """LeRobot's `index` is a dataset-wide row number, not a per-episode one."""
    out, _ = exported

    second = pq.read_table(out / "data" / "chunk-000" / "episode_000001.parquet")

    assert second.column("index").to_pylist() == [2, 3]
    assert second.column("episode_index").to_pylist() == [1, 1]


def test_the_exported_video_has_one_frame_per_table_row(exported):
    """`frame_index` only means anything if row i really is video frame i."""
    import av

    out, _ = exported
    path = (
        out / "videos" / "chunk-000" / "observation.images.cam.image"
        / "episode_000000.mp4"
    )

    with av.open(str(path)) as container:
        decoded = sum(1 for _ in container.decode(video=0))

    assert decoded == 2


def test_exporting_the_same_task_twice_reuses_its_index(exported):
    out, _ = exported

    tasks = [json.loads(line) for line in
             (out / "meta" / "tasks.jsonl").read_text().splitlines()]

    assert tasks == [{"task_index": 0, "task": "put the cup down"}]


def test_an_unsealed_episode_is_not_exported(tmp_path, recorder, schema_classes, clock):
    """Its table has no final row group and its mp4 has no footer."""
    session, sink = recorder
    obs_cls, action_cls = schema_classes
    _record_one_episode(session, sink, obs_cls, action_cls, clock, n_rows=20)
    session.start_episode()
    session.write_row(_observation(obs_cls, t=session._start_ts, shoulder=1.0,
                                   with_image=True))

    stats = export_lerobot(session.paths.root, tmp_path / "out", remap=False)

    assert stats.n_episodes == 1


# --- the archive tap on the camera bus --------------------------------------


class _FakeCameraBus(SensorBus):
    """A camera bus with the hardware taken out, to exercise the archive tap."""

    is_connected = True

    def connect(self):
        pass

    def disconnect(self):
        pass

    def read(self, data_type, component_names, normalize=True):
        return {}


def _camera_cfg(**overrides):
    cfg = SimpleNamespace(archive_width=None, archive_height=None)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_the_archive_keeps_the_frame_the_sensor_produced():
    """Unset archive_* records the capture, not what the policy is fed."""
    bus = _FakeCameraBus(port="cam0", components={})
    captured = []
    bus.archive_sinks["camera"] = SimpleNamespace(
        write_frame=lambda frame, ts: captured.append(frame.shape)
    )

    bus.archive_capture("camera", np.zeros((1520, 2028, 3), np.uint8), _camera_cfg())

    assert captured == [(1520, 2028, 3)]


def test_an_archive_resolution_downscales_the_capture():
    """The knob for encoder CPU on a Pi, applied before the frame is queued."""
    bus = _FakeCameraBus(port="cam0", components={})
    captured = []
    bus.archive_sinks["camera"] = SimpleNamespace(
        write_frame=lambda frame, ts: captured.append(frame.shape)
    )

    bus.archive_capture(
        "camera",
        np.zeros((1520, 2028, 3), np.uint8),
        _camera_cfg(archive_width=1332, archive_height=1000),
    )

    assert captured == [(1000, 1332, 3)]


def test_the_observation_and_the_archive_name_the_same_instant():
    """Two stamps would put the table's frame index on the wrong frame."""
    bus = _FakeCameraBus(port="cam0", components={})
    stamps = []
    bus.archive_sinks["camera"] = SimpleNamespace(
        write_frame=lambda frame, ts: stamps.append(ts)
    )

    returned = bus.archive_capture(
        "camera", np.zeros((8, 8, 3), np.uint8), _camera_cfg()
    )

    assert stamps == [returned] == [bus.capture_timestamps["camera"]]


def test_receive_latency_is_subtracted_from_the_capture_stamp():
    """A frame is older than the moment the bus finished reading it."""
    bus = _FakeCameraBus(port="cam0", components={})
    bus.capture_latency = 0.25

    before = synced_time()
    stamp = bus.archive_capture("camera", np.zeros((8, 8, 3), np.uint8), _camera_cfg())

    assert stamp <= before - 0.25 + 1e-3


# --- asking the robot for a boundary ---------------------------------------


class _RecordingCtrl(Controller):
    """Only the recording-request machinery: no zmq, no modules, no skills."""

    def __init__(self):
        self.cfg = SimpleNamespace(task="put the cup down")
        self.events = {"rerecord_episode": False}
        self._init_recording()


def test_a_boundary_is_resent_until_the_robot_echoes_it():
    """Every hop of the action link is CONFLATE, so one ask is not a boundary."""
    ctrl = _RecordingCtrl()
    ctrl.start_recording()

    assert ctrl._retry_record_request() is not None
    assert ctrl._retry_record_request() is not None

    ctrl.recording_state["request"] = ctrl._record_request_id
    assert ctrl._retry_record_request() is None


def test_a_newer_boundary_replaces_one_the_robot_never_answered():
    """Queued instead of replaced, a stale START would reopen the sealed episode."""
    ctrl = _RecordingCtrl()
    ctrl.start_recording()
    ctrl.stop_recording()

    assert ctrl._retry_record_request().action_type == ActionType.RECORD_STOP


def test_a_boundary_the_robot_never_answers_is_given_up_on():
    ctrl = _RecordingCtrl()
    ctrl.start_recording()
    ctrl._record_request_deadline = time.monotonic() - 1.0

    assert ctrl._retry_record_request() is None


# --- the operator's keys ----------------------------------------------------


def _press(ctrl: Controller, key: str) -> None:
    ctrl.toggle_recording_on_r(Event(EventType.KEY_DOWN, data={"key": key}))


def test_r_opens_an_episode_and_then_seals_it():
    ctrl = _RecordingCtrl()

    _press(ctrl, "r")
    assert ctrl._record_request.action_type == ActionType.RECORD_START

    # Only the robot flips this, so a toggle needs it echoed back first.
    ctrl.recording_state["active"] = True
    _press(ctrl, "r")
    assert ctrl._record_request.action_type == ActionType.RECORD_STOP


@pytest.mark.parametrize("key", ["delete", "backspace"])
def test_delete_throws_the_episode_away_under_either_name(key):
    """A laptop's delete key is backspace; only full-size boards send delete."""
    ctrl = _RecordingCtrl()
    ctrl.start_recording()

    _press(ctrl, key)

    assert ctrl._record_request.action_type == ActionType.RECORD_DROP
    assert ctrl.events["rerecord_episode"] is True
    assert ctrl.episode_over.is_set()


def test_the_gripper_keys_do_not_touch_recording():
    """``x`` closes the gripper now, so it must not discard the take too."""
    ctrl = _RecordingCtrl()
    ctrl.start_recording()
    opened = ctrl._record_request

    _press(ctrl, "x")
    _press(ctrl, "z")

    assert ctrl._record_request is opened


class _FakePygame:
    """Just the name table: pygame reports named keys as words, not chars."""

    NAMES = {1: "a", 2: "space", 3: "delete", 4: "backspace", 5: "left shift", 6: "f1"}

    class key:  # noqa: N801 - mirrors pygame's own lowercase module name
        @staticmethod
        def name(code: int) -> str:
            return _FakePygame.NAMES[code]


@pytest.mark.parametrize(
    "code, expected",
    [(1, "a"), (2, "space"), (3, "delete"), (4, "backspace"), (5, None), (6, None)],
)
def test_the_window_passes_through_the_named_keys_bindings_use(code, expected):
    """Dropping every multi-character name loses space and delete with it."""
    gui = object.__new__(ControllerGUI)
    gui._pygame = _FakePygame

    assert gui._event_char(code) == expected
