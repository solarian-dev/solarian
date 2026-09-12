from types import SimpleNamespace

from solarian.sim.runtime import collect_tcp_frames


def _robot(frame: str, prefix: str = "right/"):
    class _Arm:
        ik_backend = SimpleNamespace(frame=frame, frame_type="site")

    return SimpleNamespace(
        scene=SimpleNamespace(
            show_frames=True,
            attachments=[SimpleNamespace(module="right_arm", prefix=prefix)],
        ),
        modules={"right_arm": _Arm()},
    )


def test_collect_tcp_frames_uses_composed_name_when_frame_is_prefixed():
    assert collect_tcp_frames(_robot("right_gripper/tcp")) == [
        ("right_gripper/tcp", "site")
    ]


def test_collect_tcp_frames_prefixes_bare_frame():
    assert collect_tcp_frames(_robot("flange")) == [("right/flange", "site")]
