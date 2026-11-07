"""Status board layout, and partial observations keeping their last reading."""

from rich.console import Console

from solarian.core.data import PosData
from solarian.core.observation import ModuleObs, Observation, SensorField
from solarian.utils.log_buffer import LogBuffer
from solarian.utils.status_board import StatusBoard


def _board():
    board = object.__new__(StatusBoard)
    board.data = {}
    board.table_title = "Status"
    board._log_buf = LogBuffer(maxlen=1)
    board._console = Console(width=100, force_terminal=True)
    return board


def _divider(text: str) -> int:
    for line in text.splitlines():
        if "┃" in line:
            return line.index("┃")
    raise AssertionError(text)


def test_the_value_column_stays_put_when_its_content_grows():
    board = _board()
    board.data = {"right_gripper aperture cmd (mm)": "85.01"}
    with board._console.capture() as narrow:
        board._console.print(board.render())

    board.data = {"right_gripper aperture cmd (mm)": "x" * 400}
    with board._console.capture() as wide:
        board._console.print(board.render())

    assert _divider(narrow.get()) == _divider(wide.get())
    # The label is wider than the state column, so it folds and the value
    # stays on the first of those lines.
    folded = [line for line in narrow.get().splitlines() if "(mm)" in line]
    assert folded and "85.01" not in folded[0]


def test_a_later_partial_observation_keeps_modules_it_does_not_mention():
    previous = (Observation._active_subclass, ModuleObs._active_subclass)

    class _ArmObs(ModuleObs):
        pos = SensorField(PosData(path=("arm", "joint", "pos")))

    class _GripObs(ModuleObs):
        pos = SensorField(PosData(path=("gripper", "grip", "pos")))

    class _Obs(Observation):
        arm = _ArmObs()
        gripper = _GripObs()

    try:
        board = _board()
        board._store("Observation", _Obs({
            "arm": {"pos": {"value": 1.0, "timestamp": 10.0}},
            "gripper": {"pos": {"value": 2.0, "timestamp": 10.0}},
        }))
        board._store("Observation", _Obs({
            "gripper": {"pos": {"value": 3.0, "timestamp": 11.0}},
        }))

        shown = board.data["Observation"]
        assert shown["arm"]["pos"].value == 1.0
        assert shown["gripper"]["pos"].value == 3.0
    finally:
        Observation._active_subclass, ModuleObs._active_subclass = previous
