"""Layout helpers for the pygame dashboard. No window."""

from solarian.core.controller_gui import clip_text, split_temperature_rows


def _width(text: str) -> tuple[int, int]:
    return (len(text) * 10, 16)


def test_clip_text_keeps_a_line_that_fits():
    assert clip_text(_width, "home", 40) == "home"


def test_clip_text_ends_a_long_line_with_an_ellipsis():
    # "place object at the TAPNext point" is 34 chars, 340px. Budget is 200.
    clipped = clip_text(_width, "place object at the TAPNext point", 200)
    assert clipped.endswith("…")
    assert _width(clipped)[0] <= 200
    assert clipped.startswith("place object")


def test_clip_text_gives_up_when_even_the_ellipsis_does_not_fit():
    assert clip_text(_width, "home", 0) == ""
    assert clip_text(_width, "home", 5) == ""


def test_temperature_fields_leave_the_position_list_in_order():
    rows = [
        ("shoulder_pan", "-0.143"),
        ("shoulder_pan_temp", "29", (240, 80, 80)),
        ("elbow_flex", "45.528"),
        ("elbow_flex_temp", "37"),
    ]
    main, temps = split_temperature_rows(rows)
    assert main == [("shoulder_pan", "-0.143"), ("elbow_flex", "45.528")]
    assert temps == [
        ("shoulder_pan_temp", "29", (240, 80, 80)),
        ("elbow_flex_temp", "37"),
    ]
