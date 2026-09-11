from nlab.views.time_axis import format_duration_ns, time_axis_scale


def test_time_axis_scale_tracks_displayed_span() -> None:
    assert time_axis_scale(999).unit == "ns"
    assert time_axis_scale(1_000).unit == "\N{MICRO SIGN}s"
    assert time_axis_scale(1_000_000).unit == "ms"
    assert time_axis_scale(1_000_000_000).unit == "s"


def test_duration_format_uses_selected_physical_unit() -> None:
    assert format_duration_ns(512) == "512 ns"
    assert format_duration_ns(2_048) == "2.048 \N{MICRO SIGN}s"
