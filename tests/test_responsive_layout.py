from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import cast

import pytest
from PySide6.QtCore import QRect
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import QBoxLayout, QScrollArea, QWidget
from pytestqt.qtbot import QtBot

from nlab.app import MainAppWindow
from nlab.ui.ui_external_device_view import Ui_ExternalDeviceView
from nlab.ui.ui_global_view import Ui_GlobalView
from nlab.ui.ui_mca_view import Ui_MCAView
from nlab.ui.ui_psd_view import Ui_PSDView
from nlab.ui.ui_psu_view import Ui_PSUView
from nlab.ui.ui_scope_view import Ui_ScopeView
from nlab.views.responsive_layout import (
    configure_external_layout,
    configure_global_layout,
    configure_mca_layout,
    configure_psd_layout,
    configure_psu_layout,
    configure_scope_layout,
)

ConfigureLayout = Callable[[QWidget, object], None]
configure_mca_720p = partial(configure_mca_layout, screen_height=720)


class _Screen:
    def __init__(self, width: int, height: int) -> None:
        self._geometry = QRect(0, 0, width, height)

    def availableGeometry(self) -> QRect:  # noqa: N802 - mirrors Qt's API
        return self._geometry


class _Window:
    def __init__(self, width: int, height: int) -> None:
        self._screen = _Screen(width, height)
        self.size: tuple[int, int] | None = None

    def screen(self) -> _Screen:
        return self._screen

    def resize(self, width: int, height: int) -> None:
        self.size = width, height

    def isMaximized(self) -> bool:  # noqa: N802 - mirrors Qt's API
        return False

    def isFullScreen(self) -> bool:  # noqa: N802 - mirrors Qt's API
        return False

    def _resize_for_available_screen(self, screen: _Screen | None = None) -> None:
        MainAppWindow._resize_for_available_screen(
            cast(MainAppWindow, self),
            cast(QScreen | None, screen),
        )

    def _resize_after_screen_layout_change(self) -> None:
        self._resize_for_available_screen()


@pytest.mark.parametrize(
    ("available", "expected"),
    (
        ((1280, 680), (1248, 648)),
        ((1920, 1040), (1280, 800)),
    ),
)
def test_initial_window_size_respects_available_desktop(
    available: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    window = _Window(*available)
    MainAppWindow._resize_for_available_screen(cast(MainAppWindow, window))
    assert window.size == expected


def test_window_resizes_when_moved_to_smaller_screen() -> None:
    window = _Window(2560, 1400)
    destination = _Screen(1280, 680)

    MainAppWindow._on_screen_changed(
        cast(MainAppWindow, window),
        cast(QScreen, destination),
    )

    assert window.size == (1248, 648)


@pytest.mark.parametrize(
    ("ui_type", "configure"),
    (
        (Ui_ScopeView, configure_scope_layout),
        (Ui_MCAView, configure_mca_720p),
        (Ui_PSDView, configure_psd_layout),
        (Ui_PSUView, configure_psu_layout),
        (Ui_GlobalView, configure_global_layout),
        (Ui_ExternalDeviceView, configure_external_layout),
    ),
)
def test_primary_views_fit_inside_720p_content_area(
    qtbot: QtBot,
    ui_type: type[object],
    configure: ConfigureLayout,
) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = ui_type()
    ui.setupUi(view)  # type: ignore[attr-defined]
    configure(view, ui)

    minimum = view.minimumSizeHint()
    assert minimum.width() <= 1024
    assert minimum.height() <= 560

    view.resize(1180, 560)
    assert view.layout() is not None
    view.layout().activate()
    assert view.size().width() == 1180
    assert view.size().height() == 560


@pytest.mark.parametrize(
    "name",
    (
        "scopeControlScroll",
        "mcaControlScroll",
        "psuControlScroll",
        "globalContentScroll",
        "externalControlScroll",
    ),
)
def test_dense_control_regions_are_scrollable(
    qtbot: QtBot,
    name: str,
) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    forms = {
        "scopeControlScroll": (Ui_ScopeView, configure_scope_layout),
        "mcaControlScroll": (Ui_MCAView, configure_mca_720p),
        "psuControlScroll": (Ui_PSUView, configure_psu_layout),
        "globalContentScroll": (Ui_GlobalView, configure_global_layout),
        "externalControlScroll": (Ui_ExternalDeviceView, configure_external_layout),
    }
    ui_type, configure = forms[name]
    ui = ui_type()
    ui.setupUi(view)
    configure(view, ui)

    scroll = view.findChild(QScrollArea, name)
    assert scroll is not None
    assert scroll.widgetResizable()


def test_mca_header_groups_move_into_scrollable_controls(qtbot: QtBot) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_MCAView()
    ui.setupUi(view)
    configure_mca_layout(view, ui, screen_height=720)

    assert ui.layoutMca.direction() == QBoxLayout.Direction.TopToBottom
    assert ui.layoutMca.count() == 4
    assert ui.layoutMeasurement.direction() == QBoxLayout.Direction.TopToBottom
    assert ui.layoutMeasurement.count() == 2
    assert ui.controlLayout.indexOf(ui.groupMca) == 0
    assert ui.controlLayout.indexOf(ui.groupMeasurement) == 1
    assert ui.controlLayout.indexOf(ui.groupSignal) == 2
    assert all(
        ui.mainLayout.itemAt(index).layout() is not ui.topRow
        for index in range(ui.mainLayout.count())
    )

    view.resize(1000, 560)
    assert view.layout() is not None
    view.layout().activate()

    scroll = view.findChild(QScrollArea, "mcaControlScroll")
    assert scroll is not None
    assert ui.plotDebug.geometry().y() == scroll.geometry().y()
    assert ui.plotDebug.height() >= 100
    assert ui.plotHistogram.height() >= 140


def test_mca_restores_original_layout_after_returning_to_1080p(qtbot: QtBot) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_MCAView()
    ui.setupUi(view)
    responsive = configure_mca_layout(view, ui, screen_height=720)
    responsive.apply_screen_height(1080)

    assert ui.layoutMca.direction() == QBoxLayout.Direction.LeftToRight
    assert ui.layoutMca.count() == 8
    assert ui.layoutMeasurement.direction() == QBoxLayout.Direction.LeftToRight
    assert ui.layoutMeasurement.count() == 4
    assert ui.mainLayout.itemAt(0).layout() is ui.topRow
    assert ui.topRow.indexOf(ui.groupMca) == 0
    assert ui.topRow.indexOf(ui.groupMeasurement) == 1
    assert view.findChild(QScrollArea, "mcaControlScroll") is None

    responsive.apply_screen_height(720)
    assert ui.controlLayout.indexOf(ui.groupMca) == 0
    assert ui.controlLayout.indexOf(ui.groupMeasurement) == 1
    assert view.findChild(QScrollArea, "mcaControlScroll") is not None


def test_psd_controls_reflow_without_changing_widgets(
    qtbot: QtBot,
) -> None:
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_PSDView()
    ui.setupUi(view)
    configure_psd_layout(view, ui)

    ratio_max_index = ui.controlsLayout.indexOf(ui.spinRatioMax)
    cut_index = ui.controlsLayout.indexOf(ui.spinCut)
    assert ui.controlsLayout.getItemPosition(ratio_max_index)[:2] == (2, 1)
    assert ui.controlsLayout.getItemPosition(cut_index)[:2] == (2, 3)
