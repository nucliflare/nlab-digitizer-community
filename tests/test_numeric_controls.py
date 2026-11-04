"""Numeric input policy tests use offscreen Qt widgets, without hardware."""

import pytest
from PySide6.QtCore import QLocale, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nlab.views.numeric_controls import (
    bind_numeric_control,
    install_numeric_input_policy,
    normalize_numeric_control,
)


@pytest.mark.parametrize("widget_class", [QSpinBox, QDoubleSpinBox])
@pytest.mark.parametrize("finish", ["enter", "tab"])
@pytest.mark.parametrize("text,expected", [("0", 100), ("9999", 1250), ("409", 409)])
def test_pending_numeric_edit_clamps_before_one_commit(qtbot, widget_class, finish, text, expected):
    install_numeric_input_policy()
    # This widget is created after installation, as with modeless dialogs.
    parent = QWidget()
    qtbot.addWidget(parent)
    layout = QVBoxLayout(parent)
    spinbox = widget_class()
    spinbox.setRange(100, 1250)
    spinbox.setValue(500)
    spinbox.setSuffix(" ns")
    spinbox.setPrefix("t = ")
    layout.addWidget(spinbox)
    layout.addWidget(QPushButton("Next"))
    parent.show()
    spinbox.setFocus()
    qtbot.waitUntil(spinbox.hasFocus)
    commits = []
    spinbox.editingFinished.connect(lambda: commits.append(spinbox.value()))
    # Exercise pending text, including values Qt has not accepted into value().
    spinbox.lineEdit().setText(f"t = {text} ns")
    QTest.keyClick(spinbox, Qt.Key.Key_Return if finish == "enter" else Qt.Key.Key_Tab)
    assert spinbox.value() == expected
    assert commits == [expected]


@pytest.mark.parametrize(
    "minimum,maximum,step,requested,expected",
    [
        (3, 12, 4, 99, 11),
        (3, 12, 4, 5, 7),
        (0, 524.28, 0.008, 524.28, 524.28),
        (0, 524.28, 0.008, 0.004, 0.008),
        (0, 524.28, 0.008, 0.012, 0.016),
        (-1, 1, 0.01, -0.015, -0.01),
        (-1, 1, 0.01, -0.016, -0.02),
    ],
)
def test_discrete_grid_handles_decimal_steps_negative_ties_and_endpoints(
    qtbot, minimum, maximum, step, requested, expected
):
    spinbox = QDoubleSpinBox()
    qtbot.addWidget(spinbox)
    spinbox.setDecimals(6)
    spinbox.setRange(minimum, maximum)
    spinbox.setSingleStep(step)
    bind_numeric_control(spinbox, snap_to_step=True)
    normalize_numeric_control(spinbox, requested)
    assert spinbox.value() == expected


def test_locale_continuous_precision_and_idempotent_binding(qtbot):
    spinbox = QDoubleSpinBox()
    qtbot.addWidget(spinbox)
    spinbox.setLocale(QLocale(QLocale.Language.German))
    spinbox.setDecimals(6)
    spinbox.setRange(-1, 1)
    spinbox.setSingleStep(0.1)
    bind_numeric_control(spinbox)
    bind_numeric_control(spinbox)
    spinbox.lineEdit().setText("-0,012345")
    QTest.keyClick(spinbox, Qt.Key.Key_Return)
    assert spinbox.value() == -0.012345
    spinbox.lineEdit().setText("9,5")
    QTest.keyClick(spinbox, Qt.Key.Key_Return)
    assert spinbox.value() == 1


def test_non_numeric_edit_retains_last_value(qtbot):
    spinbox = QSpinBox()
    qtbot.addWidget(spinbox)
    spinbox.setRange(100, 1250)
    spinbox.setValue(500)
    bind_numeric_control(spinbox)
    spinbox.lineEdit().setText("invalid")
    QTest.keyClick(spinbox, Qt.Key.Key_Return)
    assert spinbox.value() == 500
