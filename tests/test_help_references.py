"""Method help topics keep citing the literature their methods follow."""

from __future__ import annotations

import pytest

from nlab.help.content import load_topics

CITED = {
    "psd": ("10.1016/0029-554X(59)90067-9", "10.1016/0168-9002(92)90617-D"),
    "mca-charge": ("10.1016/0029-554X(59)90067-9", "10.1109/NSSMIC.2014.7431222"),
    "mca-zerocross": ("10.1016/0029-554X(61)90198-7", "10.1016/j.nima.2010.06.252"),
    "fom": ("10.1016/0029-554X(71)90054-1", "10.1016/0167-5087(84)90016-4"),
    "psd-energy-fom": ("10.1088/1748-0221/15/03/P03030", "10.1016/j.nima.2020.164080"),
    "waveform": ("10.1016/j.nima.2008.06.004", "10.1002/9781119410225"),
    "scope-spectrum": (
        "10.1109/TAU.1967.1161901",
        "10.1109/PROC.1978.10837",
        "10.1073/pnas.17.12.684",
    ),
    "current-spectrum": (
        "10.1109/TAU.1967.1161901",
        "10.1109/PROC.1978.10837",
        "10.1073/pnas.17.12.684",
    ),
    "waveform-optimization": (
        "10.1016/j.nima.2023.168184",
        "10.1016/j.nima.2023.168486",
        "10.1016/j.nima.2026.171521",
    ),
    "mca": ("10.1002/9781119410225",),
    "mca-crrc": ("10.1109/TNS.2011.2164556",),
    "mca-trapezoid": ("10.1016/0168-9002(94)91011-1", "10.1016/0168-9002(94)91652-7"),
    "mca-cfd": ("10.1016/0029-554X(68)90473-4", "10.1016/j.nima.2007.04.048"),
    "coincidence-timing": (
        "10.1016/0029-554X(68)90473-4",
        "10.1016/j.nima.2007.04.048",
        "10.1016/S0029-554X(79)90170-8",
    ),
    "energy-calibration": ("10.1002/9781394286577", "10.1109/TNS.2009.2033684"),
    "compton-edge": (
        "10.1007/BF01366453",
        "10.1016/0029-554X(82)90249-X",
        "10.1016/j.radmeas.2009.10.015",
    ),
    "peak-analysis": ("10.1016/0029-554X(76)90472-9", "10.1016/S0168-9002(96)80040-4"),
    "psu-compensation": ("10.1016/j.nima.2016.09.053", "10.1109/TNS.2015.2388580"),
    "global-temperature": ("10.1016/j.nima.2006.06.039", "10.1109/TNS.2005.856616"),
    "conversion": ("10.1016/S0168-9002(97)00048-X", "10.5281/zenodo.4340632"),
    "about": ("10.1038/s41586-020-2649-2", "10.1038/s41592-019-0686-2"),
}


@pytest.mark.parametrize("topic", sorted(CITED))
def test_method_topics_cite_their_references(topic: str) -> None:
    markdown = load_topics()[topic].markdown
    assert "**References** — " in markdown
    for doi in CITED[topic]:
        assert f"doi:{doi}" in markdown
