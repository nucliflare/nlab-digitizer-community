# General Settings candidates

The General Settings panel owns behavior that is truly application-wide.
Automatic snapshots are isolated by device IP so connecting a second digitizer
cannot apply the first digitizer's settings.

The following controls have been centralized and apply to every available
channel:

| Setting | General Settings section |
| --- | --- |
| Scope display mode, persistence, and refresh | Scope display |
| MCA histogram/debug refresh | Acquisition and monitoring |
| PSU monitoring refresh and plot history | Acquisition and monitoring |
| Measurement destination and MCA output format | Acquisition files |
| MCA ROI visibility and logarithmic Y axis | Histograms |
| Automatic per-device configuration snapshot | Configuration persistence |

Scope and MCA measurement duration remain in their acquisition panels because
they can intentionally differ by channel. Global diagnostics and temperature-
correction intervals also remain in Global beside their live enable/arm state.

Trigger levels, timing, filter coefficients, bias voltages, compensation modes,
DMA enable controls, and acquisition start/stop controls should stay in their
Scope, MCA, PSU, or Global tabs. They are channel-specific hardware operations,
and moving them would hide the channel and live-state context needed to use them
safely.
