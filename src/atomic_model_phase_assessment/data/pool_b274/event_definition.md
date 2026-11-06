# Event definitions for the standard 30-case archive

This document records the operational event definitions used for the processed fields in the subatmospheric pool-boiling dataset. The definitions describe the archived analysis outputs; they do not imply that the events are thermodynamic singularities.

## Signals and reference state

The temperature record is the reference time base for processed thermal quantities. Surface temperature is reconstructed from a one-dimensional linear fit through four embedded copper-block thermocouples. Heat flux is calculated from the fitted through-thickness temperature gradient using the Fourier-law convention documented in the companion BoilingLab workflow. Saturation temperature is calculated from the measured vapor-space pressure using the IAPWS-95 water formulation through the analysis software.

## Events

- **ONB:** the stored onset-of-nucleate-boiling marker on the heating trajectory, identified from the transition from the pre-boiling/natural-convection response to sustained nucleate-boiling behavior.
- **CHF:** the stored critical-heat-flux marker on the heating trajectory. For the standard tests, CHF was identified from a CHF-related thermal excursion and the heater was turned off shortly afterward. The stored CHF value is the selected heating-branch event value, not a claim that a mathematically exact singularity was resolved.
- **T_Surf_Max:** the maximum reconstructed surface temperature during the post-CHF thermal excursion and subsequent cooling response. It is an apparent wall-temperature measure affected by the embedded-thermocouple reconstruction and heater-block thermal inertia.
- **NBR:** the selected cooling-branch point at which the heat-flux--wall-superheat trajectory returns to and remains on the pre-CHF nucleate-boiling branch. The immediate post-CHF collapse is excluded from the NBR selection. NBR therefore denotes an operational return criterion, not simply CHF evaluated in reverse.
- **q_min:** the minimum stored heat flux between the selected CHF and NBR events, where available.

The boiling-hysteresis ratio is `H = q''_NBR / q''_CHF`. The standard archive contains one transient at each surface-pressure combination and should not be treated as a replicated estimate of repeatability.

## Reproduction limits

The organized archive retains selected event times and values but not every case-specific search-window boundary, exclusion interval, manual adjudication note, or raw decision trace used in the original analysis. Consequently, the event fields are reproducible as archived outputs but a complete independent re-selection sensitivity study cannot be reconstructed from this archive alone. The companion analysis code and generated summary outputs are linked in `processing/README.md` and the BoilingLab repository.
