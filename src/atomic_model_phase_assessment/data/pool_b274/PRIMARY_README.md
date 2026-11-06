# Dataset of subatmospheric pool boiling of water on copper surfaces

## Background

The dataset supports the manuscript *Pressure-Adjusted Post-CHF Apparent Base-Wall Superheat Correlates with Boiling Hysteresis in Subatmospheric Pool Boiling*. The associated analysis and figure-generation workflow is maintained in the public [BoilingLab repository](https://github.com/UARK-NED3/BoilingLab/tree/main/manuscripts/boiling_hysteresis_subatmospheric). The manuscript uses the 30 standard cases in this archive; five separate approximately 10 kPa protocol-perturbation cases are archived under `ned3-019-BoilingHysteresis10kPa` and are not part of this dataset.

## Zenodo description

This dataset contains 30 transient saturated pool-boiling experiments with deionized water on flat copper, straight-microchannel copper, and square-micropillar copper surfaces at nominal vapor-space pressures of 10, 20, 30, 40, 50, 60, 70, 80, 90, and 100 kPa. One transient is provided for each surface-pressure combination. In each standard test, a single electrical-heating step was applied until a CHF-related thermal excursion was identified, after which the heater was turned off and the cooling/recovery response was recorded. The archive includes native thermocouple and pressure records, processed thermal histories, reconstructed surface temperature, heat flux, wall superheat, heat-transfer coefficient, selected boiling-event fields, side-view high-speed videos, NumPy data archives, and selected image-derived BubbleID quantities. The data support analysis of CHF, post-CHF recovery, boiling hysteresis, thermal history, pressure effects, and surface-structure effects. The boiling-hysteresis ratio used in the companion manuscript is $H=q''_{NBR}/q''_{CHF}$, where NBR denotes the sustained return to the nucleate-boiling branch. Reconstructed thermal quantities and event labels are derived outputs, while projected vapor fraction and bubble counts are side-view image metrics rather than volumetric void fraction or wall dry-area measurements. The dataset is not a replicated performance database and should be analyzed with condition-aware methods because pressure sequence, surface history, and run order were not fully randomized. The companion analysis and figure-generation workflow is available in [BoilingLab](https://github.com/UARK-NED3/BoilingLab/tree/main/manuscripts/boiling_hysteresis_subatmospheric).


## Dataset purpose

This dataset documents transient saturated pool-boiling experiments with deionized water at nominal vapor-space pressures from 10 to 100 kPa. It contains matched cases for flat Cu, straight-microchannel Cu, and square-micropillar Cu surfaces. The archive retains native temperature and pressure records, processed thermal records, side-view videos, and selected image-derived quantities.

The resource is intended for thermal reconstruction, boiling-event documentation, optical analysis, and condition-grouped analysis. It is not a fully replicated performance database: the standard archive contains one organized transient per surface-pressure combination.

## Experimental matrix

- Working fluid: deionized water near saturation.
- Test surfaces: flat oxygen-free Cu 101; straight-microchannel Cu; square-micropillar Cu.
- Projected boiling area: 10 mm x 10 mm.
- Nominal vapor-space pressure: 10, 20, 30, 40, 50, 60, 70, 80, 90, and 100 kPa.
- Cases: 30 total (3 surfaces x 10 nominal pressures).
- Process: a single applied-voltage step is used until a CHF-related thermal excursion, followed by power removal and cooling/recovery.

Pressure is measured in the vapor space. Local static pressure at the upward-facing test surface is slightly higher because of the liquid head.

## Directory layout

```text
.
├── A-FlatCu/
│   ├── 10kPa/ ... 100kPa/
├── B-CuMicrochannels/
│   ├── 10kPa/ ... 100kPa/
└── C_CuMicroPillars/
    ├── 10kPa/ ... 100kPa/
```

Each condition folder is identified by surface type and nominal vapor-space pressure. File names may use an internal run identifier (for example, `B-285` or `Boiling-285`); do not infer condition from that identifier alone. Use the relative folder path as the authoritative condition label.

## Files in a condition folder

| File or folder | Level | Contents and use |
|---|---|---|
| `Temperature.lvm` | Native | Embedded Cu-block thermocouples and chamber temperature channels. |
| `Pressure.lvm` | Native | Vapor-space pressure record. |
| `DC_power.lvm` | Native, nonuniform | Electrical record where retained; see Missing-modality note. |
| `*_csv_data.csv` | Processed | Tabular thermal and pressure-derived quantities, including reconstructed surface temperature, heat flux, wall superheat, HTC, and event fields. |
| `*_full_data.npz` | Processed/derived | NumPy archive containing sensor arrays, reconstruction outputs, event fields, video metadata, and image-derived arrays. |
| `*_video.mp4` | Optical source | Encoded side-view video. |
| `Plot Images/` | Derived presentation files | Case-specific PNG plots where retained; absence does not mean the primary native/processed records are absent. |

## Data classes and definitions

Native records are instrument-exported files. Processed fields are calculated from native measurements. Derived image fields are produced from the side-view imagery and are not independent measurements.

- `Surface_Temperature_degC`: reconstructed surface temperature from a one-dimensional linear fit through four embedded Cu-block thermocouple temperatures.
- `Heat_Flux_W_cm_2`: Fourier-law heat flux calculated from the fitted through-thickness temperature gradient using the analysis convention documented in the companion BoilingLab analysis workflow.
- `Wall_Superheat_degC`: reconstructed surface temperature minus pressure-derived saturation temperature.
- `HTC_W_cm_2_K_clean` and related fields: processed heat-transfer coefficient fields; inspect the exact NPZ/CSV field name and data-reduction documentation before selecting a version.
- `ONB`, `CHF`, `T_Surf_Max`, and `NBR`: stored event fields for onset of nucleate boiling, critical heat flux, maximum reconstructed surface temperature, and nucleate-boiling return, respectively. Event values are processed labels and require the accompanying event-assignment procedure for independent reproduction.
- `Bubble_count_array`, `Vapor_pixel_area_array`, and `Vapor_fraction_array`: image-derived side-view quantities. Projected vapor fraction is a two-dimensional image metric, not a volumetric void fraction.

## Time bases and video metadata: important release issue

Do not assume that MP4 container time equals the event time base.

An audit of the 30 NPZ/MP4 pairs found:

- Thermocouple time arrays increment by approximately 0.333 ms (about 3.003 kHz).
- Pressure time arrays increment by approximately 0.040 ms (about 25 kHz).
- NPZ video metadata list either 150 or 300 frames/s, depending on case.
- The inspected MP4 containers report 30 frames/s while retaining approximately the same frame count as the NPZ metadata.

Use the NPZ `Bubble_time_array`, `Frame_number_array`, and stored event times for the current processed event-to-frame mapping. Before publication, the maintainers must provide a timing-reconciliation record, common time-origin/trigger definition, and either corrected video metadata or a documented conversion rule. Until then, do not claim sub-frame or independently verified thermal-video synchronization.

## Missing modalities and archive boundaries

- `Temperature.lvm`, `Pressure.lvm`, one processed CSV, one processed NPZ, and one MP4 are present for all 30 inspected cases.
- `DC_power.lvm` is retained for all microchannel cases and for the 90 and 100 kPa micropillar cases. It is not retained for flat-Cu cases or most micropillar cases.
- The standard archive contains one organized transient per surface-pressure condition. It is not a replicate matrix.
- Pressure sequence/run order was not randomized. Surface history, aging/oxidation, pressure, and run order may therefore be partly confounded for structured-surface series.
- Surface temperature, heat flux, wall superheat, and HTC are derived under a one-dimensional conduction reconstruction with constant Cu thermal conductivity.
- Instrument-by-instrument uncertainty propagation, calibration certificates, cold-junction configuration, camera spatial calibration, segmentation provenance, and the original pointwise event-search settings are not included in this archive. These limitations are stated so that derived values are not interpreted as independently calibrated measurements.

## Recommended use

Use condition-aware analysis. Do not randomly split adjacent samples or frames from one transient between training and test data. Prefer whole-condition, pressure-held-out, surface-held-out, run-held-out, or facility-held-out evaluations, depending on the intended task.

When using processed values, retain the corresponding raw/processed file names, data version, analysis code version, and any event-selection settings. Inspect `Goodness_of_Fit_R2` when using the one-dimensional thermal reconstruction during rapid transitions.

## Release metadata

This dataset is archived at [Zenodo DOI 10.5281/zenodo.22234711](https://doi.org/10.5281/zenodo.22234711) under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). The root-level `data_dictionary.csv`, `event_definition.md`, `processing/`, `LICENSE`, and `CITATION.cff` files document the archive contents, field meanings, event definitions, analysis workflow, license, and citation metadata. The `processing/build_release_metadata.py` utility can generate a relative-path manifest with SHA-256 checksums and a per-case timing table when a local copy of the archive is available. Zenodo also provides the deposited file inventory and archive-level integrity information.

The dataset was supported by Google LLC through an unrestricted gift and the U.S. National Science Foundation (NSF) under Grant No. CBET-2323022.

## Contact

Han Hu, Department of Mechanical Engineering, University of Arkansas, Fayetteville, Arkansas, USA. Contact: hanhu@uark.edu.
