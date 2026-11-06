# HIPER measurements — separate data license

Authors: Jan Oettig, Laura Licht, Moritz Westermeier, Christopher Schifflechner, Hartmut Spliethoff. Technical University of Munich, Chair of Energy Systems.

Source: https://mediatum.ub.tum.de/1839312
DOI: https://doi.org/10.14459/2025mp1839312
License: Creative Commons Attribution-ShareAlike 4.0 International, https://creativecommons.org/licenses/by-sa/4.0/ . This license applies to this derived dataset; the repository Apache license does not replace it.

Changes: all176 published XLSM workbooks were read without running macros; all70 segment rows per workbook retained. Celsius converted to kelvin; workbook pressure in bar converted to Pa; absolute uncertainty endpoints retained. No selection by model error, fitting, clipping of equilibrium quality or synthetic observations. Full workbook filenames, rows and SHA-256 are retained for provenance.

Source discrepancies: record states176 experiments, README states189; actual archive contains176 XLSM files. README table labels pressure MPa; workbook headers specify bar, consistent with165.9bar/16.59MPa. Workbook headers govern conversion. README count discrepancy is not silently repaired. Reported X can be negative or above1 (equilibrium enthalpy quality), not a clipped phase fraction.

Workbook uncertainty endpoints are retained as published. Confidence level and covariance have not been established here; no Gaussian confidence interval is assumed. Local bulk enthalpy/quality/temperature include source data reduction and are not all independent direct sensor observations.
