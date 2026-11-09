"""All raw fields, accepted-chart derivatives and frozen arithmetic reference."""
import json
from pathlib import Path
import numpy as np
import pytest
from physics.materials import iapws

FIXTURE=json.loads(Path(__file__).with_name('fixtures').joinpath('if97_raw_reference_v1.json').read_text())


@pytest.mark.parametrize('row',FIXTURE['rows'],ids=lambda r:f"r{r['region']}-p{r['pressure_mpa']}-t{r['temperature_k']}")
def test_all_raw_properties_match_frozen_corrected_if97(row):
    result=getattr(iapws,f"_region{row['region']}_raw")(row['temperature_k'],row['pressure_mpa'])
    assert set(result)==set(row['expected'])
    for key,expected in row['expected'].items():
        assert np.isfinite(result[key])
        assert result[key]==pytest.approx(expected,rel=2e-10,abs=2e-10),key


def test_derived_coefficients_are_read_only():
    for name in ('_R1_DERIVATIVE_COEFFICIENTS','_R2_DERIVATIVE_COEFFICIENTS','_R2_IDEAL_DERIVATIVES'):
        values=getattr(iapws,name)
        with pytest.raises(ValueError):values[0,0]=0.
