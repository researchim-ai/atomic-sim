"""Synthetic admission and domain fixtures; never substituted for primary data."""

import json
from copy import deepcopy

import pytest

from scripts import screen_chf_measurements as s
from atomic_model_multiphysics.provenance import sha


def row(**updates):
    return dict(TestID=1, Source="synthetic", Provenance="unit fixture, not a measurement",
                Diameter=.008, Length=1., Pressure=1e6, MassFlux=500., XOUT=.05,
                HeatFlux=1e6, Power=25132.7412287, Perimeter=.0251327412287,
                LocalCriticalHeatFlux=1e6, **updates)


def test_numeric_candidate_does_not_promote_experimental_validation():
    result = s.screen([row()])
    assert result["models"]["biasi_inl"]["numeric_domain_candidates"] == 1
    assert result["models"]["table1995"]["numeric_domain_candidates"] == 0
    assert not result["experimental_validation_admitted"] and not result["validated"]
    assert not result["predictions_fitted"]
    assert result["rows"][0]["Provenance"] == "unit fixture, not a measurement"


@pytest.mark.parametrize("field,value,reason", [("Pressure", 15e6, "pressure"), ("MassFlux", 601., "mass_flux"),
                                               ("XOUT", -.01, "quality"), ("Diameter", .004, "diameter"),
                                               ("Length", 7., "length")])
def test_each_biasi_domain_dimension_is_preserved(field, value, reason):
    changed = row(); changed[field] = value
    assert reason in s.reasons(changed, "biasi_inl")


def test_disconnected_table_boxes_and_exact_diameter_are_not_relaxed():
    actual = row();actual.update(Pressure=15.5e6,MassFlux=1500.)
    assert not s.reasons(actual,"table1995")
    actual["Diameter"] = .00800001
    assert "exact_8mm_diameter" in s.reasons(actual,"table1995")
    actual.update(Diameter=.008,Pressure=8e6)
    assert "disconnected_pressure_massflux_quality_box" in s.reasons(actual,"table1995")


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_invalid_coordinates_cannot_silently_become_exclusions(value):
    changed = row();changed["Pressure"] = value
    with pytest.raises(ValueError):s.reasons(changed,"biasi_inl")


def test_nonexclusive_exclusions_preserve_all_rows_and_missing_local_chf():
    a=row();a.update(Pressure=21e6,MassFlux=3000.,LocalCriticalHeatFlux=None)
    result=s.screen([a])
    assert result["total_rows"]==1 and result["unresolved_local_chf_count"]==1
    assert sum(result["models"]["biasi_inl"]["nonexclusive_exclusion_counts"].values())==2
    assert result["source_family_counts"]=={"synthetic":1}


def fixture(tmp_path, monkeypatch):
    # Pin synthetic bytes solely to exercise parsing; production pins stay fixed.
    fields=row();fields.pop("Provenance")
    fields["Fluid"]="water";fields["HeatFluxDefinition"]="power_per_heated_area";fields["Provenance"]="synthetic"
    text='<struct>'+''.join('<dataset>'+''.join(f'<{k}>{i if k=="TestID" else v}</{k}>' for k,v in fields.items())+'</dataset>' for i in range(1,652))+'</struct>'
    xml=tmp_path/'fixture.xml';xml.write_text(text)
    readme=tmp_path/'readme';readme.write_text('synthetic units fixture')
    meta=tmp_path/'record.json';meta.write_text(json.dumps(dict(id=s.RECORD,metadata=dict(doi=s.DOI,license=dict(id='cc-by-4.0'),access_right='open'))))
    monkeypatch.setattr(s,'PINS',{'uniform_standardized.xml':sha(xml),'README_uniform.md':sha(readme)})
    return xml,readme,meta


def test_parser_preserves_si_values_provenance_and_algebra(tmp_path,monkeypatch):
    rows=s.read_pinned(*fixture(tmp_path,monkeypatch))
    assert len(rows)==651 and rows[-1]['TestID']==651
    assert rows[0]['Pressure']==1e6 and rows[0]['XOUT']==.05
    assert rows[0]['heat_flux_algebra_absolute_w_m2']<1e-6


@pytest.mark.parametrize('change',['xml','readme','license','doi','access'])
def test_changed_primary_inputs_or_license_are_refused(tmp_path,monkeypatch,change):
    xml,readme,meta=fixture(tmp_path,monkeypatch)
    if change=='xml':xml.write_text(xml.read_text()+' ')
    elif change=='readme':readme.write_text('changed')
    else:
        data=json.loads(meta.read_text())
        if change=='license':data['metadata']['license']['id']='not-permitted'
        elif change=='doi':data['metadata']['doi']='different'
        else:data['metadata']['access_right']='restricted'
        meta.write_text(json.dumps(data))
    with pytest.raises(ValueError):s.read_pinned(xml,readme,meta)
