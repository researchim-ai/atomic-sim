"""Flow-dependent single-phase film and explicit finite phase reserves.

Equivalent round tubes sized once to the old nominal UA and actual core volume;
not a rod-bundle model, boiling law, CHF prediction or plant calibration.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from atomic_model_multiphysics.provenance import seal, unseal, sha
from atomic_model_plant_extension import water
from physics.hydraulics import closures
from physics.materials.iapws import saturation_temperature_k
from .primary import PWRPrimarySession, PrimaryPWR, PrimaryConfig
from .core import CoreConfig, core_signature
from .power import BackupConfig
from .feed import FeedConfig
from .cooling import CoolingConfig
from .pwr_train import TrainConfig


def film_signature():
    return dict(core=core_signature(), correlation_sha256=sha(closures.__file__))


@dataclass(frozen=True)
class FilmConfig:
    heated_length_m: float = 4.
    minimum_subcooling_k: float = .1
    minimum_phase_fraction: float = 1e-8

    def __post_init__(self):
        for key, low, high in [('heated_length_m', 1., 8.),
                              ('minimum_subcooling_k', 0., 10.),
                              ('minimum_phase_fraction', 1e-9, 1e-3)]:
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Film parameter domain: ' + key)


def film_transfer(state, mass_flow_kg_s, geometry, wall_temperature_k):
    """Heating-only Dittus-Boelter; no laminar/transition extrapolation."""
    values = [mass_flow_kg_s, wall_temperature_k, state.viscosity_pa_s,
              state.conductivity_w_m_k, state.cp_j_kg_k,
              *[geometry[k] for k in ('diameter_m', 'length_m', 'flow_area_m2', 'heated_area_m2')]]
    if any(not math.isfinite(v) for v in values) or min(values[2:]) <= 0 or mass_flow_kg_s <= 0:
        raise ValueError('Positive finite forward film state/geometry required')
    if state.region != 1:
        raise ValueError('Single-phase liquid film required; no boiling closure')
    if wall_temperature_k < state.T_k:
        raise ValueError('Heating-only film; reverse heat transfer unsupported')
    d = geometry['diameter_m']
    if not .005 <= d <= .03 or geometry['length_m'] / d <= 10:
        raise ValueError('Equivalent fully developed round-tube geometry')
    re = mass_flow_kg_s * d / (geometry['flow_area_m2'] * state.viscosity_pa_s)
    pr = closures.prandtl(state.viscosity_pa_s, state.cp_j_kg_k, state.conductivity_w_m_k)
    nu = closures.nusselt_dittus_boelter(re, pr).value
    h = nu * state.conductivity_w_m_k / d
    ua = h * geometry['heated_area_m2']
    return dict(reynolds=float(re), prandtl=float(pr), nusselt=float(nu), coefficient_w_m2_k=float(h),
                conductance_w_k=float(ua), heat_w=float(ua * (wall_temperature_k - state.T_k)),
                mass_flux_kg_m2_s=mass_flow_kg_s / geometry['flow_area_m2'],
                model_id=closures.PASSPORT_DITTUS.model_id,
                source=list(closures.PASSPORT_DITTUS.data_sources),
                uncertainty='Uncalibrated equivalent tube; no assigned confidence interval',
                chf_w_m2=None, dnbr=None, boiling_supported=False, validated=False)


def phase_inventory(volume):
    """Actual M/U-owned phases; no quality clipping or state setter."""
    point = volume.thermo
    if any(not math.isfinite(v) or v <= 0 for v in (volume.mass_kg, volume.volume_m3, volume.internal_energy_j)):
        raise ValueError('Positive finite phase inventory required')
    ts = float(saturation_temperature_k(point.p_pa * 1e-6))
    result = dict(region=point.region, temperature_k=point.T_k, saturation_temperature_k=ts,
                  subcooling_k=ts-point.T_k, pressure_pa=point.p_pa,
                  total_mass_kg=volume.mass_kg)
    if point.region == 1:
        x = 0.
    elif point.region == 2:
        x = 1.
    elif point.region == 4:
        x = float(point.quality)
        if not 0 <= x <= 1:
            raise ValueError('Phase quality outside [0,1]')
        liquid, vapor = water.saturation(float(point.p_pa))
        specific_volume = (1-x)/liquid.density_kg_m3 + x/vapor.density_kg_m3
        energy = (1-x)*liquid.internal_energy_j_kg + x*vapor.internal_energy_j_kg
        density_error = abs(1/specific_volume - volume.mass_kg/volume.volume_m3)/(volume.mass_kg/volume.volume_m3)
        energy_error = abs(energy-volume.internal_energy_j/volume.mass_kg)
        result.update(mixture_density_relative_residual=density_error,
                      mixture_energy_residual_j_kg=energy_error)
        if abs(point.T_k-ts)>1e-6 or density_error>1e-8 or energy_error>.01:
            raise ValueError('Equilibrium phase inventory residual')
    else:
        raise ValueError('Unsupported station phase region')
    result.update(quality=x, liquid_mass_kg=volume.mass_kg*(1-x), vapor_mass_kg=volume.mass_kg*x)
    return result


class FilmPWR(PrimaryPWR):
    def _apply_heat_and_actuators(self):
        super()._apply_heat_and_actuators()
        if not hasattr(self, 'film_geometry'):
            return
        self.film_terms = film_transfer(self.network.volumes['core'].thermo,
            self.network.junctions['core_out'].mass_flow_kg_s, self.film_geometry, self.clad_temperature_k)
        self.clad_to_coolant_w = self.film_terms['heat_w']
        self.network.volumes['core'].heat_w = self.clad_to_coolant_w


class PWRFilmSession(PWRPrimarySession):
    def _make_plant(self):
        return FilmPWR(height_m=self.primary_config.height_m)

    def __init__(self, config=None, *, cooling=None, feed=None, backup=None, core=None,
                 primary=None, film=None, max_step_s=.1, rtol=1e-7):
        self.film_ready = False
        self.film_config = film if film is not None else FilmConfig()
        if type(self.film_config) is not FilmConfig:
            raise ValueError('FilmConfig required')
        super().__init__(config, cooling=cooling, feed=feed, backup=backup, core=core,
                         primary=primary, max_step_s=max_step_s, rtol=rtol)
        p = self.plant
        point = p.network.volumes['core'].thermo
        volume = p.network.volumes['core'].volume_m3
        length = self.film_config.heated_length_m
        area = volume / length
        flow = p.network.junctions['core_out'].mass_flow_kg_s
        pr = point.cp_j_kg_k * point.viscosity_pa_s / point.conductivity_w_m_k
        # UA = C * D**(-1.2), with V=A_flow*L and A_heat=4V/D.
        coefficient = .023*(flow/(area*point.viscosity_pa_s))**.8*pr**.4*point.conductivity_w_m_k*4*volume
        diameter = float((coefficient/p.clad_coolant_ua)**(1/1.2))
        self.film_geometry = dict(diameter_m=diameter, length_m=length, flow_area_m2=area,
            heated_area_m2=4*volume/diameter, core_volume_m3=volume,
            equivalent_tube_count=4*area/(math.pi*diameter**2),
            sizing='One nominal UA match at unchanged M/U, clad temperature and actual core volume; not measured geometry')
        p.film_geometry = deepcopy(self.film_geometry)
        self.film_ready = True
        self._rhs(self.time, self.state)
        self._admit(self.state)

    def _admit(self, y):
        super()._admit(y)
        if not self.film_ready:
            return
        c = self.film_config
        phases = {name: phase_inventory(node) for name, node in self.plant.network.volumes.items()}
        for name in ('core', 'hot_leg', 'sg_primary', 'cold_leg'):
            r = phases[name]
            if r['region'] != 1 or r['subcooling_k'] < c.minimum_subcooling_k:
                raise ValueError('Primary subcooling/phase limit: ' + name)
        wall = float(y[self.base_size+1])
        if wall < phases['core']['temperature_k'] or phases['core']['saturation_temperature_k']-wall < c.minimum_subcooling_k:
            raise ValueError('Heating-only subcooled wall film limit')
        for name in ('pressurizer', 'sg_secondary'):
            r = phases[name]
            if r['liquid_mass_kg'] < c.minimum_phase_fraction*r['total_mass_kg']:
                raise ValueError('Liquid phase reserve exhausted: ' + name)
        sg = phases['sg_secondary']
        if sg['region'] != 4 or sg['vapor_mass_kg'] < c.minimum_phase_fraction*sg['total_mass_kg']:
            raise ValueError('SG equilibrium vapor reserve exhausted')
        self.phase_terms = phases

    def parameter_registry(self):
        result = super().parameter_registry()
        if self.film_ready:
            for key, value in asdict(self.film_config).items():
                unit = 'm' if key.endswith('_m') else 'K' if key.endswith('_k') else '1'
                result['parameters']['film.'+key] = dict(value=float(value), unit=unit,
                    source='Declared equivalent tube and phase guard; not a rod-bundle calibration',
                    family='illustrative', version='1', uncertainty=None)
            for key in ('diameter_m', 'flow_area_m2', 'heated_area_m2'):
                unit = 'm' if key=='diameter_m' else 'm^2'
                result['parameters']['film_design.'+key] = dict(value=self.film_geometry[key], unit=unit,
                    source=self.film_geometry['sizing'], family='illustrative', version='1', uncertainty=None)
        return result

    def snapshot(self):
        result = super().snapshot()
        if self.film_ready:
            # Re-audit the committed state; snapshot must not expose a solver probe.
            self._admit(self.state)
            result.update(model_id='npp-pwr-single-phase-film', film=deepcopy(self.plant.film_terms),
                film_config=asdict(self.film_config), film_geometry=deepcopy(self.film_geometry),
                phase_inventory=deepcopy(self.phase_terms),
                phase_scope='Subcooled primary, equilibrium SG/pressurizer reserves; no boiling/CHF prediction')
        return result

    def checkpoint(self):
        payload = unseal(super().checkpoint())
        payload.update(film_config=asdict(self.film_config), film_source=film_signature())
        return seal(payload)

    @classmethod
    def restore(cls, saved):
        p = unseal(deepcopy(saved))
        if p['film_source'] != film_signature() or type(p['history']) is not list or len(p['history']) > 256:
            raise ValueError('Film replay source/budget')
        session = cls(TrainConfig(**p['config']), cooling=CoolingConfig(**p['cooling_config']),
            feed=FeedConfig(**p['feed_config']), backup=BackupConfig(**p['backup_config']),
            core=CoreConfig(**p['core_config']), primary=PrimaryConfig(**p['primary_config']),
            film=FilmConfig(**p['film_config']), max_step_s=p['max_step_s'], rtol=p['rtol'])
        for row in p['history']:
            session.advance(**row)
        if session.checkpoint() != saved:
            raise ValueError('Film replay differs')
        return session
