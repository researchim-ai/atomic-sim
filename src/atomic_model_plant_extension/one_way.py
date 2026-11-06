"""One-way inertial feed path with zero-flow events and a closed first law.

New, separate physical configuration. The frozen legacy feed-lag model is not
changed. Feed pressure work, kinetic storage and dissipation are accounted once.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from core.continuation import DenseInterval
from core.solvers import IntegrationResult
from core.status import DomainError, ModelStatus, SolverError
from plants.vver import VVERPlant

from .volume import StrictVolume


class ReducedOutput:
    def __init__(self, dense, retained, known, size):
        self.dense, self.retained, self.known, self.size = dense, retained, known, size
        self.ts = dense.ts

    def __call__(self, time):
        q = np.asarray(time)
        out = np.empty((self.size, *q.shape))
        out[self.retained] = self.dense(time)
        for index, fn in self.known.items():
            out[index] = fn(q)
        return out


class OneWayFeedPlant(VVERPlant):
    def __init__(self):
        super().__init__()
        feed = self.network.junctions["feed"]
        feed.flow_law = "momentum"
        feed.check_valve = True
        vessel = self.network.volumes["pressurizer"]
        vessel.boron_kg = vessel.mass_kg * (1 - vessel.thermo.quality) * self.boron_ppm_ref * 1e-6
        self.feed_density_ref = float(self.network.boundaries["condenser"].density_kg_m3)
        self.feed_inertance = feed.length_m / feed.area_m2
        self._feed_open = True
        self._capture = None
        self._hybrid = None
        self._hybrid_signature = None
        self.maximum_internal_step_s = 2.0
        self.passport = replace(
            self.passport,
            model_id="vver-lumped-inertial-feed",
            version="1.0.0",
            status=ModelStatus.IMPLEMENTED,
            approximations=(
                *self.passport.approximations,
                "feed pipe: constant reference density, true inertance and zero-flow check-valve events",
                "pressurizer bottom port selects saturated liquid and nonvolatile dissolved boron",
                "network stock inversion: separate IF97 regions1/2/4 chart0.1..16.5MPa",
                "target steam/spray/dump lags are analytically eliminated when source is zero",
            ),
        )
        self.refresh_algebraic(self.pack())

    def _sync_hotwell(self):
        # A Newton guess must not turn RHS into a function of earlier rejected
        # trials. Keep the unchanged low-pressure IF97 kernel, but seed it from
        # the declared condenser reference on every call.
        from physics.materials.iapws import saturation_properties

        self.condenser._pressure_guess_pa = self.p_condenser
        self.condenser._temperature_guess_k = saturation_properties(self.p_condenser)[0].T_k
        return super()._sync_hotwell()

    @property
    def feed_index(self):
        return 13 + 3 * len(self.network.volumes) + self.network.junction_names().index("feed")

    def _build_network(self):
        network = super()._build_network()
        network.volumes = {
            name: StrictVolume(**vars(volume)) for name, volume in network.volumes.items()
        }
        from .ports import PhasePortNetwork

        return PhasePortNetwork(network.volumes, network.junctions, network.boundaries)

    def feed_terms(self):
        feed = self.network.junctions["feed"]
        source = self.network.boundaries["condenser"]
        back = self.network.volumes["sg_secondary"].pressure_pa - source.pressure_pa
        friction = feed.friction_dp(self.feed_density_ref)
        return float(feed.pump_pressure_pa - back - friction), float(friction), float(back)

    def _apply_heat_and_actuators(self):
        super()._apply_heat_and_actuators()
        if not hasattr(self, "feed_density_ref"):
            return
        feed = self.network.junctions["feed"]
        source = self.network.boundaries["condenser"]
        _, friction, back = self.feed_terms()
        if abs(source.density_kg_m3 / self.feed_density_ref - 1) > 0.02:
            raise DomainError("Feed incompressible-reference density contrast exceeds2%")
        # Parent network adds q*head/rho_source once. Remove that addition from
        # delivered enthalpy; replace it with the pressure rise + dissipated heat.
        feed.delivery_enthalpy_j_kg = (
            source.enthalpy_j_kg
            + (back + friction) / self.feed_density_ref
            - feed.pump_pressure_pa / source.density_kg_m3
        )

    def derivatives(self, time, state):
        rate = super().derivatives(time, state)
        drive, friction, back = self.feed_terms()
        feed = self.network.junctions["feed"]
        q = feed.mass_flow_kg_s
        if self._feed_open and q < 0:
            # Smooth continuation of the OPEN-side vector field for implicit
            # trial stages and zero-root bracketing. Accepted negative flow is
            # still refused. Switching the donor to SG steam at a trial q<0
            # creates an artificial enthalpy kink at the valve root.
            source = self.network.boundaries["condenser"]
            receiver = self.network.volumes["sg_secondary"]
            desired = source.enthalpy_j_kg + (back + friction) / self.feed_density_ref
            i = 13 + 3 * self.network.volume_names().index("sg_secondary")
            rate[i + 1] += q * (desired - receiver.thermo.h_j_kg)
            rate[i + 2] += q * (source.boron_fraction - receiver.boron_fraction)
        rate[self.feed_index] = drive / self.feed_inertance if self._feed_open else 0.0
        return rate

    def stored_accounts_j(self):
        result = super().stored_accounts_j()
        if hasattr(self, "feed_density_ref"):
            q = self.network.junctions["feed"].mass_flow_kg_s
            result["feed_pipe_j"] = 0.5 * self.feed_inertance * q * q / self.feed_density_ref
        return result

    def account_rates_w(self, dot):
        result = super().account_rates_w(dot)
        q = self.network.junctions["feed"].mass_flow_kg_s
        result["feed_pipe_w"] = (
            self.feed_inertance * q * float(dot[self.feed_index]) / self.feed_density_ref
        )
        return result

    def boundary_terms_w(self):
        result = super().boundary_terms_w()
        feed = self.network.junctions["feed"]
        result["feed_work_w"] = feed.mass_flow_kg_s * feed.pump_pressure_pa / self.feed_density_ref
        return result

    def _reject_if_out_of_domain(self):
        super()._reject_if_out_of_domain()
        if self.network.junctions["feed"].mass_flow_kg_s < 0:
            raise DomainError("Accepted feed check valve violates nonnegative flow")

    def advance(self, *args, **kwargs):
        mode = self._feed_open
        saved = self.snapshot()
        diagnostics = {
            name: deepcopy(getattr(self, name))
            for name in ("last_energy_ledger", "last_motion_events", "last_protection_event_s")
        }
        ledger = None if self.horizon_ledger is None else self.horizon_ledger.checkpoint()
        try:
            return super().advance(*args, **kwargs)
        except Exception:
            self.restore(saved)
            self.refresh_algebraic(self.pack())
            for name, value in diagnostics.items():
                setattr(self, name, value)
            self._feed_open = mode
            self._hybrid = None
            self._hybrid_signature = None
            if ledger is not None:
                self.horizon_ledger.rollback(ledger)
            raise

    def _integrate_interval(self, *args, **kwargs):
        previous = self._capture
        self._capture = []
        try:
            result = super()._integrate_interval(*args, **kwargs)
            events = list(result.events)
            for event in self._capture:
                if event not in events:
                    events.append(event)
            result.events = tuple(
                sorted(events, key=lambda e: (e["t"], 0 if e["kind"] == "rod_stop" else 1))
            )
            return result
        finally:
            self._capture = previous

    def _solve_interval(self, y0, t0, t1, rtol, atol, first, jacstep, method):
        index = self.feed_index
        if y0[index] < 0:
            raise DomainError("Negative initial feed inventory momentum")
        state = y0.copy()
        now = t0
        pieces = []
        source_just_opened = set()
        self.refresh_algebraic(state)
        self._feed_open = bool(state[index] > 0 or self.feed_terms()[0] > 0)
        for segment in range(32):
            self.refresh_algebraic(state)
            targets = {
                13 + 3 * len(self.network.volumes) + self.network.junction_names().index(n): n
                for n, j in self.network.junctions.items()
                if j.flow_law == "target"
                and j.flow_target_kg_s == 0
                and n not in source_just_opened
            }
            known = {
                i: (
                    lambda time, i=i, q=float(state[i]), origin=now, tau=self.network.junctions[n].flow_time_constant_s: (
                        q * np.exp(-(time - origin) / tau)
                    )
                )
                for i, n in targets.items()
            }
            if not self._feed_open:
                known[index] = lambda time: np.zeros_like(time, dtype=float)
            retained = np.array([i for i in range(len(state)) if i not in known])

            def full(time, small):
                result = state.copy()
                result[retained] = small
                for i, fn in known.items():
                    result[i] = fn(np.asarray(time))
                return result

            def fun(time, small):
                return self.derivatives(time, full(time, small))[retained]

            open_mode = self._feed_open

            def valve_event(time, small):
                self.refresh_algebraic(full(time, small))
                if open_mode:
                    return self.network.junctions["feed"].mass_flow_kg_s
                return self.feed_terms()[0]

            valve_event.terminal = True
            valve_event.direction = -1 if open_mode else 1
            events = [valve_event]
            labels = ["feed_closed" if open_mode else "feed_opened"]
            for i, n in targets.items():
                if n == "steam":
                    continue

                def active_source(time, small, n=n):
                    self.refresh_algebraic(full(time, small))
                    if n == "dump":
                        return (
                            self.network.volumes["sg_secondary"].pressure_pa - self.p_dump_open_pa
                        )
                    return (
                        self.network.volumes["pressurizer"].pressure_pa
                        - self.p_pressurizer_ref_pa
                        - self.pressurizer_deadband_pa
                    )

                active_source.terminal = True
                active_source.direction = 1
                if active_source(now, state[retained]) <= 0:
                    events.append(active_source)
                    labels.append(n + "_source_opened")
            from .continuation import HybridContinuation

            signature = (
                tuple(sorted(known)),
                self._feed_open,
                self.throttle,
                self.motor_on,
                self.scram_latched,
                self.rod_command,
                self._rod_rate_override,
                self.feed_bias_kg_s,
                self.t_sink,
                rtol,
                tuple(np.atleast_1d(atol)),
                jacstep,
                method,
                self.maximum_internal_step_s,
            )
            a = atol if np.ndim(atol) == 0 else np.asarray(atol)[retained]
            context = self._hybrid
            if (
                context is None
                or context.invalid
                or signature != self._hybrid_signature
                or context.solver.t != now
                or not np.array_equal(context.solver.y, state[retained])
            ):
                context = HybridContinuation(
                    fun,
                    state[retained],
                    now,
                    t1,
                    rtol=rtol,
                    atol=a,
                    first=first,
                    max_step=self.maximum_internal_step_s,
                    jacstep=jacstep,
                    method=method,
                )
                self._hybrid = context
                self._hybrid_signature = signature
            piece, hit = context.advance_until(t1, events)
            piece.y = np.column_stack([full(t, v) for t, v in zip(piece.t, piece.y.T)])
            if piece.dense is not None:
                piece.dense = ReducedOutput(piece.dense, retained, known, len(state))
            end = float(piece.t[-1])
            endpoint = piece.final.copy()
            self.refresh_algebraic(endpoint)
            if end < t1 and not hit:
                raise SolverError("Stopped segment has no guard certificate")
            if hit:
                gaps = [abs(float(e(end, endpoint[retained]))) for e in events]
                root_events = []
                for which in sorted(hit):
                    label = labels[which]
                    removed_energy = 0.0
                    if label == "feed_closed":
                        if abs(endpoint[index]) > 1e-6:
                            raise SolverError("Feed zero root residual too large")
                        removed_energy = (
                            0.5
                            * self.feed_inertance
                            * float(endpoint[index]) ** 2
                            / self.feed_density_ref
                        )
                        endpoint[index] = 0.0
                        piece.y[index, -1] = 0.0
                        self._feed_open = False
                    elif label == "feed_opened":
                        self._feed_open = True
                    else:
                        source_just_opened.add(label.removesuffix("_source_opened"))
                    event = dict(
                        kind=label,
                        t=end,
                        flow_kg_s=float(endpoint[index]),
                        root_residual=float(gaps[which]),
                        root_roundoff_energy_j=removed_energy,
                    )
                    root_events.append(event)
                    if self._capture is not None:
                        self._capture.append(event)
                piece.events = tuple(root_events)
                if end == now:
                    state = endpoint
                    continue
            if np.any(piece.y[index] < -1e-6):
                raise SolverError("Feed trajectory crossed check-valve constraint")
            pieces.append(piece)
            state = endpoint
            now = end
            if now == t1:
                break
        else:
            raise SolverError("Feed hybrid event budget exhausted")
        self._continuation = None
        self._continuation_signature = None
        if len(pieces) == 1:
            return pieces[0]
        times = np.concatenate([pieces[0].t, *[p.t[1:] for p in pieces[1:]]])
        return IntegrationResult(
            t=times,
            y=np.column_stack([pieces[0].y, *[p.y[:, 1:] for p in pieces[1:]]]),
            n_steps=sum(p.n_steps for p in pieces),
            success=True,
            message="Inertial feed check-valve events",
            dense=DenseInterval(
                [t0, *[float(p.t[-1]) for p in pieces]], [p.dense for p in pieces], mesh_times=times
            ),
            n_rhs_evaluations=sum(p.n_rhs_evaluations for p in pieces),
            n_jacobian_evaluations=sum(p.n_jacobian_evaluations for p in pieces),
            n_lu_decompositions=sum(p.n_lu_decompositions for p in pieces),
            n_accepted_steps=sum(p.n_accepted_steps or 0 for p in pieces),
            events=tuple(e for p in pieces for e in p.events),
        )
