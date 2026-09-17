"""CHO/mAb balances, following Kumar et al. (2022), equations (1)–(14).

Internal state: [volume, viable cells, mAb, glucose, lactate, dead cells].
Time is in minutes; volume in mL; cells in 10^6 cells/mL; mAb in mg/L;
glucose and lactate in g/L. Viability is a derived percentage.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Parameters:
    """Original fitted values, except lacmax1 uses Table 3's reported 628 g/L.

    See docs/model_review.md for the unresolved provenance of the extra fitted digits.
    """

    mumax: float = 1.53065905e-1
    ks: float = 8.50356543e-1
    mud: float = 3.95525447e-5
    ypx: float = 6.61966255e-5
    mp: float = 1.20866880e-2
    yxs: float = 2.05140571
    ms: float = 1.10167094e-4
    kl: float = 3.44210432e2
    yxl: float = 1.98322526
    yls: float = 9.33512253e-2
    kp: float = 6.87871174e-4
    lacmax1: float = 628.0
    lacmax2: float = 4.99990345e-1
    mlac: float = 1.88752322e-5
    k1: float = 1.68890312e3
    k2: float = 5.24079503e2

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative.")
        for name in ("yxs", "kl", "yxl", "lacmax1", "lacmax2"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class Operation:
    """Constant operating conditions.

    The sample outlet carries all species at reactor concentration. The
    perfusion outlet retains cells AND mAb, as in the paper's equations
    (2), (8), and (13), while removing glucose and lactate (equation 10).
    A well-mixed continuous harvest uses sample_flow, with perfusion_flow=0.
    """

    inlet_flow: float = 0.0        # mL/min
    sample_flow: float = 0.0       # mL/min
    perfusion_flow: float = 0.0    # mL/min
    feed_glucose: float = 18.0    # g/L, numerically equal to mg/mL
    temperature: float = 308.0   # K; retained from the original notebook

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative.")
        if self.temperature == 0:
            raise ValueError("temperature must be positive, in kelvin.")


DEFAULT_PARAMETERS = Parameters()
DEFAULT_OPERATION = Operation()
DEFAULT_INITIAL_STATE = (5650.0, 0.7, 0.0, 6.05, 0.12, 0.7 / 99.0)
DEFAULT_DURATION = 5000.0
CSV_COLUMNS = (
    "Time (min)",
    "Volume of Mixture (ml)",
    "Viable Cells (10^6 cells/ml)",
    "mAb Concentration (mg/l)",
    "Glucose Concentration (g/l)",
    "Lactate Concentration (g/l)",
    "Viability (%)",
)


class ModelDomainError(ValueError):
    """The requested trajectory leaves the physical domain of these kinetics."""


def rhs(t, state, parameters=DEFAULT_PARAMETERS, operation=DEFAULT_OPERATION):
    """Concentration balances for solve_ivp; operation is constant in time.

    Nonnegative trial concentrations extend the RHS just beyond an event so
    the solver can locate a boundary without evaluating a singular rate.
    simulate() rejects trajectories crossing that boundary.
    """
    v, x, product, glucose, lactate, dead = state
    if v <= 0:
        raise ModelDomainError("Reactor volume must remain positive.")
    p, op = parameters, operation
    s, l = max(glucose, 0.0), max(lactate, 0.0)
    denominator = p.ks * x + s
    substrate_factor = s / denominator if denominator > 0 else 0.0
    mu = (
        p.mumax * substrate_factor * p.kl / (p.kl + l)
        * (1.0 - p.kp * product) * np.exp(-p.k1 / op.temperature)
    )
    death_rate = p.mud * np.exp(-p.k2 / op.temperature)
    q_glucose = -mu / p.yxs - p.ms
    # Eq. (12): q_glucose is SIGNED, and the first multiplier covers BOTH terms.
    q_lactate = (
        (mu / p.yxl + p.yls * q_glucose) * (1.0 - l / p.lacmax1)
        + p.mlac * (1.0 - l / p.lacmax2)
    )
    q_product = p.ypx * mu + p.mp
    dv = op.inlet_flow - op.sample_flow - op.perfusion_flow
    # Product rule: d(C V)/dt = V dC/dt + C dV/dt.
    retained_dilution = (op.sample_flow + dv) / v
    return np.array([
        dv,
        (mu - death_rate - retained_dilution) * x,
        q_product * x - retained_dilution * product,
        op.inlet_flow / v * (op.feed_glucose - glucose) + q_glucose * x,
        q_lactate * x - op.inlet_flow / v * lactate,
        death_rate * x - retained_dilution * dead,
    ])


def simulate(t_eval, initial_state=DEFAULT_INITIAL_STATE,
             parameters=DEFAULT_PARAMETERS, operation=DEFAULT_OPERATION,
             *, rtol=1e-8, atol=1e-10):
    """Return an (n_times, 6) state array; column 5 is DEAD CELL DENSITY.

    Raises on invalid inputs, solver failure, glucose/lactate depletion, or
    crossing the linear product inhibition limit. No post-depletion model
    is assumed. Use observations() for the original viability-based columns.
    """
    times = np.asarray(t_eval, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("t_eval must contain at least two finite, increasing times.")
    if initial.shape != (6,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must contain six finite values.")
    if initial[0] <= 0 or np.any(initial[1:] < 0):
        raise ValueError("Volume must be positive and concentrations nonnegative.")
    if not np.isfinite(rtol) or not np.isfinite(atol) or min(rtol, atol) <= 0:
        raise ValueError("rtol and atol must be finite and positive.")
    dv = operation.inlet_flow - operation.sample_flow - operation.perfusion_flow
    if initial[0] + dv * (times[-1] - times[0]) <= 0:
        raise ModelDomainError("The requested outlet flow would empty the reactor.")
    if parameters.kp * initial[2] > 1:
        raise ModelDomainError("Initial mAb exceeds the linear inhibition limit 1/kp.")

    def glucose_boundary(t, state):
        return state[3]

    def lactate_boundary(t, state):
        return state[4]

    def inhibition_boundary(t, state):
        return 1.0 - parameters.kp * state[2]

    events = []
    labels = []
    if initial[1] > 0:
        first_rate = rhs(times[0], initial, parameters, operation)
        for index, event, label in (
            (3, glucose_boundary, "glucose depletion"),
            (4, lactate_boundary, "lactate depletion"),
        ):
            if initial[index] == 0 and first_rate[index] <= 0:
                raise ModelDomainError(f"Initial state is at {label}; revise the conditions.")
            event.terminal, event.direction = True, -1
            events.append(event)
            labels.append(label)
        if parameters.kp > 0:
            inhibition_boundary.terminal, inhibition_boundary.direction = True, -1
            events.append(inhibition_boundary)
            labels.append("the linear mAb inhibition limit")

    solution = solve_ivp(
        lambda t, state: rhs(t, state, parameters, operation),
        (times[0], times[-1]), initial, t_eval=times, method="DOP853",
        rtol=rtol, atol=atol, max_step=30.0, events=events or None,
    )
    if not solution.success:
        raise RuntimeError(f"Bioreactor integration failed: {solution.message}")
    if solution.status == 1:
        for label, event_times in zip(labels, solution.t_events):
            if event_times.size:
                raise ModelDomainError(
                    f"Simulation reached {label} at t={event_times[0]:.6f} min. "
                    "Shorten the run or revise operating conditions; "
                    "the model cannot be extrapolated past this boundary."
                )
    states = solution.y.T
    if not np.all(np.isfinite(states)) or np.any(states < -10 * atol):
        raise RuntimeError("Integration returned nonfinite or negative physical states.")
    # Remove only roundoff-sized negative concentrations after validation.
    return np.maximum(states, 0.0)


def observations(states):
    """Replace dead-cell density with viability (%); empty culture reports 0%."""
    values = np.asarray(states, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("states must have shape (n_times, 6).")
    result = values.copy()
    total_cells = values[:, 1] + values[:, 5]
    result[:, 5] = np.divide(
        100.0 * values[:, 1], total_cells,
        out=np.zeros_like(total_cells), where=total_cells > 0,
    )
    return result


def export_csv(path, times, states):
    """Write the original seven-column CSV schema, including time and viability."""
    data = np.column_stack((times, observations(states)))
    np.savetxt(Path(path), data, delimiter=",", header=",".join(CSV_COLUMNS),
               comments="", fmt="%.12g")


if __name__ == "__main__":
    times = np.linspace(0.0, DEFAULT_DURATION, int(DEFAULT_DURATION) + 1)
    states = simulate(times)
    destination = Path(__file__).resolve().parent / "data" / "example_batch.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_csv(destination, times, states)
    print(f"Saved {len(times)} samples to {destination.name}")
    print(f"Final mAb: {states[-1, 2]:.6f} mg/L; "
          f"viability: {observations(states)[-1, 5]:.6f}%")
