"""
Gauss's Method for Preliminary Orbit Determination

Given three optical observations (right ascension & declination) and the
observer's geocentric ECI position at each epoch, estimates the position and
velocity vector of the orbiting body at the middle epoch.

Algorithm
---------
1.  Compute topocentric direction-cosine unit vectors from RA/Dec.
2.  Build the D-matrix from the cross products of the unit vectors and the
    observer position vectors.
3.  Express ρ₂ (middle-epoch slant range) as a function of r₂ = |r₂_vec|
    using second-order Lagrange f,g coefficients.
4.  Solve the resulting scalar constraint  r₂ = |ρ₂(r₂)·L̂₂ + R₂|
    via bisection over a physical range.
5.  Recover ρ₁, ρ₃ and the velocity at the middle epoch.

Reference
---------
Curtis, H.D., "Orbital Mechanics for Engineering Students", 3rd ed., §5.6
Bate, Mueller & White, "Fundamentals of Astrodynamics", §2.10
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MU = 398600.4418        # Earth gravitational parameter  [km³/s²]
RE = 6378.137           # Earth equatorial radius        [km]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """A single topocentric optical observation."""
    t: float          # epoch [s from reference]
    ra: float         # right ascension [rad]
    dec: float        # declination [rad]
    R: np.ndarray     # observer geocentric ECI position [km], shape (3,)


@dataclass
class OrbitState:
    """Cartesian state vector at the middle observation epoch."""
    t2: float
    r: np.ndarray     # position [km]
    v: np.ndarray     # velocity [km/s]
    rho: Tuple[float, float, float]   # slant ranges [km]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unit_vector(ra: float, dec: float) -> np.ndarray:
    """Geocentric direction unit vector from (RA, Dec)."""
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def _lagrange_fg(tau: float, r2: float, mu: float = MU) -> Tuple[float, float]:
    """
    Fourth-order Lagrange f and g coefficients (Taylor series in τ).

    Parameters
    ----------
    tau : time offset from middle epoch [s]  (i.e. tᵢ − t₂)
    r2  : scalar distance of object at middle epoch [km]
    """
    u = mu / r2**3
    u2 = u * u
    tau2 = tau * tau
    f = 1.0 - (u / 2.0) * tau2 + (u2 / 24.0) * tau2**2
    g = tau * (1.0 - (u / 6.0) * tau2 + (u2 / 120.0) * tau2**2)
    return f, g


def _compute_D(L1, L2, L3, R1, R2, R3):
    """
    Build D₀ and D matrix.

    D₀    = L̂₁ · (L̂₂ × L̂₃)
    p[i]  = cross products:  p₀ = L̂₂×L̂₃,  p₁ = L̂₁×L̂₃,  p₂ = L̂₁×L̂₂
    D[i,j] = Rⱼ · pᵢ      (row = cross-product index,  col = station index)
    """
    p = [np.cross(L2, L3), np.cross(L1, L3), np.cross(L1, L2)]
    D0 = float(np.dot(L1, p[0]))
    Rs = [R1, R2, R3]
    D = np.array([[np.dot(Rs[j], p[i]) for j in range(3)] for i in range(3)])
    return D0, D


def _rho2_from_r2(r2: float, D: np.ndarray, D0: float,
                  tau1: float, tau3: float) -> Tuple[float, float, float, float]:
    """
    Given a trial scalar distance r₂, return (rho₂, c₁, c₃, W) using the
    second-order Lagrange f,g expansion.
    """
    f1, g1 = _lagrange_fg(tau1, r2)
    f3, g3 = _lagrange_fg(tau3, r2)
    W = f1 * g3 - f3 * g1
    c1 = g3 / W
    c3 = -g1 / W
    rho2 = (D[1, 1] - c1 * D[1, 0] - c3 * D[1, 2]) / D0
    return rho2, c1, c3, W


def _residual(r2: float, D: np.ndarray, D0: float,
              L2: np.ndarray, R2: np.ndarray,
              tau1: float, tau3: float) -> float:
    """
    Scalar residual  F(r₂) = |ρ₂(r₂)·L̂₂ + R₂| − r₂.
    A root of F is a physically self-consistent r₂.
    """
    rho2, *_ = _rho2_from_r2(r2, D, D0, tau1, tau3)
    r2_implied = float(np.linalg.norm(rho2 * L2 + R2))
    return r2_implied - r2


# --------------------------------------------------------------------------- #
# Bisection root finder
# --------------------------------------------------------------------------- #
def _bisect(f, a: float, b: float, tol: float = 1e-10,
            max_iter: int = 200) -> float:
    """Bisection method to find a root of f in [a, b]."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            f"Bisection: no sign change in [{a:.1f}, {b:.1f}]  "
            f"(f(a)={fa:.3g}, f(b)={fb:.3g})."
        )
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm  = f(mid)
        if abs(b - a) < tol or fm == 0.0:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


# --------------------------------------------------------------------------- #
# Gauss's method
# --------------------------------------------------------------------------- #
def gauss_method(obs1: Observation, obs2: Observation, obs3: Observation,
                 r2_min: float = RE + 100.0, r2_max: float = 500_000.0,
                 n_scan: int = 2000, tol: float = 1e-10) -> OrbitState:
    """
    Apply Gauss's method to three topocentric optical observations.

    The self-consistency equation  r₂ = |ρ₂(r₂)·L̂₂ + R₂|  is solved by
    bisection after a coarse scan over [r2_min, r2_max] to bracket the root.

    Parameters
    ----------
    obs1, obs2, obs3 : Observation
        Three observations in chronological order (t₁ < t₂ < t₃).
    r2_min, r2_max : float
        Search range for the middle-epoch geocentric distance [km].
    n_scan : int
        Number of points in the coarse scan used to bracket roots.
    tol : float
        Bisection convergence tolerance on r₂ [km].

    Returns
    -------
    OrbitState
        Position and velocity at the middle epoch t₂.
    """
    # --- Direction-cosine unit vectors ---
    L1 = _unit_vector(obs1.ra, obs1.dec)
    L2 = _unit_vector(obs2.ra, obs2.dec)
    L3 = _unit_vector(obs3.ra, obs3.dec)
    R1, R2, R3 = obs1.R, obs2.R, obs3.R

    # --- Time intervals relative to t₂ ---
    tau1 = obs1.t - obs2.t   # ≤ 0
    tau3 = obs3.t - obs2.t   # ≥ 0

    # --- D-matrix ---
    D0, D = _compute_D(L1, L2, L3, R1, R2, R3)
    if abs(D0) < 1e-14:
        raise ValueError("D₀ ≈ 0 — observations are nearly co-planar; "
                         "Gauss's method is degenerate.")
    if abs(D0) < 1e-2:
        raise ValueError(
            f"D₀ = {D0:.3e} is too small for reliable results. "
            "The three lines of sight are nearly coplanar (observer is near "
            "the orbital plane), causing catastrophic cancellation. "
            "Try wider time spacing between observations or observations "
            "taken when the satellite is further from the orbital-plane "
            "crossing point."
        )

    # --- Define the residual function ---
    def F(r2):
        return _residual(r2, D, D0, L2, R2, tau1, tau3)

    # --- Coarse scan to find bracket(s) ---
    r2_grid = np.linspace(r2_min, r2_max, n_scan)
    F_grid  = np.array([F(r) for r in r2_grid])

    brackets = []
    for k in range(len(r2_grid) - 1):
        if F_grid[k] * F_grid[k + 1] < 0:
            brackets.append((r2_grid[k], r2_grid[k + 1]))

    if not brackets:
        raise ValueError(
            "No root bracket found in the scan range "
            f"[{r2_min:.0f}, {r2_max:.0f}] km. "
            "Try widening r2_max or checking your observations."
        )

    # --- Refine each bracket and pick the physically best root ---
    candidates = []
    for a, b in brackets:
        try:
            r2_sol = _bisect(F, a, b, tol=tol)
            rho2, c1, c3, W = _rho2_from_r2(r2_sol, D, D0, tau1, tau3)
            if rho2 > 0:
                candidates.append((r2_sol, rho2, c1, c3, W))
        except ValueError:
            pass

    if not candidates:
        raise ValueError("No physically valid solution found (ρ₂ > 0).")

    # Prefer the smallest r₂ (closest orbit that is consistent)
    r2_sol, rho2, c1, c3, W = min(candidates, key=lambda x: x[0])

    # --- Recover the full state ---
    f1, g1 = _lagrange_fg(tau1, r2_sol)
    f3, g3 = _lagrange_fg(tau3, r2_sol)

    rho1 = (D[0, 1] - c1 * D[0, 0] - c3 * D[0, 2]) / (c1 * D0)
    rho3 = (D[2, 1] - c1 * D[2, 0] - c3 * D[2, 2]) / (c3 * D0)

    r1_vec = rho1 * L1 + R1
    r2_vec = rho2 * L2 + R2
    r3_vec = rho3 * L3 + R3

    # Velocity at middle epoch via Lagrange f,g
    v2_vec = (-f3 * r1_vec + f1 * r3_vec) / W

    return OrbitState(
        t2=obs2.t,
        r=r2_vec,
        v=v2_vec,
        rho=(rho1, rho2, rho3),
    )


# --------------------------------------------------------------------------- #
# Orbital elements from Cartesian state
# --------------------------------------------------------------------------- #
def state_to_elements(r_vec: np.ndarray, v_vec: np.ndarray,
                       mu: float = MU) -> dict:
    """
    Convert a Cartesian state vector to classical orbital elements (COEs).

    Returns
    -------
    dict with keys:
        a     semi-major axis [km]
        e     eccentricity [-]
        i     inclination [deg]
        raan  right ascension of ascending node [deg]
        argp  argument of perigee [deg]
        nu    true anomaly [deg]
        T     orbital period [s]
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h     = np.linalg.norm(h_vec)

    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)
    n     = np.linalg.norm(n_vec)

    e_vec = ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e     = np.linalg.norm(e_vec)

    energy = 0.5 * v**2 - mu / r
    a = -mu / (2.0 * energy) if abs(energy) > 1e-12 else np.inf

    i = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0)))

    raan = 0.0
    if n > 1e-10:
        raan = np.degrees(np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0)))
        if n_vec[1] < 0:
            raan = 360.0 - raan

    argp = 0.0
    if n > 1e-10 and e > 1e-10:
        argp = np.degrees(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0)))
        if e_vec[2] < 0:
            argp = 360.0 - argp

    nu = 0.0
    if e > 1e-10:
        nu = np.degrees(np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0)))
        if np.dot(r_vec, v_vec) < 0:
            nu = 360.0 - nu

    T = 2.0 * np.pi * np.sqrt(a**3 / mu) if a > 0 else np.inf

    return {"a": a, "e": e, "i": i, "raan": raan, "argp": argp, "nu": nu, "T": T}


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def print_state(state: OrbitState) -> None:
    r, v = state.r, state.v
    print(f"\n{'='*58}")
    print(f"Orbit state at t₂ = {state.t2:.1f} s")
    print(f"  r = [{r[0]:12.4f}, {r[1]:12.4f}, {r[2]:12.4f}] km   |r| = {np.linalg.norm(r):.4f} km")
    print(f"  v = [{v[0]:12.6f}, {v[1]:12.6f}, {v[2]:12.6f}] km/s |v| = {np.linalg.norm(v):.6f} km/s")
    print(f"  Slant ranges:  ρ₁ = {state.rho[0]:.3f}  ρ₂ = {state.rho[1]:.3f}  ρ₃ = {state.rho[2]:.3f} km")


def print_elements(elems: dict) -> None:
    print(f"\n{'='*58}")
    print("Classical Orbital Elements")
    print(f"  Semi-major axis   a  = {elems['a']:12.4f} km")
    print(f"  Eccentricity      e  = {elems['e']:12.6f}")
    print(f"  Inclination       i  = {elems['i']:12.4f} deg")
    print(f"  RAAN              Ω  = {elems['raan']:12.4f} deg")
    print(f"  Arg of perigee    ω  = {elems['argp']:12.4f} deg")
    print(f"  True anomaly      ν  = {elems['nu']:12.4f} deg")
    print(f"  Orbital period    T  = {elems['T']:12.2f} s  ({elems['T']/3600:.4f} hr)")
