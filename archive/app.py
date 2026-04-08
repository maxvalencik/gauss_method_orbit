"""
Flask web app — Gauss's Method for Preliminary Orbit Determination
Curtis, Algorithms 5.5 + 5.6 (Orbital Mechanics for Engineering Students, 4th ed.)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import io, base64
from math import radians, sqrt, cos, sin

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, session, send_file, Response
import json

from gauss_method import (
    Observation, gauss_method, state_to_elements,
    _compute_D, _unit_vector, RE, MU,
)

app = Flask(__name__)
app.secret_key = "gauss-orbit-2026"

F = 1 / 298.257   # WGS-84 flattening


# ─────────────────────────────────────────────────────────────────────────────
# Observer position (Curtis Eq. 5.56)
# ─────────────────────────────────────────────────────────────────────────────
def observer_R_from_lat_lst(phi_deg, H_km, lst_deg):
    phi   = radians(phi_deg)
    theta = radians(lst_deg)
    N = RE / sqrt(1 - (2 * F - F**2) * sin(phi)**2)
    return np.array([
        (N + H_km) * cos(phi) * cos(theta),
        (N + H_km) * cos(phi) * sin(theta),
        (N * (1 - F)**2 + H_km) * sin(phi),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Stumpff functions  (Curtis §3.4)
# ─────────────────────────────────────────────────────────────────────────────
def _C(z):
    if   z > 0: return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0: return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    return 0.5

def _S(z):
    if z > 0:
        s = np.sqrt(z)
        return (s - np.sin(s)) / z**1.5
    elif z < 0:
        s = np.sqrt(-z)
        return (np.sinh(s) - s) / (-z)**1.5
    return 1.0 / 6.0


# ─────────────────────────────────────────────────────────────────────────────
# Universal Kepler equation solver  (Curtis Algorithm 3.3)
# ─────────────────────────────────────────────────────────────────────────────
def _universal_kepler(r0, vr0, alpha, tau, mu=MU, tol=1e-10, max_iter=50):
    """Return universal anomaly χ for time interval tau."""
    sqmu = np.sqrt(mu)
    chi = sqmu * abs(alpha) * tau          # initial guess
    for _ in range(max_iter):
        z   = alpha * chi**2
        C   = _C(z);  S = _S(z)
        F   = (r0 * vr0 / sqmu) * chi**2 * C + (1 - alpha * r0) * chi**3 * S + r0 * chi - sqmu * tau
        dF  = (r0 * vr0 / sqmu) * chi * (1 - z * S) + (1 - alpha * r0) * chi**2 * C + r0
        if abs(dF) < 1e-30:
            break
        delta = F / dF
        chi -= delta
        if abs(delta) < tol:
            break
    return chi


def _fg(chi, tau, r0, alpha, mu=MU):
    """Lagrange f and g from universal anomaly  (Curtis Eqs. 3.69)."""
    z = alpha * chi**2
    f = 1.0 - chi**2 / r0 * _C(z)
    g = tau  - chi**3 / np.sqrt(mu) * _S(z)
    return f, g


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 5.6 — iterative refinement
# ─────────────────────────────────────────────────────────────────────────────
def refine_gauss_56(r2, v2, tau1, tau3, D0, D, L1, L2, L3, R1, R2, R3,
                    n_iter=5, mu=MU):
    """
    Curtis Algorithm 5.6: refine the Gauss preliminary orbit using the
    universal-variable Lagrange f, g coefficients.

    Returns updated (r2, v2, rho1, rho2, rho3).
    """
    rho1 = rho2 = rho3 = 0.0
    for _ in range(n_iter):
        r2_mag = np.linalg.norm(r2)
        v2_mag = np.linalg.norm(v2)
        alpha  = 2.0 / r2_mag - v2_mag**2 / mu
        vr2    = np.dot(v2, r2) / r2_mag

        chi1 = _universal_kepler(r2_mag, vr2, alpha, tau1, mu)
        chi3 = _universal_kepler(r2_mag, vr2, alpha, tau3, mu)

        f1, g1 = _fg(chi1, tau1, r2_mag, alpha, mu)
        f3, g3 = _fg(chi3, tau3, r2_mag, alpha, mu)

        W  = f1 * g3 - f3 * g1
        c1 =  g3 / W
        c3 = -g1 / W

        # Re-solve for slant ranges using the same sign convention as
        # gauss_method.py (/ D0, works correctly when D0 < 0)
        rho1 = (D[0, 1] - c1 * D[0, 0] - c3 * D[0, 2]) / (c1 * D0)
        rho2 = (D[1, 1] - c1 * D[1, 0] - c3 * D[1, 2]) / D0
        rho3 = (D[2, 1] - c1 * D[2, 0] - c3 * D[2, 2]) / (c3 * D0)

        r1 = R1 + rho1 * L1
        r2 = R2 + rho2 * L2
        r3 = R3 + rho3 * L3

        v2 = (-f3 * r1 + f1 * r3) / W

    return r2, v2, rho1, rho2, rho3


# ─────────────────────────────────────────────────────────────────────────────
# Orbit trajectory plot
# ─────────────────────────────────────────────────────────────────────────────
def _orbit_from_elements(elems, n_pts=400):
    """Return (3, n_pts) ECI array of points along the orbit ellipse."""
    a, e = elems["a"], elems["e"]
    # Reject non-elliptical, infinite, or NaN elements (NaN comparisons are
    # always False, so the old single guard silently let NaN through into Q)
    if not np.isfinite(a) or a <= 0 or not np.isfinite(e) or e >= 1:
        return None
    if not all(np.isfinite([elems["i"], elems["raan"], elems["argp"]])):
        return None
    i    = np.radians(elems["i"])
    raan = np.radians(elems["raan"])
    argp = np.radians(elems["argp"])

    nu = np.linspace(0, 2 * np.pi, n_pts)
    p  = a * (1 - e**2)
    r  = p / (1 + e * np.cos(nu))

    # Perifocal frame
    xp = r * np.cos(nu)
    yp = r * np.sin(nu)

    # Rotation: perifocal → ECI  (standard Q_peri_to_eci)
    co, so = np.cos(argp), np.sin(argp)
    ci, si = np.cos(i),    np.sin(i)
    cr, sr = np.cos(raan), np.sin(raan)

    Q = np.array([
        [ cr*co - sr*so*ci,  -cr*so - sr*co*ci,  sr*si],
        [ sr*co + cr*so*ci,  -sr*so + cr*co*ci, -cr*si],
        [ so*si,              co*si,              ci   ],
    ])

    pts_peri = np.vstack([xp, yp, np.zeros(n_pts)])
    # Use np.dot instead of @ — NumPy 2.0 / BLAS bug triggers spurious
    # "divide by zero in matmul" via the @ operator even for finite arrays.
    return np.dot(Q, pts_peri)


def generate_orbit_plot(r_vec, v_vec, Rs=None):
    """Return base64-encoded PNG of the 3-D orbit."""
    # Guard against degenerate input vectors before calling state_to_elements
    if (not np.all(np.isfinite(r_vec)) or not np.all(np.isfinite(v_vec))
            or np.linalg.norm(r_vec) < 1.0 or np.linalg.norm(v_vec) < 1e-12):
        return None
    elems = state_to_elements(r_vec, v_vec)
    pts   = _orbit_from_elements(elems)

    BG = "#0d1117"
    fig = plt.figure(figsize=(6, 5.2), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor(BG)

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    xe = RE * np.outer(np.cos(u), np.sin(v))
    ye = RE * np.outer(np.sin(u), np.sin(v))
    ze = RE * np.outer(np.ones_like(u), np.cos(v))
    # Suppress the same BLAS RuntimeWarning triggered inside matplotlib's
    # normals-shading code (art3d.py) for the Earth sphere surface.
    with np.errstate(divide="ignore", overflow="ignore", invalid="ignore"):
        ax.plot_surface(xe, ye, ze, color="#1e6fa8", alpha=0.55, linewidth=0, zorder=1)

    # Orbit ellipse
    if pts is not None:
        ax.plot(pts[0], pts[1], pts[2], color="#f0a500", lw=1.6, zorder=3)

    # Satellite position r₂
    ax.scatter(*r_vec, color="#56d364", s=45, zorder=5, depthshade=False)

    # Observer positions on Earth surface
    if Rs:
        for R in Rs:
            ax.scatter(*R, color="#ff7b72", s=20, zorder=4, depthshade=False)

    # Equatorial circle guide
    theta_eq = np.linspace(0, 2 * np.pi, 200)
    r_eq = RE * 1.001
    ax.plot(r_eq * np.cos(theta_eq), r_eq * np.sin(theta_eq),
            np.zeros(200), color="#30363d", lw=0.7, zorder=2)

    # Style
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill   = False
        pane.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=6)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color("#8b949e")
    ax.set_xlabel("X (km)", fontsize=7, color="#8b949e", labelpad=2)
    ax.set_ylabel("Y (km)", fontsize=7, color="#8b949e", labelpad=2)
    ax.set_zlabel("Z (km)", fontsize=7, color="#8b949e", labelpad=2)

    lim = np.linalg.norm(r_vec) * 1.25
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Flask route
# ─────────────────────────────────────────────────────────────────────────────
def _f(v):
    return float(v.strip())


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error  = None
    form   = {}

    if request.method == "POST":
        form = request.form
        mode   = form.get("obs_mode", "lat_lst")
        n_iter = max(0, min(20, int(form.get("n_iter", 0) or 0)))

        try:
            observations = []
            Ls, Rs = [], []
            for i in range(1, 4):
                t   = _f(form[f"t{i}"])
                ra  = radians(_f(form[f"ra{i}"]))
                dec = radians(_f(form[f"dec{i}"]))
                L   = _unit_vector(ra, dec)

                if mode == "lat_lst":
                    phi = _f(form["phi"])
                    H   = _f(form["H"])
                    lst = _f(form[f"lst{i}"])
                    R   = observer_R_from_lat_lst(phi, H, lst)
                else:
                    R = np.array([
                        _f(form[f"Rx{i}"]),
                        _f(form[f"Ry{i}"]),
                        _f(form[f"Rz{i}"]),
                    ])

                observations.append(Observation(t=t, ra=ra, dec=dec, R=R))
                Ls.append(L)
                Rs.append(R)

            # ── Algorithm 5.5 ─────────────────────────────────────────
            state = gauss_method(*observations)
            r2, v2 = state.r.copy(), state.v.copy()
            rho1, rho2, rho3 = state.rho

            # ── Algorithm 5.6 (optional refinement) ───────────────────
            if n_iter > 0:
                tau1 = observations[0].t - observations[1].t
                tau3 = observations[2].t - observations[1].t
                D0, D = _compute_D(*Ls, *Rs)
                r2, v2, rho1, rho2, rho3 = refine_gauss_56(
                    r2, v2, tau1, tau3, D0, D,
                    Ls[0], Ls[1], Ls[2],
                    Rs[0], Rs[1], Rs[2],
                    n_iter=n_iter,
                )

            elems = state_to_elements(r2, v2)

            # ── Store inputs in session for Excel download ──────────────
            session["last_inputs"] = {
                "obs": [
                    {"t": ob.t,
                     "ra_deg": float(np.degrees(ob.ra)),
                     "dec_deg": float(np.degrees(ob.dec))}
                    for ob in observations
                ],
                "Rs": [R.tolist() for R in Rs],
                "n_iter": n_iter,
            }

            # ── Orbit plot ─────────────────────────────────────────────
            plot_b64 = generate_orbit_plot(r2, v2, Rs=Rs)

            def fmt3(vec):
                return [f"{v:.4f}" for v in vec]

            result = {
                "r2":      fmt3(r2),
                "r2_mag":  f"{np.linalg.norm(r2):.4f}",
                "v2":      fmt3(v2),
                "v2_mag":  f"{np.linalg.norm(v2):.6f}",
                "rho1":    f"{rho1:.4f}",
                "rho2":    f"{rho2:.4f}",
                "rho3":    f"{rho3:.4f}",
                "a":       f"{elems['a']:.4f}",
                "e":       f"{elems['e']:.6f}",
                "i":       f"{elems['i']:.4f}",
                "raan":    f"{elems['raan']:.4f}",
                "argp":    f"{elems['argp']:.4f}",
                "nu":      f"{elems['nu']:.4f}",
                "T_s":     f"{elems['T']:.2f}",
                "T_hr":    f"{elems['T']/3600:.4f}",
                "t2":      f"{observations[1].t:.2f}",
                "n_iter":  n_iter,
                "plot":    plot_b64,
            }

        except Exception as exc:
            error = str(exc)

    return render_template("index.html", result=result, error=error, form=form)


@app.route("/download")
def download():
    inputs = session.get("last_inputs")
    if not inputs:
        return Response("No computation result available. Run the algorithm first.", 400)
    from excel_export import build_workbook
    inputs["Rs"] = [np.array(R) for R in inputs["Rs"]]
    data = build_workbook(inputs)
    return send_file(
        io.BytesIO(data),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="gauss_orbit.xlsx",
    )


if __name__ == "__main__":
    app.run(debug=True)
