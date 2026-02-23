#!/usr/bin/env python3
"""
6D Stiff RLC Ladder: LGN-SD vs SymODENet
==========================================
3-section RLC with two cases:
  Uniform: R = [0.5, 0.5, 0.5] → all modes similar → SymODENet matches LGN
  Stiff:   R = [0.01, 0.1, 1.0] → 100× spread → SymODENet loses slow modes

Key insight: τ-horizon training sees ~0.25% of the slow mode's time constant.
Within that window, slow modes look constant → no gradient signal → lost.
Full-trajectory + matrix_exp sees the full decay → recovers everything.

Methods:
  LGN-SD:    A = S - D (skew + pos diag), matrix_exp, full trajectory
  SymODENet: A = (J - R)M (port-Hamiltonian), RK4, τ-horizon

Both guarantee stability. Same optimizer (L-BFGS-B). Same data.
The only difference is information content: full trajectory vs τ-window.

Pure numpy/scipy — runs in ~5 min total.
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, time, sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# =================================================================
# CONFIG
# =================================================================
N_SEC = 3
DIM = 2 * N_SEC  # 6
L_VALS = [1.0] * N_SEC
C_VALS = [1.0] * N_SEC

TRAIN_T = 50.0
TEST_T = 200.0
DT = 0.1
TAU = 5            # same τ that works for uniform case
N_IC = 5
SEED = 42
N_RESTARTS = 5
LGN_SUBSAMPLE = 5  # use every 5th training point (same info, 5× faster)

# Param counts
N_UPPER = DIM * (DIM - 1) // 2  # 15
N_LGN = N_UPPER + DIM            # 21
N_PH = N_UPPER + 2 * DIM         # 27


# =================================================================
# PHYSICS
# =================================================================
def build_A(R_vals):
    A = np.zeros((DIM, DIM))
    for k in range(N_SEC):
        v, i = 2*k, 2*k + 1
        if k > 0:
            A[v, 2*(k-1)+1] = 1.0 / C_VALS[k]
        A[v, i] = -1.0 / C_VALS[k]
        A[i, v] = 1.0 / L_VALS[k]
        if k < N_SEC - 1:
            A[i, 2*(k+1)] = -1.0 / L_VALS[k]
        A[i, i] = -R_vals[k] / L_VALS[k]
    return A


def energy(x):
    E = np.zeros(x.shape[0]) if x.ndim == 2 else 0.0
    for k in range(N_SEC):
        v, i = 2*k, 2*k + 1
        if x.ndim == 1:
            E += 0.5*C_VALS[k]*x[v]**2 + 0.5*L_VALS[k]*x[i]**2
        else:
            E += 0.5*C_VALS[k]*x[:, v]**2 + 0.5*L_VALS[k]*x[:, i]**2
    return E


def gen_train(A, T, dt, n_ic, seed):
    rng = np.random.RandomState(seed)
    t = np.arange(0, T, dt)
    x0s, trajs = [], []
    for _ in range(n_ic):
        x0 = rng.randn(DIM) * 0.5
        x0 /= np.sqrt(energy(x0) + 1e-6)
        sol = solve_ivp(lambda t, y: A @ y, [0, T], x0,
                        t_eval=t, method='RK45', rtol=1e-10, atol=1e-12)
        x0s.append(x0)
        trajs.append(sol.y.T)
    return t, np.array(x0s), np.array(trajs)


def gen_test(A, T, dt):
    x0 = np.zeros(DIM); x0[0] = 1.0; x0[1] = 0.5
    t = np.arange(0, T, dt)
    sol = solve_ivp(lambda t, y: A @ y, [0, T], x0,
                    t_eval=t, method='RK45', rtol=1e-10, atol=1e-12)
    return sol.t, sol.y.T


# =================================================================
# LGN-SD: A = S - D, full trajectory, matrix_exp
# =================================================================
def lgn_unpack(params):
    S = np.zeros((DIM, DIM))
    idx = 0
    for i in range(DIM):
        for j in range(i+1, DIM):
            S[i, j] = params[idx]
            S[j, i] = -params[idx]
            idx += 1
    d = np.log1p(np.exp(params[N_UPPER:N_UPPER+DIM]))
    return S - np.diag(d)


def lgn_loss(params, t_sub, X0, X_true_sub):
    A = lgn_unpack(params)
    dt_sub = t_sub[1] - t_sub[0]
    eAdt = expm(A * dt_sub)
    eAdtT = eAdt.T
    n_ic, n_sub, dim = X_true_sub.shape
    X_pred = np.zeros_like(X_true_sub)
    X_pred[:, 0, :] = X0
    for i in range(1, n_sub):
        X_pred[:, i, :] = X_pred[:, i-1, :] @ eAdtT
    return np.mean((X_pred - X_true_sub)**2)


def propagate(A, t, x0):
    dt = t[1] - t[0]
    eAdt = expm(A * dt)
    pred = np.zeros((len(t), DIM))
    pred[0] = x0
    for i in range(1, len(t)):
        pred[i] = eAdt @ pred[i-1]
    return pred


# =================================================================
# SymODENet (pH-ODE): A = (J-R)M, τ-horizon, RK4
# =================================================================
def ph_unpack(params):
    J = np.zeros((DIM, DIM))
    idx = 0
    for i in range(DIM):
        for j in range(i+1, DIM):
            J[i, j] = params[idx]
            J[j, i] = -params[idx]
            idx += 1
    R = np.diag(np.exp(params[N_UPPER:N_UPPER+DIM]))
    M = np.diag(np.exp(params[N_UPPER+DIM:N_UPPER+2*DIM]))
    return J, R, M


def ph_A_eff(params):
    J, R, M = ph_unpack(params)
    return (J - R) @ M


def ph_tau_loss(params, W0, W_target, tau):
    """Batched τ-horizon loss: all windows propagated simultaneously."""
    J, R, M = ph_unpack(params)
    AT = ((J - R) @ M).T
    pred = np.zeros_like(W_target)
    pred[:, 0, :] = W0
    for step in range(tau):
        x = pred[:, step, :]
        k1 = x @ AT
        k2 = (x + 0.5*DT*k1) @ AT
        k3 = (x + 0.5*DT*k2) @ AT
        k4 = (x + DT*k3) @ AT
        pred[:, step+1, :] = x + (DT/6)*(k1 + 2*k2 + 2*k3 + k4)
    return np.mean((pred - W_target)**2)


def make_windows(trajs, tau):
    starts, targets = [], []
    for traj in trajs:
        for si in range(0, traj.shape[0] - tau, tau):
            starts.append(traj[si])
            targets.append(traj[si:si+tau+1])
    return np.array(starts), np.array(targets)


# =================================================================
# EIGENVALUE TOOLS
# =================================================================
def match_eigs(eig_true_sorted, eig_pred):
    cost = np.abs(eig_true_sorted[:, None] - eig_pred[None, :])
    _, c = linear_sum_assignment(cost)
    matched = eig_pred[c]
    return matched, np.abs(eig_true_sorted - matched)


def eig_rmse(eig_true, eig_pred):
    cost = np.abs(eig_true[:, None] - eig_pred[None, :])
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(np.mean(np.abs(eig_true[r] - eig_pred[c])**2)))


def slow_rmse(eig_true, eig_pred, n_slow=2):
    idx = np.argsort(eig_true.real)[::-1][:n_slow]
    slow = eig_true[idx]
    used = set()
    errs = []
    for z in slow:
        dists = np.abs(eig_pred - z)
        for j in used: dists[j] = np.inf
        j = np.argmin(dists); used.add(j)
        errs.append(eig_pred[j] - z)
    return float(np.sqrt(np.mean(np.abs(np.array(errs))**2)))


# =================================================================
# OPTIMIZER WITH RESTARTS
# =================================================================
def optimize(loss_fn, n_params, n_restarts, seed, label, args):
    best, best_loss = None, np.inf
    rng = np.random.RandomState(seed)
    t_total = time.time()
    for r in range(n_restarts):
        p0 = rng.randn(n_params) * 0.1
        t0 = time.time()
        res = minimize(loss_fn, p0, args=args, method='L-BFGS-B',
                       options={'maxiter': 3000, 'maxfun': 200000,
                                'ftol': 1e-15, 'gtol': 1e-12})
        print(f"  [{label}] {r+1}/{n_restarts}: loss={res.fun:.6e}  "
              f"nit={res.nit}  {time.time()-t0:.1f}s  "
              f"{'✓' if res.success else '✗ '+res.message[:40]}", flush=True)
        if res.fun < best_loss:
            best_loss = res.fun; best = res
    print(f"  [{label}] Best: {best_loss:.6e}  ({time.time()-t_total:.1f}s total)", flush=True)
    return best, time.time() - t_total


# =================================================================
# RUN ONE CASE
# =================================================================
def run_case(R_vals, case_name, out_dir):
    print(f"\n{'#'*70}")
    print(f"  {case_name}  |  R = {R_vals}")
    print(f"{'#'*70}")

    A_true = build_A(R_vals)
    eig_true = np.linalg.eigvals(A_true)
    sort_idx = np.argsort(eig_true.real)
    eig_sorted = eig_true[sort_idx]

    print(f"\nTrue eigenvalues:")
    for i, e in enumerate(eig_sorted):
        tc = -1/e.real if e.real < 0 else np.inf
        vis = TAU * DT / tc * 100
        print(f"  λ_{i}: {e.real:+9.5f} {e.imag:+9.5f}j   τ={tc:8.1f}   "
              f"window sees {vis:.1f}%")

    # Data
    t_tr, X0, trajs = gen_train(A_true, TRAIN_T, DT, N_IC, SEED)
    t_te, x_te = gen_test(A_true, TEST_T, DT)

    t_sub = t_tr[::LGN_SUBSAMPLE]
    trajs_sub = trajs[:, ::LGN_SUBSAMPLE, :]
    W0, Wt = make_windows(trajs, TAU)
    print(f"\nTrain: {trajs.shape[1]} steps × {N_IC} ICs  "
          f"(LGN: {trajs_sub.shape[1]} subsampled)")
    print(f"SymODENet: {W0.shape[0]} windows × τ={TAU}")
    print(f"Test: {len(t_te)} steps, T={TEST_T}\n")

    # ---- LGN-SD ----
    print(f"{'='*60}")
    print(f"LGN-SD  ({N_LGN} params)  full trajectory + matrix_exp")
    print(f"{'='*60}")
    res_l, time_l = optimize(lgn_loss, N_LGN, N_RESTARTS, SEED,
                              "LGN", (t_sub, X0, trajs_sub))
    A_lgn = lgn_unpack(res_l.x)
    eig_lgn = np.linalg.eigvals(A_lgn)

    # ---- SymODENet ----
    print(f"\n{'='*60}")
    print(f"SymODENet  ({N_PH} params)  τ={TAU} horizon + RK4")
    print(f"{'='*60}")
    res_p, time_p = optimize(ph_tau_loss, N_PH, N_RESTARTS, SEED+100,
                              "SymODENet", (W0, Wt, TAU))
    A_ph = ph_A_eff(res_p.x)
    eig_ph = np.linalg.eigvals(A_ph)

    # ---- Evaluate ----
    pred_l = propagate(A_lgn, t_te, x_te[0])
    pred_p = propagate(A_ph, t_te, x_te[0])

    rmse_l = float(np.sqrt(np.mean((pred_l - x_te)**2)))
    rmse_p = float(np.sqrt(np.mean((pred_p - x_te)**2)))
    crmse_l = eig_rmse(eig_true, eig_lgn)
    crmse_p = eig_rmse(eig_true, eig_ph)
    srmse_l = slow_rmse(eig_true, eig_lgn)
    srmse_p = slow_rmse(eig_true, eig_ph)
    aerr_l = float(np.linalg.norm(A_lgn - A_true, 'fro') /
                   np.linalg.norm(A_true, 'fro'))
    aerr_p = float(np.linalg.norm(A_ph - A_true, 'fro') /
                   np.linalg.norm(A_true, 'fro'))
    stab_l = bool(np.all(eig_lgn.real <= 1e-8))
    stab_p = bool(np.all(eig_ph.real <= 1e-8))

    # ---- Per-eigenvalue detail ----
    lgn_matched, lgn_errs = match_eigs(eig_sorted, eig_lgn)
    ph_matched, ph_errs = match_eigs(eig_sorted, eig_ph)

    print(f"\n{'='*70}")
    print(f"  RESULTS: {case_name}")
    print(f"{'='*70}")
    print(f"  {'Method':<14} {'Params':<8} {'RMSE':<12} {'λ RMSE':<12} "
          f"{'λ slow':<12} {'||ΔA||':<12} {'Stable'}")
    print(f"  {'-'*76}")
    print(f"  {'LGN-SD':<14} {N_LGN:<8} {rmse_l:<12.4e} {crmse_l:<12.4e} "
          f"{srmse_l:<12.4e} {aerr_l:<12.4e} {stab_l}")
    print(f"  {'SymODENet':<14} {N_PH:<8} {rmse_p:<12.4e} {crmse_p:<12.4e} "
          f"{srmse_p:<12.4e} {aerr_p:<12.4e} {stab_p}")

    print(f"\n  Per-eigenvalue errors:")
    print(f"  {'True λ':<28} {'LGN err':<14} {'Sym err':<14} {'τ_mode':<10} {'Window %'}")
    print(f"  {'-'*78}")
    for i in range(DIM):
        e = eig_sorted[i]
        tc = -1/e.real if e.real < 0 else np.inf
        vis = TAU * DT / tc * 100
        print(f"  {e.real:+8.5f}{e.imag:+8.5f}j   "
              f"{lgn_errs[i]:<14.4e} {ph_errs[i]:<14.4e} "
              f"{tc:<10.1f} {vis:.1f}%")

    if crmse_l > 0:
        print(f"\n  Ratios:  λ RMSE: SymODENet/LGN = {crmse_p/crmse_l:.1f}×")
    if srmse_l > 0:
        print(f"           λ slow: SymODENet/LGN = {srmse_p/srmse_l:.1f}×")

    # ---- Save ----
    d = out_dir / case_name; d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d/'A_true.csv', A_true, delimiter=',')
    np.savetxt(d/'A_lgn_sd.csv', A_lgn, delimiter=',')
    np.savetxt(d/'A_symoden.csv', A_ph, delimiter=',')

    for lbl, eig in [('true', eig_true), ('lgn', eig_lgn), ('sym', eig_ph)]:
        np.savetxt(d/f'eig_{lbl}.csv', np.column_stack([eig.real, eig.imag]),
                   delimiter=',', header='re,im')

    eig_detail = np.column_stack([
        eig_sorted.real, eig_sorted.imag,
        lgn_matched.real, lgn_matched.imag, lgn_errs,
        ph_matched.real, ph_matched.imag, ph_errs])
    np.savetxt(d/'eig_detail.csv', eig_detail, delimiter=',',
               header='true_re,true_im,lgn_re,lgn_im,lgn_err,sym_re,sym_im,sym_err')

    hdr = ','.join(['t'] + [f'x{i}' for i in range(DIM)])
    np.savetxt(d/'test_truth.csv', np.column_stack([t_te, x_te]),
               delimiter=',', header=hdr)
    np.savetxt(d/'test_lgn.csv', np.column_stack([t_te, pred_l]),
               delimiter=',', header=hdr)
    np.savetxt(d/'test_sym.csv', np.column_stack([t_te, pred_p]),
               delimiter=',', header=hdr)

    rmse_l_t = np.sqrt(np.mean((pred_l - x_te)**2, axis=1))
    rmse_p_t = np.sqrt(np.mean((pred_p - x_te)**2, axis=1))
    np.savetxt(d/'rmse_vs_t.csv', np.column_stack([t_te, rmse_l_t, rmse_p_t]),
               delimiter=',', header='t,lgn,symoden')

    E_tr = energy(x_te); E_l = energy(pred_l); E_p = energy(pred_p)
    np.savetxt(d/'energy.csv', np.column_stack([t_te, E_tr, E_l, E_p]),
               delimiter=',', header='t,true,lgn,symoden')

    summary = {
        'config': {'R': R_vals, 'dim': DIM, 'tau': TAU, 'dt': DT,
                   'train_T': TRAIN_T, 'test_T': TEST_T},
        'lgn_sd': {'params': N_LGN, 'time': time_l, 'loss': float(res_l.fun),
                   'rmse': rmse_l, 'A_err': aerr_l, 'c_rmse': crmse_l,
                   'slow_rmse': srmse_l, 'stable': stab_l},
        'symoden': {'params': N_PH, 'time': time_p, 'loss': float(res_p.fun),
                    'rmse': rmse_p, 'A_err': aerr_p, 'c_rmse': crmse_p,
                    'slow_rmse': srmse_p, 'stable': stab_p},
    }
    with open(d/'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return {
        't_te': t_te, 'x_te': x_te, 'pred_l': pred_l, 'pred_p': pred_p,
        'eig_true': eig_true, 'eig_lgn': eig_lgn, 'eig_ph': eig_ph,
        'eig_sorted': eig_sorted, 'lgn_matched': lgn_matched, 'ph_matched': ph_matched,
        'lgn_errs': lgn_errs, 'ph_errs': ph_errs,
        'rmse_l_t': rmse_l_t, 'rmse_p_t': rmse_p_t,
        'summary': summary,
    }


# =================================================================
# FIGURE
# =================================================================
def make_figure(data_u, data_s, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    colors = {'lgn': '#2196F3', 'sym': '#E53935', 'true': '#222222'}

    for row, (label, d) in enumerate([
        ("Uniform R = [0.5, 0.5, 0.5]", data_u),
        ("Stiff R = [0.01, 0.1, 1.0]", data_s)]):

        # --- Eigenvalue plane ---
        ax = axes[row, 0]
        ax.scatter(d['eig_true'].real, d['eig_true'].imag,
                   s=120, marker='o', facecolors='none', edgecolors=colors['true'],
                   linewidths=2, label='True', zorder=3)
        ax.scatter(d['eig_lgn'].real, d['eig_lgn'].imag,
                   s=50, marker='^', color=colors['lgn'], label='LGN-SD', zorder=2)
        ax.scatter(d['eig_ph'].real, d['eig_ph'].imag,
                   s=50, marker='s', color=colors['sym'], label='SymODENet', zorder=2)
        ax.axvline(0, color='gray', ls='--', alpha=0.3)
        ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
        ax.set_title(f'{label}\nEigenvalues')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

        # --- Trajectory ---
        ax = axes[row, 1]
        ax.plot(d['t_te'], d['x_te'][:, 0], '-', color=colors['true'],
                lw=1.5, label='True', alpha=0.8)
        ax.plot(d['t_te'], d['pred_l'][:, 0], '--', color=colors['lgn'],
                lw=1, label='LGN-SD')
        ax.plot(d['t_te'], d['pred_p'][:, 0], '--', color=colors['sym'],
                lw=1, label='SymODENet')
        ax.set_xlabel('t'); ax.set_ylabel('x₀ (voltage)')
        ax.set_title('Test Trajectory (x₀)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

        # --- RMSE vs time ---
        ax = axes[row, 2]
        ax.semilogy(d['t_te'], d['rmse_l_t'], color=colors['lgn'],
                    lw=1.5, label='LGN-SD')
        ax.semilogy(d['t_te'], d['rmse_p_t'], color=colors['sym'],
                    lw=1.5, label='SymODENet')
        ax.set_xlabel('t'); ax.set_ylabel('RMSE(t)')
        ax.set_title('Prediction Error vs Time')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_dir / 'comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure → {out_dir}/comparison.png")


# =================================================================
# MAIN
# =================================================================
def main():
    out = Path('./exp5_stiff_6d'); out.mkdir(exist_ok=True)

    print("="*70)
    print("  EXP 5: 6D RLC — LGN-SD vs SymODENet")
    print(f"  dim={DIM}  τ={TAU}  dt={DT}  T_train={TRAIN_T}  T_test={TEST_T}")
    print(f"  LGN: {N_LGN} params  |  SymODENet: {N_PH} params")
    print("="*70)

    data_u = run_case([0.5, 0.5, 0.5], "uniform", out)
    data_s = run_case([0.01, 0.1, 1.0], "stiff", out)

    make_figure(data_u, data_s, out)

    # Final comparison
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    su, ss = data_u['summary'], data_s['summary']
    print(f"  Uniform:  λ RMSE ratio (Sym/LGN) = "
          f"{su['symoden']['c_rmse']/(su['lgn_sd']['c_rmse']+1e-30):.1f}×")
    print(f"  Stiff:    λ RMSE ratio (Sym/LGN) = "
          f"{ss['symoden']['c_rmse']/(ss['lgn_sd']['c_rmse']+1e-30):.1f}×")
    print(f"  Stiff:    slow-mode ratio         = "
          f"{ss['symoden']['slow_rmse']/(ss['lgn_sd']['slow_rmse']+1e-30):.1f}×")

    print("\nDone!")


if __name__ == '__main__':
    main()
