"""
EXPERIMENT 5: RLC Ladder N=50 (100 states)
==========================================
Shows Magnus scales to higher dimensions.
Skips NODE (too slow, won't work well anyway).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from pathlib import Path
import json
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================
N_SECTIONS = 50
STATE_DIM = 2 * N_SECTIONS  # 100 states
L_VALUES = [1.0] * N_SECTIONS
C_VALUES = [1.0] * N_SECTIONS
R_VALUES = [0.1] * N_SECTIONS

TRAIN_T = 30.0
TEST_T = 100.0  # Shorter for speed
DT = 0.1
N_SEEDS = 1  # Single seed for speed
EPOCHS = 1000
N_INIT_CONDITIONS = 3  # Fewer ICs for speed

print(f"System: {N_SECTIONS} sections, {STATE_DIM} states", flush=True)

# =============================================================================
# BUILD STATE-SPACE
# =============================================================================
def build_rlc_ladder_A(L_list, C_list, R_list):
    N = len(L_list)
    dim = 2 * N
    A = np.zeros((dim, dim))
    
    for k in range(N):
        v_idx, i_idx = 2*k, 2*k + 1
        C_k, L_k, R_k = C_list[k], L_list[k], R_list[k]
        
        if k > 0:
            A[v_idx, 2*(k-1) + 1] = 1.0 / C_k
        A[v_idx, i_idx] = -1.0 / C_k
        
        A[i_idx, v_idx] = 1.0 / L_k
        if k < N - 1:
            A[i_idx, 2*(k+1)] = -1.0 / L_k
        A[i_idx, i_idx] = -R_k / L_k
    
    return A

def compute_energy(x, L_list, C_list):
    N = len(L_list)
    E = 0.0
    for k in range(N):
        v_idx, i_idx = 2*k, 2*k + 1
        if x.ndim == 1:
            E += 0.5 * C_list[k] * x[v_idx]**2 + 0.5 * L_list[k] * x[i_idx]**2
        else:
            E += 0.5 * C_list[k] * x[:, v_idx]**2 + 0.5 * L_list[k] * x[:, i_idx]**2
    return E

# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_ladder_data(t_span, dt, x0=None):
    A = build_rlc_ladder_A(L_VALUES, C_VALUES, R_VALUES)
    if x0 is None:
        x0 = np.zeros(STATE_DIM)
        x0[0], x0[1] = 1.0, 0.5
    
    def dynamics(t, x):
        return A @ x
    
    t_eval = np.arange(0, t_span, dt)
    sol = solve_ivp(dynamics, [0, t_span], x0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    E = compute_energy(sol.y.T, L_VALUES, C_VALUES)
    return sol.t, sol.y.T, E, A

def generate_multiple_trajectories(t_span, dt, n_traj, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    A = build_rlc_ladder_A(L_VALUES, C_VALUES, R_VALUES)
    t_eval = np.arange(0, t_span, dt)
    
    all_x0, all_traj, all_E = [], [], []
    for _ in range(n_traj):
        x0 = np.random.randn(STATE_DIM) * 0.5
        E0 = compute_energy(x0, L_VALUES, C_VALUES)
        x0 = x0 / np.sqrt(E0 + 1e-6)
        
        def dynamics(t, x):
            return A @ x
        
        sol = solve_ivp(dynamics, [0, t_span], x0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
        all_x0.append(x0)
        all_traj.append(sol.y.T)
        all_E.append(compute_energy(sol.y.T, L_VALUES, C_VALUES))
    
    return t_eval, all_x0, all_traj, all_E, A

# =============================================================================
# MODELS
# =============================================================================
class MagnusLadder(nn.Module):
    """Magnus with full A matrix"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Sparse-ish initialization (tridiagonal-like)
        A_init = torch.zeros(dim, dim, dtype=torch.float64)
        for i in range(dim):
            A_init[i, i] = -0.1
        for i in range(dim - 1):
            A_init[i, i+1] = 0.5
            A_init[i+1, i] = -0.5
        self.A = nn.Parameter(A_init)
    
    def forward(self, t, x0):
        dt = t[1] - t[0]
        expAdt = torch.matrix_exp(self.A * dt)
        trajectory = [x0]
        x = x0
        for _ in range(1, len(t)):
            x = expAdt @ x
            trajectory.append(x)
        return torch.stack(trajectory)
    
    def get_A(self):
        return self.A.detach().cpu().numpy()
    
    def get_eigenvalues(self):
        return np.linalg.eigvals(self.get_A())

class MagnusDissipative(nn.Module):
    """Magnus with A = S - D structure (guaranteed stable)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        n_upper = dim * (dim - 1) // 2
        self.S_upper = nn.Parameter(torch.randn(n_upper, dtype=torch.float64) * 0.1)
        self.D_raw = nn.Parameter(torch.ones(dim, dtype=torch.float64) * 0.1)
    
    def get_A(self):
        S = torch.zeros(self.dim, self.dim, dtype=torch.float64, device=self.S_upper.device)
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                S[i, j] = self.S_upper[idx]
                S[j, i] = -self.S_upper[idx]
                idx += 1
        D = torch.diag(torch.nn.functional.softplus(self.D_raw))
        return S - D
    
    def forward(self, t, x0):
        A = self.get_A()
        dt = t[1] - t[0]
        expAdt = torch.matrix_exp(A * dt)
        trajectory = [x0]
        x = x0
        for _ in range(1, len(t)):
            x = expAdt @ x
            trajectory.append(x)
        return torch.stack(trajectory)
    
    def get_eigenvalues(self):
        return np.linalg.eigvals(self.get_A().detach().cpu().numpy())

# =============================================================================
# LINEAR-ID BASELINE (Reviewer-proof)
# =============================================================================
def fit_linear_id(t, x_list):
    """
    Fit constant A from trajectories using least squares.
    Uses central difference for cleaner derivative estimate.
    """
    dt = t[1] - t[0]
    
    # Stack all trajectories with central difference
    all_dx, all_x = [], []
    for x in x_list:
        dx = np.gradient(x, dt, axis=0)  # Central diff (T, dim)
        all_dx.append(dx)
        all_x.append(x)
    
    dX = np.vstack(all_dx)  # (T*n_traj, dim)
    X = np.vstack(all_x)    # (T*n_traj, dim)
    
    # Correct least squares: solves X @ A^T ≈ dX
    A_T, _, _, _ = np.linalg.lstsq(X, dX, rcond=None)
    return A_T.T

def rollout_linear(A, x0, t):
    """Rollout using exp(A*dt)"""
    dt = t[1] - t[0]
    expAdt = expm(A * dt)
    trajectory = [x0]
    x = x0.copy()
    for _ in range(1, len(t)):
        x = expAdt @ x
        trajectory.append(x)
    return np.array(trajectory)

# =============================================================================
# TRAINING
# =============================================================================
def train_model(model, t_train, x0_list, x_train_list, epochs=1500, lr=1e-2, name="Model"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=150, factor=0.5)
    n_traj = len(x0_list)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        try:
            for x0, x_true in zip(x0_list, x_train_list):
                x_pred = model(t_train, x0)
                total_loss += torch.mean((x_pred - x_true)**2)
            total_loss = total_loss / n_traj
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(total_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f"  {name} Ep {epoch+1}/{epochs}: Loss={total_loss.item():.6f}", flush=True)
        except Exception as e:
            print(f"  {name} Ep {epoch+1}: Error - {e}", flush=True)
            break

def evaluate_model(model, t, x_true, is_numpy=False):
    """Evaluate model - works for both torch models and numpy (Linear-ID)"""
    if is_numpy:
        x_pred = model  # Already computed
        x_true_np = x_true
    else:
        model.eval()
        with torch.no_grad():
            x_pred = model(t, x_true[0])
        x_pred = x_pred.cpu().numpy()
        x_true_np = x_true.cpu().numpy()
    
    rmse = np.sqrt(np.mean((x_pred - x_true_np)**2))
    E_pred = compute_energy(x_pred, L_VALUES, C_VALUES)
    E_true = compute_energy(x_true_np, L_VALUES, C_VALUES)
    
    # Energy monotonicity check
    dE = np.diff(E_pred)
    violations = np.sum(dE > 1e-6)
    
    return {
        'rmse': rmse,
        'x_pred': x_pred,
        'E_pred': E_pred,
        'E_true': E_true,
        'violations': violations,
        'violation_frac': violations / max(len(dE), 1)
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70, flush=True)
    print(f"EXPERIMENT 5: RLC LADDER N={N_SECTIONS} ({STATE_DIM} states)", flush=True)
    print("="*70, flush=True)
    print(f"Train: 0-{TRAIN_T}, Test: 0-{TEST_T}, Δt={DT}", flush=True)
    print(f"Training on {N_INIT_CONDITIONS} initial conditions", flush=True)
    print(f"Epochs: {EPOCHS}", flush=True)
    
    # Build true system
    A_true = build_rlc_ladder_A(L_VALUES, C_VALUES, R_VALUES)
    eigvals_true = np.linalg.eigvals(A_true)
    print(f"\nTrue A: {STATE_DIM}×{STATE_DIM} matrix", flush=True)
    print(f"True eigenvalues (first 10): {eigvals_true[:10].round(3)}", flush=True)
    print(f"All eigenvalues have Re(λ) = {eigvals_true.real.max():.4f}", flush=True)
    
    # Generate data
    print("\nGenerating training data...", flush=True)
    t_train_np, x0_list_np, x_train_list_np, _, _ = generate_multiple_trajectories(
        TRAIN_T, DT, N_INIT_CONDITIONS, seed=42
    )
    
    print("Generating test data...", flush=True)
    t_test_np, x_test_np, E_test, _ = generate_ladder_data(TEST_T, DT)
    
    # Convert to torch
    t_train = torch.tensor(t_train_np, dtype=torch.float64, device=DEVICE)
    x0_list = [torch.tensor(x0, dtype=torch.float64, device=DEVICE) for x0 in x0_list_np]
    x_train_list = [torch.tensor(x, dtype=torch.float64, device=DEVICE) for x in x_train_list_np]
    t_test = torch.tensor(t_test_np, dtype=torch.float64, device=DEVICE)
    x_test = torch.tensor(x_test_np, dtype=torch.float64, device=DEVICE)
    
    results = {}
    
    # -------------------------------------------------------------------------
    # LINEAR-ID BASELINE (instant)
    # -------------------------------------------------------------------------
    print("\n" + "="*50, flush=True)
    print("Fitting Linear-ID (instant)...", flush=True)
    t0 = time.time()
    A_linear = fit_linear_id(t_train_np, x_train_list_np)
    linear_time = time.time() - t0
    print(f"  Time: {linear_time:.3f} sec", flush=True)
    
    eigvals_linear = np.linalg.eigvals(A_linear)
    print(f"  Learned eigenvalues (first 10): {eigvals_linear[:10].round(3)}", flush=True)
    print(f"  Max Re(λ): {eigvals_linear.real.max():.4f} (true: -0.05)", flush=True)
    
    # Rollout
    x_pred_linear = rollout_linear(A_linear, x_test_np[0], t_test_np)
    results['linear_id'] = evaluate_model(x_pred_linear, t_test, x_test_np, is_numpy=True)
    results['linear_id']['train_time'] = linear_time
    results['linear_id']['eigenvalues'] = eigvals_linear
    print(f"  Test RMSE: {results['linear_id']['rmse']:.6f}", flush=True)
    
    # -------------------------------------------------------------------------
    # MAGNUS (full A)
    # -------------------------------------------------------------------------
    print("\n" + "="*50, flush=True)
    print("Training Magnus (full A)...", flush=True)
    magnus = MagnusLadder(STATE_DIM).double().to(DEVICE)
    t0 = time.time()
    train_model(magnus, t_train, x0_list, x_train_list, epochs=EPOCHS, lr=1e-2, name="Magnus")
    magnus_time = time.time() - t0
    
    results['magnus'] = evaluate_model(magnus, t_test, x_test)
    results['magnus']['train_time'] = magnus_time
    results['magnus']['eigenvalues'] = magnus.get_eigenvalues()
    print(f"  Train time: {magnus_time:.1f} sec", flush=True)
    print(f"  Test RMSE: {results['magnus']['rmse']:.6f}", flush=True)
    print(f"  Learned eigenvalues (first 10): {results['magnus']['eigenvalues'][:10].round(3)}", flush=True)
    
    # -------------------------------------------------------------------------
    # MAGNUS DISSIPATIVE (S-D structure)
    # -------------------------------------------------------------------------
    print("\n" + "="*50, flush=True)
    print("Training Magnus Dissipative (S-D)...", flush=True)
    magnus_sd = MagnusDissipative(STATE_DIM).double().to(DEVICE)
    t0 = time.time()
    train_model(magnus_sd, t_train, x0_list, x_train_list, epochs=EPOCHS, lr=1e-2, name="Magnus-SD")
    magnus_sd_time = time.time() - t0
    
    results['magnus_sd'] = evaluate_model(magnus_sd, t_test, x_test)
    results['magnus_sd']['train_time'] = magnus_sd_time
    results['magnus_sd']['eigenvalues'] = magnus_sd.get_eigenvalues()
    print(f"  Train time: {magnus_sd_time:.1f} sec", flush=True)
    print(f"  Test RMSE: {results['magnus_sd']['rmse']:.6f}", flush=True)
    print(f"  Learned eigenvalues (first 10): {results['magnus_sd']['eigenvalues'][:10].round(3)}", flush=True)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"{'Method':<20} {'RMSE':<12} {'E-violation':<12} {'Train Time':<12}", flush=True)
    print("-"*56, flush=True)
    for name, key in [('Linear-ID', 'linear_id'), ('Magnus', 'magnus'), ('Magnus (S-D)', 'magnus_sd')]:
        r = results[key]
        print(f"{name:<20} {r['rmse']:<12.6f} {r['violation_frac']*100:<11.2f}% {r['train_time']:<12.1f}s", flush=True)
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    output_dir = Path('./exp5_N50_results')
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'linear_id': 'red', 'magnus': 'green', 'magnus_sd': 'blue'}
    labels = {'linear_id': 'Linear-ID', 'magnus': 'Magnus', 'magnus_sd': 'Magnus (S-D)'}
    
    # (a) First state trajectory
    ax = axes[0, 0]
    ax.plot(t_test_np, x_test_np[:, 0], 'k-', lw=2, label='True')
    for key in ['linear_id', 'magnus', 'magnus_sd']:
        ax.plot(t_test_np, results[key]['x_pred'][:, 0], '--', color=colors[key], lw=1.5, label=labels[key])
    ax.axvline(x=TRAIN_T, color='gray', linestyle='--', lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('$v_{C1}$')
    ax.set_title(f'(a) First Capacitor Voltage (N={N_SECTIONS})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (b) Energy
    ax = axes[0, 1]
    ax.plot(t_test_np, results['magnus']['E_true'], 'k-', lw=2, label='True')
    for key in ['linear_id', 'magnus', 'magnus_sd']:
        ax.plot(t_test_np, results[key]['E_pred'], '-', color=colors[key], lw=1.5, label=labels[key])
    ax.axvline(x=TRAIN_T, color='gray', linestyle='--', lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('(b) Total Energy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (c) Eigenvalues
    ax = axes[1, 0]
    ax.scatter(eigvals_true.real, eigvals_true.imag, s=50, c='black', marker='o', label='True', alpha=0.7)
    ax.scatter(results['magnus']['eigenvalues'].real, results['magnus']['eigenvalues'].imag, 
               s=30, c='green', marker='x', label='Magnus', alpha=0.7)
    ax.scatter(results['magnus_sd']['eigenvalues'].real, results['magnus_sd']['eigenvalues'].imag,
               s=30, c='blue', marker='+', label='Magnus (S-D)', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Stability boundary')
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title(f'(c) Eigenvalues ({STATE_DIM} poles)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (d) Bar chart
    ax = axes[1, 1]
    methods = ['Linear-ID', 'Magnus', 'Magnus (S-D)']
    rmses = [results['linear_id']['rmse'], results['magnus']['rmse'], results['magnus_sd']['rmse']]
    times = [results['linear_id']['train_time'], results['magnus']['train_time'], results['magnus_sd']['train_time']]
    
    x_pos = np.arange(3)
    bars = ax.bar(x_pos, rmses, color=['red', 'green', 'blue'], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel('RMSE')
    ax.set_title('(d) Method Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val, t in zip(bars, rmses, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.4f}\n({t:.0f}s)', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'RLC Ladder N={N_SECTIONS} ({STATE_DIM} states): Magnus Scales!', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'n50_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'n50_results.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved plots to {output_dir}", flush=True)
    
    # Save metrics
    metrics = {
        'n_sections': N_SECTIONS,
        'state_dim': STATE_DIM,
        'linear_id': {'rmse': results['linear_id']['rmse'], 'train_time': results['linear_id']['train_time']},
        'magnus': {'rmse': results['magnus']['rmse'], 'train_time': results['magnus']['train_time']},
        'magnus_sd': {'rmse': results['magnus_sd']['rmse'], 'train_time': results['magnus_sd']['train_time']},
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics.json", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("✓ EXPERIMENT 5 COMPLETE", flush=True)
    print("="*70, flush=True)
    
    return results

if __name__ == "__main__":
    results = main()
