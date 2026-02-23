"""
EXPERIMENT: RLC-3 (6D) - Noise Robustness & Δt Generalization
==============================================================
Key Claims:
1. Structure (S-D) provides regularization under noise
2. exp(A·Δt) generalizes to any Δt - discrete methods fail

UPDATED:
- Saves eigenvalues, trajectories, error over time
- Faster NODE training via segment-based backprop
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

try:
    from torchdiffeq import odeint
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'torchdiffeq', '--break-system-packages', '-q'])
    from torchdiffeq import odeint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

# =============================================================================
# SYSTEM PARAMETERS - 6D RLC-3
# =============================================================================
N_SECTIONS = 3
STATE_DIM = 2 * N_SECTIONS  # 6D
L_VALUES = [1.0] * N_SECTIONS
C_VALUES = [1.0] * N_SECTIONS
R_VALUES = [0.1] * N_SECTIONS

TRAIN_T = 30.0
TEST_T = 100.0
DT_TRAIN = 0.1
N_SEEDS = 5
EPOCHS = 1000
N_INIT_CONDITIONS = 5

# NODE speedup: train on short segments
NODE_SEGMENT_LEN = 20  # steps per segment (vs 300 for full trajectory)

# Experiment parameters
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
DT_RATIOS = [1.0, 1.5, 2.0, 5.0, 10.0]

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

A_TRUE = build_rlc_ladder_A(L_VALUES, C_VALUES, R_VALUES)
TRUE_EIGVALS = np.linalg.eigvals(A_TRUE)

# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_data(t_span, dt, x0=None, A=None, noise_sigma=0.0):
    if A is None:
        A = A_TRUE
    if x0 is None:
        x0 = np.zeros(STATE_DIM)
        x0[0], x0[1] = 1.0, 0.5
    
    def dynamics(t, x):
        return A @ x
    
    t_eval = np.arange(0, t_span, dt)
    sol = solve_ivp(dynamics, [0, t_span], x0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    x_clean = sol.y.T
    
    if noise_sigma > 0:
        signal_std = np.std(x_clean)
        noise = np.random.randn(*x_clean.shape) * noise_sigma * signal_std
        x_noisy = x_clean + noise
    else:
        x_noisy = x_clean
    
    return sol.t, x_clean, x_noisy, A

def generate_multiple_trajectories(t_span, dt, n_traj, noise_sigma=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    t_eval = np.arange(0, t_span, dt)
    
    all_x0, all_clean, all_noisy = [], [], []
    for _ in range(n_traj):
        x0 = np.random.randn(STATE_DIM) * 0.5
        E0 = compute_energy(x0, L_VALUES, C_VALUES)
        x0 = x0 / np.sqrt(E0 + 1e-6)
        
        _, x_clean, x_noisy, _ = generate_data(t_span, dt, x0, A_TRUE, noise_sigma)
        all_x0.append(x0)
        all_clean.append(x_clean)
        all_noisy.append(x_noisy)
    
    return t_eval, all_x0, all_clean, all_noisy

# =============================================================================
# MODELS
# =============================================================================
class LieDissipative(nn.Module):
    """Lie with A = S - D structure (guaranteed stable)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        n_upper = dim * (dim - 1) // 2
        self.S_upper = nn.Parameter(torch.randn(n_upper, dtype=torch.float64) * 0.3)
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
    
    def rollout_numpy(self, x0, t, dt=None):
        A = self.get_A().detach().cpu().numpy()
        if dt is None:
            dt = t[1] - t[0]
        expAdt = expm(A * dt)
        trajectory = [x0]
        x = x0.copy()
        for _ in range(1, len(t)):
            x = expAdt @ x
            trajectory.append(x)
        return np.array(trajectory)
    
    def get_A_numpy(self):
        return self.get_A().detach().cpu().numpy()
    
    def get_eigenvalues(self):
        return np.linalg.eigvals(self.get_A_numpy())

class LinearID:
    """Classical Linear System Identification"""
    def __init__(self, dim):
        self.dim = dim
        self.A = None
    
    def fit(self, t, trajectories):
        dt = t[1] - t[0]
        X_all, dX_all = [], []
        for traj in trajectories:
            dX = np.gradient(traj, dt, axis=0)
            X_all.append(traj)
            dX_all.append(dX)
        X = np.vstack(X_all)
        dX = np.vstack(dX_all)
        A_T, _, _, _ = np.linalg.lstsq(X, dX, rcond=None)
        self.A = A_T.T
        return self.A
    
    def rollout(self, x0, t, dt=None):
        if dt is None:
            dt = t[1] - t[0]
        expAdt = expm(self.A * dt)
        trajectory = [x0]
        x = x0.copy()
        for _ in range(1, len(t)):
            x = expAdt @ x
            trajectory.append(x)
        return np.array(trajectory)
    
    def get_eigenvalues(self):
        return np.linalg.eigvals(self.A)

class DiscreteNN(nn.Module):
    """Discrete-time NN: x_{k+1} = x_k + f(x_k)"""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward_step(self, x):
        return x + self.net(x)
    
    def rollout(self, x0, n_steps):
        trajectory = [x0]
        x = x0
        for _ in range(n_steps - 1):
            x = self.forward_step(x)
            trajectory.append(x)
        return torch.stack(trajectory)

class NeuralODE(nn.Module):
    """Continuous-time Neural ODE"""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def dynamics(self, t, x):
        return self.net(x)
    
    def forward(self, t, x0):
        dt = float(t[1] - t[0])
        return odeint(self.dynamics, x0, t, method='rk4', options={'step_size': dt})

# =============================================================================
# TRAINING
# =============================================================================
def train_lie(model, t, x0_list, x_traj_list, epochs=1000, lr=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5, min_lr=1e-5)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        for x0, x_true in zip(x0_list, x_traj_list):
            x_pred = model(t, x0)
            total_loss += torch.mean((x_pred - x_true)**2)
        total_loss /= len(x0_list)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        if (epoch + 1) % 200 == 0:
            print(f"    Epoch {epoch+1}: Loss = {total_loss.item():.2e}", flush=True)
    
    return total_loss.item()

def train_discrete(model, x_traj_list, epochs=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        count = 0
        for x_traj in x_traj_list:
            for i in range(len(x_traj) - 1):
                x_next_pred = model.forward_step(x_traj[i])
                total_loss += torch.mean((x_next_pred - x_traj[i+1])**2)
                count += 1
        total_loss /= count
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"    Epoch {epoch+1}: Loss = {total_loss.item():.2e}", flush=True)
    
    return total_loss.item()

def train_node_fast(model, t, x0_list, x_traj_list, epochs=1000, lr=1e-3, segment_len=20):
    """
    FASTER NODE training: backprop through short segments only.
    Same data, ~10x speedup.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dt = t[1] - t[0]
    
    # Pre-extract segments from trajectories
    segments = []
    for x_traj in x_traj_list:
        n_segments = len(x_traj) // segment_len
        for i in range(n_segments):
            start = i * segment_len
            end = start + segment_len
            seg_x0 = x_traj[start]
            seg_target = x_traj[start:end]
            seg_t = t[:segment_len]
            segments.append((seg_x0, seg_target, seg_t))
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Sample random segments
        batch_idx = np.random.choice(len(segments), min(5, len(segments)), replace=False)
        
        total_loss = 0.0
        for idx in batch_idx:
            seg_x0, seg_target, seg_t = segments[idx]
            seg_pred = model(seg_t, seg_x0)
            total_loss += torch.mean((seg_pred - seg_target)**2)
        total_loss /= len(batch_idx)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"    Epoch {epoch+1}: Loss = {total_loss.item():.2e}", flush=True)
    
    return total_loss.item()

# =============================================================================
# EVALUATION WITH FULL DATA SAVING
# =============================================================================
def evaluate_with_full_data(model_or_obj, x0, t_test, x_test_clean, method_name, dt=None):
    """Evaluate and return full data for plotting"""
    
    if method_name == 'lie':
        x_pred = model_or_obj.rollout_numpy(x0, t_test, dt=dt)
        eigvals = model_or_obj.get_eigenvalues()
        A_learned = model_or_obj.get_A_numpy()
    elif method_name == 'linear':
        x_pred = model_or_obj.rollout(x0, t_test, dt=dt)
        eigvals = model_or_obj.get_eigenvalues()
        A_learned = model_or_obj.A
    elif method_name == 'discrete':
        model_or_obj.eval()
        with torch.no_grad():
            x0_torch = torch.tensor(x0, dtype=torch.float64, device=DEVICE)
            x_pred = model_or_obj.rollout(x0_torch, len(t_test)).cpu().numpy()
        eigvals = None
        A_learned = None
    elif method_name == 'node':
        model_or_obj.eval()
        with torch.no_grad():
            t_torch = torch.tensor(t_test, dtype=torch.float64, device=DEVICE)
            x0_torch = torch.tensor(x0, dtype=torch.float64, device=DEVICE)
            x_pred = model_or_obj(t_torch, x0_torch).cpu().numpy()
        eigvals = None
        A_learned = None
    
    # Compute metrics
    error_over_time = np.sqrt(np.sum((x_pred - x_test_clean)**2, axis=1))
    rmse = np.sqrt(np.mean((x_pred - x_test_clean)**2))
    energy_pred = compute_energy(x_pred, L_VALUES, C_VALUES)
    energy_true = compute_energy(x_test_clean, L_VALUES, C_VALUES)
    
    # Stability check
    if eigvals is not None:
        unstable_count = np.sum(np.real(eigvals) > 0)
        max_real_eigval = np.max(np.real(eigvals))
    else:
        unstable_count = None
        max_real_eigval = None
    
    return {
        'x_pred': x_pred,
        'error_over_time': error_over_time,
        'rmse': rmse,
        'energy_pred': energy_pred,
        'energy_true': energy_true,
        'eigvals': eigvals,
        'A_learned': A_learned,
        'unstable_count': unstable_count,
        'max_real_eigval': max_real_eigval,
    }

# =============================================================================
# EXPERIMENT 1: NOISE ROBUSTNESS
# =============================================================================
def run_noise_experiment(seed):
    print(f"\n{'='*60}", flush=True)
    print(f"NOISE ROBUSTNESS EXPERIMENT (Seed {seed})", flush=True)
    print(f"{'='*60}", flush=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    results = {sigma: {} for sigma in NOISE_LEVELS}
    
    # Fixed test trajectory (clean)
    x0_test = np.zeros(STATE_DIM)
    x0_test[0], x0_test[1] = 1.0, 0.5
    t_test, x_test_clean, _, _ = generate_data(TEST_T, DT_TRAIN, x0=x0_test)
    
    for sigma in NOISE_LEVELS:
        print(f"\n--- Noise σ = {sigma} ---", flush=True)
        
        # Generate training data with noise
        t_train, x0_list, x_clean_list, x_noisy_list = generate_multiple_trajectories(
            TRAIN_T, DT_TRAIN, N_INIT_CONDITIONS, noise_sigma=sigma, seed=seed
        )
        
        # Convert to torch
        t_train_torch = torch.tensor(t_train, dtype=torch.float64, device=DEVICE)
        x0_list_torch = [torch.tensor(x0, dtype=torch.float64, device=DEVICE) for x0 in x0_list]
        x_noisy_list_torch = [torch.tensor(x, dtype=torch.float64, device=DEVICE) for x in x_noisy_list]
        
        # --- Lie (S-D) ---
        print("  Training Lie (S-D)...", flush=True)
        lie = LieDissipative(STATE_DIM).double().to(DEVICE)
        train_lie(lie, t_train_torch, x0_list_torch, x_noisy_list_torch, epochs=EPOCHS, lr=1e-2)
        results[sigma]['lie'] = evaluate_with_full_data(lie, x0_test, t_test, x_test_clean, 'lie')
        
        # --- Linear-ID ---
        print("  Fitting Linear-ID...", flush=True)
        linear = LinearID(STATE_DIM)
        linear.fit(t_train, x_noisy_list)
        results[sigma]['linear'] = evaluate_with_full_data(linear, x0_test, t_test, x_test_clean, 'linear')
        
        # --- NODE (fast training) ---
        print("  Training NODE...", flush=True)
        node = NeuralODE(STATE_DIM, hidden_dim=64).double().to(DEVICE)
        train_node_fast(node, t_train_torch, x0_list_torch, x_noisy_list_torch, 
                       epochs=EPOCHS, lr=1e-3, segment_len=NODE_SEGMENT_LEN)
        results[sigma]['node'] = evaluate_with_full_data(node, x0_test, t_test, x_test_clean, 'node')
        
        # Print summary
        print(f"  Lie:    RMSE={results[sigma]['lie']['rmse']:.4f}, Unstable={results[sigma]['lie']['unstable_count']}", flush=True)
        print(f"  Linear: RMSE={results[sigma]['linear']['rmse']:.4f}, Unstable={results[sigma]['linear']['unstable_count']}", flush=True)
        print(f"  NODE:   RMSE={results[sigma]['node']['rmse']:.4f}", flush=True)
    
    # Add test data for plotting
    results['t_test'] = t_test
    results['x_test_clean'] = x_test_clean
    
    return results

# =============================================================================
# EXPERIMENT 2: Δt GENERALIZATION
# =============================================================================
def run_dt_experiment(seed):
    print(f"\n{'='*60}", flush=True)
    print(f"Δt GENERALIZATION EXPERIMENT (Seed {seed})", flush=True)
    print(f"{'='*60}", flush=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate clean training data at DT_TRAIN
    t_train, x0_list, x_clean_list, _ = generate_multiple_trajectories(
        TRAIN_T, DT_TRAIN, N_INIT_CONDITIONS, noise_sigma=0.0, seed=seed
    )
    
    # Convert to torch
    t_train_torch = torch.tensor(t_train, dtype=torch.float64, device=DEVICE)
    x0_list_torch = [torch.tensor(x0, dtype=torch.float64, device=DEVICE) for x0 in x0_list]
    x_clean_list_torch = [torch.tensor(x, dtype=torch.float64, device=DEVICE) for x in x_clean_list]
    
    # --- Train all methods at DT_TRAIN ---
    print("\nTraining Lie (S-D)...", flush=True)
    lie = LieDissipative(STATE_DIM).double().to(DEVICE)
    train_lie(lie, t_train_torch, x0_list_torch, x_clean_list_torch, epochs=EPOCHS, lr=1e-2)
    
    print("\nFitting Linear-ID...", flush=True)
    linear = LinearID(STATE_DIM)
    linear.fit(t_train, x_clean_list)
    
    print("\nTraining Discrete-NN...", flush=True)
    discrete = DiscreteNN(STATE_DIM, hidden_dim=64).double().to(DEVICE)
    train_discrete(discrete, x_clean_list_torch, epochs=EPOCHS, lr=1e-3)
    
    print("\nTraining NODE...", flush=True)
    node = NeuralODE(STATE_DIM, hidden_dim=64).double().to(DEVICE)
    train_node_fast(node, t_train_torch, x0_list_torch, x_clean_list_torch, 
                   epochs=EPOCHS, lr=1e-3, segment_len=NODE_SEGMENT_LEN)
    
    # --- Evaluate at different Δt ---
    results = {ratio: {} for ratio in DT_RATIOS}
    
    x0_test = np.zeros(STATE_DIM)
    x0_test[0], x0_test[1] = 1.0, 0.5
    
    for ratio in DT_RATIOS:
        dt_test = DT_TRAIN * ratio
        print(f"\n--- Δt ratio = {ratio} (Δt_test = {dt_test:.3f}) ---", flush=True)
        
        # Ground truth at this dt
        t_test, x_test_clean, _, _ = generate_data(TEST_T, dt_test, x0=x0_test)
        
        # Evaluate each method
        results[ratio]['lie'] = evaluate_with_full_data(lie, x0_test, t_test, x_test_clean, 'lie', dt=dt_test)
        results[ratio]['linear'] = evaluate_with_full_data(linear, x0_test, t_test, x_test_clean, 'linear', dt=dt_test)
        results[ratio]['discrete'] = evaluate_with_full_data(discrete, x0_test, t_test, x_test_clean, 'discrete')
        results[ratio]['node'] = evaluate_with_full_data(node, x0_test, t_test, x_test_clean, 'node')
        
        # Store test data
        results[ratio]['t_test'] = t_test
        results[ratio]['x_test_clean'] = x_test_clean
        
        print(f"  Lie:      RMSE={results[ratio]['lie']['rmse']:.4f}", flush=True)
        print(f"  Linear:   RMSE={results[ratio]['linear']['rmse']:.4f}", flush=True)
        print(f"  Discrete: RMSE={results[ratio]['discrete']['rmse']:.4f}", flush=True)
        print(f"  NODE:     RMSE={results[ratio]['node']['rmse']:.4f}", flush=True)
    
    return results

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70, flush=True)
    print(f"RLC-3 (6D): NOISE & Δt GENERALIZATION EXPERIMENTS", flush=True)
    print("="*70, flush=True)
    print(f"System: {N_SECTIONS}-section RLC ladder ({STATE_DIM} states)", flush=True)
    print(f"Train: 0-{TRAIN_T}, Test: 0-{TEST_T}, Δt_train={DT_TRAIN}", flush=True)
    print(f"Seeds: {N_SEEDS}, Epochs: {EPOCHS}", flush=True)
    print(f"NODE segment length: {NODE_SEGMENT_LEN} (speedup)", flush=True)
    print(f"Noise levels: {NOISE_LEVELS}", flush=True)
    print(f"Δt ratios: {DT_RATIOS}", flush=True)
    
    output_dir = Path('./exp_robustness_results')
    output_dir.mkdir(exist_ok=True)
    
    # ==========================================================================
    # RUN EXPERIMENTS
    # ==========================================================================
    all_noise_results = []
    all_dt_results = []
    
    for seed in range(N_SEEDS):
        noise_res = run_noise_experiment(seed)
        all_noise_results.append(noise_res)
        
        dt_res = run_dt_experiment(seed)
        all_dt_results.append(dt_res)
    
    # ==========================================================================
    # AGGREGATE RESULTS
    # ==========================================================================
    print("\n" + "="*70, flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("="*70, flush=True)
    
    # --- Noise Summary ---
    print("\n--- NOISE ROBUSTNESS ---")
    noise_summary = {}
    for sigma in NOISE_LEVELS:
        noise_summary[sigma] = {}
        for method in ['lie', 'linear', 'node']:
            rmses = [r[sigma][method]['rmse'] for r in all_noise_results]
            noise_summary[sigma][method] = {
                'rmse_mean': np.mean(rmses), 
                'rmse_std': np.std(rmses)
            }
        
        # Unstable counts
        for method in ['lie', 'linear']:
            unstables = [r[sigma][method]['unstable_count'] for r in all_noise_results]
            noise_summary[sigma][method]['unstable_mean'] = np.mean(unstables)
            noise_summary[sigma][method]['unstable_std'] = np.std(unstables)
        
        print(f"σ={sigma}: Lie={noise_summary[sigma]['lie']['rmse_mean']:.4f}±{noise_summary[sigma]['lie']['rmse_std']:.4f} (unstab={noise_summary[sigma]['lie']['unstable_mean']:.1f}), "
              f"Linear={noise_summary[sigma]['linear']['rmse_mean']:.4f} (unstab={noise_summary[sigma]['linear']['unstable_mean']:.1f}), "
              f"NODE={noise_summary[sigma]['node']['rmse_mean']:.4f}")
    
    # --- Δt Summary ---
    print("\n--- Δt GENERALIZATION ---")
    dt_summary = {}
    for ratio in DT_RATIOS:
        dt_summary[ratio] = {}
        for method in ['lie', 'linear', 'discrete', 'node']:
            rmses = [r[ratio][method]['rmse'] for r in all_dt_results]
            dt_summary[ratio][method] = {
                'rmse_mean': np.mean(rmses),
                'rmse_std': np.std(rmses)
            }
        
        print(f"Δt×{ratio}: Lie={dt_summary[ratio]['lie']['rmse_mean']:.4f}, "
              f"Linear={dt_summary[ratio]['linear']['rmse_mean']:.4f}, "
              f"Discrete={dt_summary[ratio]['discrete']['rmse_mean']:.4f}, "
              f"NODE={dt_summary[ratio]['node']['rmse_mean']:.4f}")
    
    # ==========================================================================
    # SAVE DETAILED RESULTS
    # ==========================================================================
    
    # --- Save representative trajectories and eigenvalues (from seed 0) ---
    r_noise = all_noise_results[0]
    r_dt = all_dt_results[0]
    
    # Noise experiment: save for each sigma
    for sigma in NOISE_LEVELS:
        sigma_str = f"{sigma:.2f}".replace('.', 'p')
        
        # Eigenvalues
        for method in ['lie', 'linear']:
            eigvals = r_noise[sigma][method]['eigvals']
            if eigvals is not None:
                np.savetxt(output_dir / f'noise_{sigma_str}_{method}_eigvals.csv',
                          np.column_stack([np.real(eigvals), np.imag(eigvals)]),
                          delimiter=',', header='real,imag', comments='')
        
        # Error over time
        np.savetxt(output_dir / f'noise_{sigma_str}_error_time.csv',
                  np.column_stack([
                      r_noise['t_test'],
                      r_noise[sigma]['lie']['error_over_time'],
                      r_noise[sigma]['linear']['error_over_time'],
                      r_noise[sigma]['node']['error_over_time'],
                  ]),
                  delimiter=',', header='t,lie,linear,node', comments='')
        
        # Trajectory (first state)
        np.savetxt(output_dir / f'noise_{sigma_str}_trajectory.csv',
                  np.column_stack([
                      r_noise['t_test'],
                      r_noise['x_test_clean'][:, 0],
                      r_noise[sigma]['lie']['x_pred'][:, 0],
                      r_noise[sigma]['linear']['x_pred'][:, 0],
                      r_noise[sigma]['node']['x_pred'][:, 0],
                  ]),
                  delimiter=',', header='t,true,lie,linear,node', comments='')
    
    # Δt experiment: save for each ratio
    for ratio in DT_RATIOS:
        ratio_str = f"{ratio:.1f}".replace('.', 'p')
        
        np.savetxt(output_dir / f'dt_{ratio_str}_error_time.csv',
                  np.column_stack([
                      r_dt[ratio]['t_test'],
                      r_dt[ratio]['lie']['error_over_time'],
                      r_dt[ratio]['linear']['error_over_time'],
                      r_dt[ratio]['discrete']['error_over_time'],
                      r_dt[ratio]['node']['error_over_time'],
                  ]),
                  delimiter=',', header='t,lie,linear,discrete,node', comments='')
    
    # --- Summary CSVs for MATLAB plotting ---
    with open(output_dir / 'noise_summary.csv', 'w') as f:
        f.write('sigma,method,rmse_mean,rmse_std,unstable_mean,unstable_std\n')
        for sigma in NOISE_LEVELS:
            for method in ['lie', 'linear', 'node']:
                m = noise_summary[sigma][method]
                unstable_mean = m.get('unstable_mean', 0)
                unstable_std = m.get('unstable_std', 0)
                f.write(f"{sigma},{method},{m['rmse_mean']:.6f},{m['rmse_std']:.6f},{unstable_mean:.2f},{unstable_std:.2f}\n")
    
    with open(output_dir / 'dt_summary.csv', 'w') as f:
        f.write('dt_ratio,method,rmse_mean,rmse_std\n')
        for ratio in DT_RATIOS:
            for method in ['lie', 'linear', 'discrete', 'node']:
                m = dt_summary[ratio][method]
                f.write(f"{ratio},{method},{m['rmse_mean']:.6f},{m['rmse_std']:.6f}\n")
    
    # Full JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump({
            'noise_summary': convert_to_serializable(noise_summary),
            'dt_summary': convert_to_serializable(dt_summary),
            'true_eigenvalues': TRUE_EIGVALS.tolist(),
        }, f, indent=2)
    
    print(f"\n✓ Saved results to {output_dir}", flush=True)
    
    # ==========================================================================
    # QUICK PLOTS
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'lie': '#1f77b4', 'linear': '#d62728', 'discrete': '#2ca02c', 'node': '#ff7f0e'}
    
    # --- Plot 1: Noise Robustness ---
    ax = axes[0]
    for method in ['lie', 'linear', 'node']:
        means = [noise_summary[s][method]['rmse_mean'] for s in NOISE_LEVELS]
        stds = [noise_summary[s][method]['rmse_std'] for s in NOISE_LEVELS]
        label = {'lie': 'Lie (S-D)', 'linear': 'Linear-ID', 'node': 'NODE'}[method]
        ax.errorbar(NOISE_LEVELS, means, yerr=stds, marker='o', label=label, 
                    color=colors[method], linewidth=2, capsize=3)
    ax.set_xlabel('Noise σ (relative)', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('(a) Noise Robustness', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: Δt Generalization ---
    ax = axes[1]
    for method in ['lie', 'linear', 'discrete', 'node']:
        means = [dt_summary[r][method]['rmse_mean'] for r in DT_RATIOS]
        stds = [dt_summary[r][method]['rmse_std'] for r in DT_RATIOS]
        label = {'lie': 'Lie (S-D)', 'linear': 'Linear-ID', 'discrete': 'Discrete-NN', 'node': 'NODE'}[method]
        ax.errorbar(DT_RATIOS, means, yerr=stds, marker='o', label=label, 
                    color=colors[method], linewidth=2, capsize=3)
    ax.set_xlabel('Δt_test / Δt_train', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('(b) Δt Generalization', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'robustness_summary.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plots", flush=True)
    print("\n" + "="*70, flush=True)
    print("✓ ALL EXPERIMENTS COMPLETE", flush=True)
    print("="*70, flush=True)
    
    return all_noise_results, all_dt_results, noise_summary, dt_summary

if __name__ == "__main__":
    all_noise, all_dt, noise_summary, dt_summary = main()
