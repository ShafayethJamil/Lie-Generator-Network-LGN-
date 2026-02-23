"""
HNN and NODE Training - FAIR COMPARISON
========================================
Both trained on vector field (same training signal)
Only difference: HNN has Hamiltonian structure, NODE is black-box
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys

sys.stdout.reconfigure(line_buffering=True)

from torchdiffeq import odeint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================
OMEGA_0 = 1.0
TRAIN_T = 10.0
TEST_T = 100.0
DT = 0.1
N_SEEDS = 3
EPOCHS = 3000
LR = 1e-3  # Same for both

# =============================================================================
# GROUND TRUTH
# =============================================================================
def generate_lc_data(t_span, dt, q0=1.0, p0=0.0):
    """Generate LC circuit ground truth with velocities"""
    t = np.arange(0, t_span, dt)
    q = q0 * np.cos(OMEGA_0 * t)
    p = -q0 * OMEGA_0 * np.sin(OMEGA_0 * t)
    # Velocities (for vector field training)
    dq = p  # dq/dt = p
    dp = -OMEGA_0**2 * q  # dp/dt = -ω²q
    E = 0.5 * (p**2 + OMEGA_0**2 * q**2)
    return t, np.stack([q, p], axis=1), np.stack([dq, dp], axis=1), E

def energy(state):
    q, p = state[..., 0], state[..., 1]
    return 0.5 * (p**2 + OMEGA_0**2 * q**2)

# =============================================================================
# HNN - Hamiltonian Neural Network
# =============================================================================
class HNN(nn.Module):
    """
    Hamiltonian Neural Network
    - Learns H(q,p)
    - Dynamics derived via Hamilton's equations
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def hamiltonian(self, x):
        return self.net(x)
    
    def dynamics(self, x):
        """Compute (dq/dt, dp/dt) from Hamiltonian"""
        x = x.requires_grad_(True)
        H = self.hamiltonian(x)
        dHdx = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dHdq = dHdx[..., 0]
        dHdp = dHdx[..., 1]
        # Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        return torch.stack([dHdp, -dHdq], dim=-1)
    
    def forward(self, t, x0):
        """Rollout using RK4"""
        def dynamics_wrapper(t_scalar, x):
            return self.dynamics(x)
        dt = float(t[1] - t[0])
        trajectory = odeint(dynamics_wrapper, x0, t, method='rk4', options={'step_size': dt})
        return trajectory

# =============================================================================
# NODE - Neural ODE (Black-box)
# =============================================================================
class NeuralODE(nn.Module):
    """
    Neural ODE - learns dynamics directly (no structure)
    Same architecture as HNN but outputs 2D instead of 1D
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # Same as HNN
                nn.init.zeros_(m.bias)
    
    def dynamics(self, x):
        return self.net(x)
    
    def forward(self, t, x0):
        """Rollout using RK4"""
        def dynamics_wrapper(t_scalar, x):
            return self.dynamics(x)
        dt = float(t[1] - t[0])
        trajectory = odeint(dynamics_wrapper, x0, t, method='rk4', options={'step_size': dt})
        return trajectory

# =============================================================================
# TRAINING - Vector Field for BOTH (fair comparison)
# =============================================================================
def train_vectorfield(model, x_train, dx_train, name="Model", epochs=3000, lr=1e-3):
    """
    Train on vector field: Loss = ||f_θ(x) - dx/dt||²
    Same for HNN and NODE - fair comparison
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=1e-6)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Predict dynamics at all training points
        dx_pred = model.dynamics(x_train)
        loss = torch.mean((dx_pred - dx_train)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if (epoch + 1) % 500 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  {name} Ep {epoch+1}/{epochs}: Loss={loss.item():.2e}, lr={current_lr:.1e}", flush=True)
    
    print(f"  {name} Final: Loss={loss.item():.2e}, Best={best_loss:.2e}", flush=True)
    return loss.item()

# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(model, t, x_true, name="Model"):
    model.eval()
    x0 = x_true[0]
    
    with torch.enable_grad():
        x_pred = model(t, x0.requires_grad_(True))
        x_pred = x_pred.detach()
    
    x_pred_np = x_pred.cpu().numpy()
    x_true_np = x_true.cpu().numpy()
    
    rmse = np.sqrt(np.mean((x_pred_np - x_true_np)**2))
    E_pred = energy(x_pred_np)
    E_std = np.std(E_pred)
    E_drift = np.abs(E_pred[-1] - E_pred[0])
    
    print(f"  {name}: RMSE={rmse:.6f}, E_std={E_std:.6f}, E_drift={E_drift:.6f}", flush=True)
    
    return {
        'rmse': rmse,
        'E_std': E_std,
        'E_drift': E_drift,
        'x_pred': x_pred_np,
        'E_pred': E_pred,
    }

# =============================================================================
# MAIN
# =============================================================================
def run_experiment(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}", flush=True)
    print(f"SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Generate data
    t_train_np, x_train_np, dx_train_np, E_train = generate_lc_data(TRAIN_T, DT)
    t_test_np, x_test_np, _, E_test = generate_lc_data(TEST_T, DT)
    
    t_train = torch.tensor(t_train_np, dtype=torch.float64, device=DEVICE)
    x_train = torch.tensor(x_train_np, dtype=torch.float64, device=DEVICE)
    dx_train = torch.tensor(dx_train_np, dtype=torch.float64, device=DEVICE)
    t_test = torch.tensor(t_test_np, dtype=torch.float64, device=DEVICE)
    x_test = torch.tensor(x_test_np, dtype=torch.float64, device=DEVICE)
    
    results = {}
    
    # --- HNN ---
    print("\nTraining HNN (vector field)...", flush=True)
    hnn = HNN(hidden_dim=64).double().to(DEVICE)
    t0 = time.time()
    train_vectorfield(hnn, x_train, dx_train, name="HNN", epochs=EPOCHS, lr=LR)
    results['hnn'] = evaluate_model(hnn, t_test, x_test, "HNN")
    results['hnn']['train_time'] = time.time() - t0
    
    # --- NODE (same training, no structure) ---
    print("\nTraining NODE (vector field)...", flush=True)
    node = NeuralODE(hidden_dim=64).double().to(DEVICE)
    t0 = time.time()
    train_vectorfield(node, x_train, dx_train, name="NODE", epochs=EPOCHS, lr=LR)
    results['node'] = evaluate_model(node, t_test, x_test, "NODE")
    results['node']['train_time'] = time.time() - t0
    
    results['t_test'] = t_test_np
    results['x_test'] = x_test_np
    results['E_test'] = E_test
    
    return results

def main():
    print("="*70, flush=True)
    print("HNN & NODE TRAINING - FAIR COMPARISON", flush=True)
    print("="*70, flush=True)
    print(f"ω₀ = {OMEGA_0}", flush=True)
    print(f"Train: 0-{TRAIN_T}, Test: 0-{TEST_T}, Δt = {DT}", flush=True)
    print(f"Seeds: {N_SEEDS}, Epochs: {EPOCHS}, LR: {LR}", flush=True)
    print(f"Both trained on vector field (same training signal)", flush=True)
    print(f"Difference: HNN has Hamiltonian structure, NODE is black-box", flush=True)
    
    all_results = []
    for seed in range(N_SEEDS):
        results = run_experiment(seed)
        all_results.append(results)
    
    # Aggregate
    print("\n" + "="*70, flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("="*70, flush=True)
    
    metrics = {}
    for method in ['hnn', 'node']:
        rmse_list = [r[method]['rmse'] for r in all_results]
        E_std_list = [r[method]['E_std'] for r in all_results]
        metrics[method] = {
            'rmse_mean': float(np.mean(rmse_list)),
            'rmse_std': float(np.std(rmse_list)),
            'E_std_mean': float(np.mean(E_std_list)),
            'E_std_std': float(np.std(E_std_list)),
        }
        print(f"\n{method.upper()}:", flush=True)
        print(f"  RMSE:     {metrics[method]['rmse_mean']:.6f} ± {metrics[method]['rmse_std']:.6f}", flush=True)
        print(f"  Energy σ: {metrics[method]['E_std_mean']:.6f} ± {metrics[method]['E_std_std']:.6f}", flush=True)
    
    # Save
    output_dir = Path('./lc_results_fair')
    output_dir.mkdir(exist_ok=True)
    
    r = all_results[0]
    t = r['t_test']
    x_true = r['x_test']
    E_true = r['E_test']
    
    # Save ground truth
    np.savetxt(output_dir / 'lc_ground_truth.csv', 
               np.column_stack([t, x_true[:, 0], x_true[:, 1], E_true]),
               delimiter=',', header='t,q,p,E', comments='')
    
    # Save predictions
    for method in ['hnn', 'node']:
        x_pred = r[method]['x_pred']
        E_pred = r[method]['E_pred']
        np.savetxt(output_dir / f'lc_{method}_pred.csv', 
                   np.column_stack([t, x_pred[:, 0], x_pred[:, 1], E_pred]),
                   delimiter=',', header='t,q,p,E', comments='')
    
    # Save metrics
    with open(output_dir / 'lc_hnn_node_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}:", flush=True)
    print(f"  - lc_ground_truth.csv", flush=True)
    print(f"  - lc_hnn_pred.csv", flush=True)
    print(f"  - lc_node_pred.csv", flush=True)
    print(f"  - lc_hnn_node_metrics.json", flush=True)
    
    # Quick plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    ax = axes[0]
    ax.plot(x_true[:, 0], x_true[:, 1], 'k-', lw=2, label='True')
    ax.plot(r['hnn']['x_pred'][:, 0], r['hnn']['x_pred'][:, 1], 'b--', lw=1.5, label='HNN')
    ax.plot(r['node']['x_pred'][:, 0], r['node']['x_pred'][:, 1], 'r:', lw=1.5, label='NODE')
    ax.set_xlabel('q'); ax.set_ylabel('p'); ax.set_title('Phase Portrait')
    ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    
    ax = axes[1]
    ax.axhline(y=0.5, color='k', lw=2, label='True')
    ax.plot(t, r['hnn']['E_pred'], 'b-', lw=1.5, label='HNN')
    ax.plot(t, r['node']['E_pred'], 'r-', lw=1.5, label='NODE')
    ax.axvline(x=TRAIN_T, color='gray', ls='--')
    ax.set_xlabel('Time'); ax.set_ylabel('Energy'); ax.set_title('Energy Conservation')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.5)
    
    ax = axes[2]
    ax.plot(t, x_true[:, 0], 'k-', lw=2, label='True')
    ax.plot(t, r['hnn']['x_pred'][:, 0], 'b--', lw=1.5, label='HNN')
    ax.plot(t, r['node']['x_pred'][:, 0], 'r:', lw=1.5, label='NODE')
    ax.axvline(x=TRAIN_T, color='gray', ls='--')
    ax.set_xlabel('Time'); ax.set_ylabel('q(t)'); ax.set_title('Trajectory')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  - lc_comparison.png", flush=True)
    print("\n" + "="*70, flush=True)
    print("✓ DONE", flush=True)
    print("="*70, flush=True)
    
    return all_results, metrics

if __name__ == "__main__":
    all_results, metrics = main()
