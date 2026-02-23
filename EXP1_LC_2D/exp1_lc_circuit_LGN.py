"""
EXPERIMENT 1: LC Circuit (Conservative/Hamiltonian)
====================================================
System: dq/dt = p, dp/dt = -ω₀²q
Energy: E = ½(p² + ω₀²q²) = constant

Comparisons: Magnus vs HNN vs Neural ODE
Key Claim: Magnus matches HNN on Hamiltonian systems; both preserve energy while NODE drifts.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Install torchdiffeq if needed
try:
    from torchdiffeq import odeint
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'torchdiffeq', '--break-system-packages', '-q'])
    from torchdiffeq import odeint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================
OMEGA_0 = 1.0
TRAIN_T = 10.0      # 10 time units (~10 cycles)
TEST_T = 100.0      # 100 time units (10× extrapolation)
DT = 0.1
N_SEEDS = 3
EPOCHS = 2000

# =============================================================================
# GROUND TRUTH: LC Circuit
# =============================================================================
def generate_lc_data(t_span, dt, q0=1.0, p0=0.0):
    """Generate LC circuit ground truth"""
    t = np.arange(0, t_span, dt)
    q = q0 * np.cos(OMEGA_0 * t)
    p = -q0 * OMEGA_0 * np.sin(OMEGA_0 * t)
    E = 0.5 * (p**2 + OMEGA_0**2 * q**2)
    return t, np.stack([q, p], axis=1), E

def energy(state):
    """Compute energy E = ½(p² + ω²q²)"""
    q, p = state[..., 0], state[..., 1]
    return 0.5 * (p**2 + OMEGA_0**2 * q**2)

# =============================================================================
# MODEL 1: Magnus Neural Network (A ∈ sp(2))
# =============================================================================
class MagnusLC(nn.Module):
    """Magnus for LC circuit - learns ω² parameter"""
    def __init__(self):
        super().__init__()
        self.omega_sq = nn.Parameter(torch.tensor(1.5))
        
    def get_A(self):
        A = torch.zeros(2, 2, device=self.omega_sq.device, dtype=torch.float64)
        A[0, 1] = 1.0
        A[1, 0] = -self.omega_sq
        return A
    
    def forward(self, t, x0):
        A = self.get_A()
        trajectory = [x0]
        for i in range(1, len(t)):
            dt_i = t[i] - t[0]
            expAt = torch.matrix_exp(A * dt_i)
            x_t = expAt @ x0
            trajectory.append(x_t)
        return torch.stack(trajectory)
    
    def get_params(self):
        return {'omega_sq': self.omega_sq.item(), 'omega': np.sqrt(abs(self.omega_sq.item()))}

# =============================================================================
# MODEL 2: Hamiltonian Neural Network (HNN) - FIXED INDEXING
# =============================================================================
class HNN(nn.Module):
    """HNN - learns H(q,p), derives dynamics from gradients"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def hamiltonian(self, x):
        return self.net(x)
    
    def forward(self, t, x0):
        def dynamics(t_scalar, x):
            x = x.requires_grad_(True)
            H = self.hamiltonian(x)
            dHdx = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            # FIXED: Use [..., idx] for proper batched gradient extraction
            dHdq = dHdx[..., 0]
            dHdp = dHdx[..., 1]
            return torch.stack([dHdp, -dHdq], dim=-1)
        
        trajectory = odeint(dynamics, x0, t, method='dopri5')
        return trajectory

# =============================================================================
# MODEL 3: Neural ODE (Generic)
# =============================================================================
class NeuralODE(nn.Module):
    """Generic Neural ODE - no structure preservation"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x0):
        def dynamics(t_scalar, x):
            return self.net(x)
        trajectory = odeint(dynamics, x0, t, method='dopri5')
        return trajectory

# =============================================================================
# TRAINING
# =============================================================================
def train_model(model, t_train, x_train, epochs=2000, lr=1e-3, name="Model"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)
    
    x0 = x_train[0]
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        try:
            x_pred = model(t_train, x0)
            loss = torch.mean((x_pred - x_train)**2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)
            
            losses.append(loss.item())
            
            if (epoch + 1) % 200 == 0:
                print(f"  {name} Ep {epoch+1}/{epochs}: Loss={loss.item():.6f}", flush=True)
                
        except Exception as e:
            print(f"  {name} Ep {epoch+1}: Error - {e}", flush=True)
            losses.append(losses[-1] if losses else 1.0)
    
    return losses

# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(model, t, x_true, name="Model", needs_grad=False):
    model.eval()
    x0 = x_true[0]
    
    if needs_grad:
        with torch.enable_grad():
            x_pred = model(t, x0.requires_grad_(True))
            x_pred = x_pred.detach()
    else:
        with torch.no_grad():
            x_pred = model(t, x0)
    
    x_pred_np = x_pred.cpu().numpy()
    x_true_np = x_true.cpu().numpy()
    
    rmse = np.sqrt(np.mean((x_pred_np - x_true_np)**2))
    E_pred = energy(x_pred_np)
    E_std = np.std(E_pred)
    E_drift = np.abs(E_pred[-1] - E_pred[0])
    
    return {
        'rmse': rmse,
        'E_std': E_std,
        'E_drift': E_drift,
        'x_pred': x_pred_np,
        'E_pred': E_pred,
    }

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_experiment(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}", flush=True)
    print(f"SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Generate data
    t_train_np, x_train_np, E_train = generate_lc_data(TRAIN_T, DT)
    t_test_np, x_test_np, E_test = generate_lc_data(TEST_T, DT)
    
    t_train = torch.tensor(t_train_np, dtype=torch.float64, device=DEVICE)
    x_train = torch.tensor(x_train_np, dtype=torch.float64, device=DEVICE)
    t_test = torch.tensor(t_test_np, dtype=torch.float64, device=DEVICE)
    x_test = torch.tensor(x_test_np, dtype=torch.float64, device=DEVICE)
    
    results = {}
    
    # --- Magnus ---
    print("\nTraining Magnus...", flush=True)
    magnus = MagnusLC().double().to(DEVICE)
    t0 = time.time()
    train_model(magnus, t_train, x_train, epochs=EPOCHS, lr=1e-2, name="Magnus")
    results['magnus'] = evaluate_model(magnus, t_test, x_test, "Magnus")
    results['magnus']['train_time'] = time.time() - t0
    results['magnus']['params'] = magnus.get_params()
    print(f"  Magnus: ω²={magnus.get_params()['omega_sq']:.4f} (true=1.0)", flush=True)
    
    # --- HNN ---
    print("\nTraining HNN...", flush=True)
    hnn = HNN(hidden_dim=64).double().to(DEVICE)
    t0 = time.time()
    train_model(hnn, t_train, x_train, epochs=EPOCHS, lr=1e-3, name="HNN")
    results['hnn'] = evaluate_model(hnn, t_test, x_test, "HNN", needs_grad=True)
    results['hnn']['train_time'] = time.time() - t0
    
    # --- Neural ODE ---
    print("\nTraining Neural ODE...", flush=True)
    node = NeuralODE(hidden_dim=64).double().to(DEVICE)
    t0 = time.time()
    train_model(node, t_train, x_train, epochs=EPOCHS, lr=1e-3, name="NODE")
    results['node'] = evaluate_model(node, t_test, x_test, "NODE")
    results['node']['train_time'] = time.time() - t0
    
    results['t_test'] = t_test_np
    results['x_test'] = x_test_np
    results['E_test'] = E_test
    results['train_horizon'] = TRAIN_T
    
    return results

def main():
    print("="*70, flush=True)
    print("EXPERIMENT 1: LC CIRCUIT (CONSERVATIVE/HAMILTONIAN)", flush=True)
    print("="*70, flush=True)
    print(f"ω₀ = {OMEGA_0}", flush=True)
    print(f"Train: 0-{TRAIN_T}, Test: 0-{TEST_T}, Δt = {DT}", flush=True)
    print(f"Seeds: {N_SEEDS}, Epochs: {EPOCHS}", flush=True)
    
    all_results = []
    for seed in range(N_SEEDS):
        results = run_experiment(seed)
        all_results.append(results)
    
    # Aggregate statistics
    print("\n" + "="*70, flush=True)
    print("AGGREGATE RESULTS (mean ± std over seeds)", flush=True)
    print("="*70, flush=True)
    
    metrics = {}
    for method in ['magnus', 'hnn', 'node']:
        rmse_list = [r[method]['rmse'] for r in all_results]
        E_std_list = [r[method]['E_std'] for r in all_results]
        
        metrics[method] = {
            'rmse_mean': np.mean(rmse_list),
            'rmse_std': np.std(rmse_list),
            'E_std_mean': np.mean(E_std_list),
            'E_std_std': np.std(E_std_list),
        }
        
        print(f"\n{method.upper()}:", flush=True)
        print(f"  RMSE:     {metrics[method]['rmse_mean']:.6f} ± {metrics[method]['rmse_std']:.6f}", flush=True)
        print(f"  Energy σ: {metrics[method]['E_std_mean']:.6f} ± {metrics[method]['E_std_std']:.6f}", flush=True)
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    output_dir = Path('./exp1_results')
    output_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created output directory: {output_dir.absolute()}", flush=True)
    
    # Plot
    print("\nGenerating plots...", flush=True)
    r = all_results[0]
    t = r['t_test']
    x_true = r['x_test']
    E_true = r['E_test']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'magnus': 'green', 'hnn': 'blue', 'node': 'red'}
    labels = {'magnus': 'Magnus', 'hnn': 'HNN', 'node': 'Neural ODE'}
    
    # (a) Phase Space
    ax = axes[0]
    ax.plot(x_true[:, 0], x_true[:, 1], 'k-', lw=2, label='True', alpha=0.7)
    for method in ['magnus', 'hnn', 'node']:
        x_pred = r[method]['x_pred']
        ax.plot(x_pred[:, 0], x_pred[:, 1], '--', color=colors[method], lw=1.5, label=labels[method], alpha=0.8)
    ax.set_xlabel('q'); ax.set_ylabel('p'); ax.set_title('(a) Phase Space')
    ax.legend(fontsize=9); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    
    # (b) Energy vs Time
    ax = axes[1]
    ax.axhline(y=0.5, color='k', linestyle='-', lw=2, label='True (E=0.5)')
    for method in ['magnus', 'hnn', 'node']:
        ax.plot(t, r[method]['E_pred'], '-', color=colors[method], lw=1.5, label=labels[method], alpha=0.8)
    ax.axvline(x=TRAIN_T, color='gray', linestyle='--', lw=1, label='Train/Test')
    ax.set_xlabel('Time'); ax.set_ylabel('Energy'); ax.set_title('(b) Energy')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0.3, 0.7)
    
    # (c) Trajectory
    ax = axes[2]
    ax.plot(t, x_true[:, 0], 'k-', lw=2, label='True', alpha=0.7)
    for method in ['magnus', 'hnn', 'node']:
        ax.plot(t, r[method]['x_pred'][:, 0], '--', color=colors[method], lw=1.5, label=labels[method], alpha=0.8)
    ax.axvline(x=TRAIN_T, color='gray', linestyle='--', lw=1, label='Train/Test')
    ax.set_xlabel('Time'); ax.set_ylabel('q(t)'); ax.set_title('(c) Trajectory')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 50)
    
    plt.suptitle('LC Circuit: Magnus ≈ HNN >> NODE', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'lc_circuit_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'lc_circuit_results.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plots", flush=True)
    
    # Save CSVs
    np.savetxt(output_dir / 'ground_truth.csv', np.column_stack([t, x_true[:, 0], x_true[:, 1], E_true]),
               delimiter=',', header='t,q,p,E', comments='')
    for method in ['magnus', 'hnn', 'node']:
        x_pred = r[method]['x_pred']
        E_pred = r[method]['E_pred']
        np.savetxt(output_dir / f'{method}_pred.csv', np.column_stack([t, x_pred[:, 0], x_pred[:, 1], E_pred]),
                   delimiter=',', header='t,q,p,E', comments='')
    
    with open(output_dir / 'summary_stats.csv', 'w') as f:
        f.write('method,rmse_mean,rmse_std,E_std_mean,E_std_std\n')
        for method in ['magnus', 'hnn', 'node']:
            m = metrics[method]
            f.write(f"{method},{m['rmse_mean']:.8f},{m['rmse_std']:.8f},{m['E_std_mean']:.8f},{m['E_std_std']:.8f}\n")
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✓ Saved CSVs: ground_truth.csv, magnus_pred.csv, hnn_pred.csv, node_pred.csv", flush=True)
    print(f"  ✓ Saved summary_stats.csv, metrics.json", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("✓ EXPERIMENT 1 COMPLETE", flush=True)
    print("="*70, flush=True)
    
    return all_results, metrics

if __name__ == "__main__":
    all_results, metrics = main()
