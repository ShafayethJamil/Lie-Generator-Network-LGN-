"""
EXPERIMENT 2: RLC Circuit (Dissipative)
=======================================
System: dq/dt = p, dp/dt = -ω₀²q - γp
Energy: E = ½(p² + ω₀²q²), dE/dt = -γp² ≤ 0 (monotonically decreasing)

Comparisons: LGN vs HNN vs Neural ODE
Key Claim: HNN CANNOT represent dissipation by design. Magnus handles BOTH conservative AND dissipative.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import sys
from scipy.integrate import solve_ivp

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
# SYSTEM PARAMETERS
# =============================================================================
OMEGA_0 = 1.0
GAMMA = 0.1           # Damping coefficient
TRAIN_T = 20.0
TEST_T = 100.0
DT = 0.1
N_SEEDS = 3
EPOCHS = 2000

# =============================================================================
# GROUND TRUTH
# =============================================================================
def generate_rlc_data(t_span, dt, q0=1.0, p0=0.0):
    def dynamics(t, x):
        q, p = x
        return [p, -OMEGA_0**2 * q - GAMMA * p]
    
    t_eval = np.arange(0, t_span, dt)
    sol = solve_ivp(dynamics, [0, t_span], [q0, p0], t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    t = sol.t
    x = sol.y.T
    E = 0.5 * (x[:, 1]**2 + OMEGA_0**2 * x[:, 0]**2)
    return t, x, E

def energy(state):
    q, p = state[..., 0], state[..., 1]
    return 0.5 * (p**2 + OMEGA_0**2 * q**2)

# =============================================================================
# MODEL 1: Magnus Neural Network (Dissipative)
# =============================================================================
class MagnusRLC(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega_sq = nn.Parameter(torch.tensor(1.2))
        self.gamma = nn.Parameter(torch.tensor(0.15))
        
    def get_A(self):
        A = torch.zeros(2, 2, device=self.omega_sq.device, dtype=torch.float64)
        A[0, 1] = 1.0
        A[1, 0] = -self.omega_sq
        A[1, 1] = -self.gamma
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
        return {'omega_sq': self.omega_sq.item(), 'gamma': self.gamma.item()}

# =============================================================================
# MODEL 2: HNN - WILL FAIL (can't do dissipation)
# =============================================================================
class HNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
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
            dHdq = dHdx[..., 0]
            dHdp = dHdx[..., 1]
            return torch.stack([dHdp, -dHdq], dim=-1)
        return odeint(dynamics, x0, t, method='dopri5')

# =============================================================================
# MODEL 3: Neural ODE
# =============================================================================
class NeuralODE(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x0):
        def dynamics(t_scalar, x):
            return self.net(x)
        return odeint(dynamics, x0, t, method='dopri5')

# =============================================================================
# TRAINING & EVALUATION
# =============================================================================
def train_model(model, t_train, x_train, epochs=2000, lr=1e-3, name="Model"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)
    x0 = x_train[0]
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        try:
            x_pred = model(t_train, x0)
            loss = torch.mean((x_pred - x_train)**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)
            
            if (epoch + 1) % 200 == 0:
                print(f"  {name} Ep {epoch+1}/{epochs}: Loss={loss.item():.6f}", flush=True)
        except Exception as e:
            print(f"  {name} Ep {epoch+1}: Error - {e}", flush=True)

def evaluate_model(model, t, x_true, E_true, name="Model", needs_grad=False):
    model.eval()
    x0 = x_true[0]
    
    if needs_grad:
        with torch.enable_grad():
            x_pred = model(t, x0.requires_grad_(True)).detach()
    else:
        with torch.no_grad():
            x_pred = model(t, x0)
    
    x_pred_np = x_pred.cpu().numpy()
    x_true_np = x_true.cpu().numpy()
    
    rmse = np.sqrt(np.mean((x_pred_np - x_true_np)**2))
    E_pred = energy(x_pred_np)
    
    dE = np.diff(E_pred)
    violations = np.sum(dE > 1e-6)
    violation_frac = violations / len(dE)
    
    return {
        'rmse': rmse, 'E_pred': E_pred, 'x_pred': x_pred_np,
        'violations': violations, 'violation_frac': violation_frac,
        'E_max': np.max(E_pred), 'E_0': E_pred[0]
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
    
    t_train_np, x_train_np, E_train = generate_rlc_data(TRAIN_T, DT)
    t_test_np, x_test_np, E_test = generate_rlc_data(TEST_T, DT)
    
    t_train = torch.tensor(t_train_np, dtype=torch.float64, device=DEVICE)
    x_train = torch.tensor(x_train_np, dtype=torch.float64, device=DEVICE)
    t_test = torch.tensor(t_test_np, dtype=torch.float64, device=DEVICE)
    x_test = torch.tensor(x_test_np, dtype=torch.float64, device=DEVICE)
    
    results = {}
    
    print("\nTraining Magnus...", flush=True)
    magnus = MagnusRLC().double().to(DEVICE)
    t0 = time.time()
    train_model(magnus, t_train, x_train, epochs=EPOCHS, lr=1e-2, name="Magnus")
    results['magnus'] = evaluate_model(magnus, t_test, x_test, E_test, "Magnus")
    results['magnus']['train_time'] = time.time() - t0
    results['magnus']['params'] = magnus.get_params()
    print(f"  Magnus: ω²={magnus.get_params()['omega_sq']:.4f}, γ={magnus.get_params()['gamma']:.4f}", flush=True)
    
    print("\nTraining HNN (expected to fail on dissipation)...", flush=True)
    hnn = HNN(hidden_dim=64).double().to(DEVICE)
    t0 = time.time()
    train_model(hnn, t_train, x_train, epochs=EPOCHS, lr=1e-3, name="HNN")
    results['hnn'] = evaluate_model(hnn, t_test, x_test, E_test, "HNN", needs_grad=True)
    results['hnn']['train_time'] = time.time() - t0
    
    print("\nTraining Neural ODE...", flush=True)
    node = NeuralODE(hidden_dim=64).double().to(DEVICE)
    t0 = time.time()
    train_model(node, t_train, x_train, epochs=EPOCHS, lr=1e-3, name="NODE")
    results['node'] = evaluate_model(node, t_test, x_test, E_test, "NODE")
    results['node']['train_time'] = time.time() - t0
    
    results['t_test'] = t_test_np
    results['x_test'] = x_test_np
    results['E_test'] = E_test
    
    return results

def main():
    print("="*70, flush=True)
    print("EXPERIMENT 2: RLC CIRCUIT (DISSIPATIVE)", flush=True)
    print("="*70, flush=True)
    print(f"ω₀ = {OMEGA_0}, γ = {GAMMA}", flush=True)
    print(f"Train: 0-{TRAIN_T}, Test: 0-{TEST_T}, Δt = {DT}", flush=True)
    print(f"Seeds: {N_SEEDS}, Epochs: {EPOCHS}", flush=True)
    
    all_results = []
    for seed in range(N_SEEDS):
        all_results.append(run_experiment(seed))
    
    print("\n" + "="*70, flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("="*70, flush=True)
    
    metrics = {}
    for method in ['magnus', 'hnn', 'node']:
        rmse_list = [r[method]['rmse'] for r in all_results]
        viol_list = [r[method]['violation_frac'] for r in all_results]
        metrics[method] = {
            'rmse_mean': np.mean(rmse_list), 'rmse_std': np.std(rmse_list),
            'violation_frac_mean': np.mean(viol_list), 'violation_frac_std': np.std(viol_list),
        }
        print(f"\n{method.upper()}:", flush=True)
        print(f"  RMSE:        {metrics[method]['rmse_mean']:.6f} ± {metrics[method]['rmse_std']:.6f}", flush=True)
        print(f"  E-violation: {metrics[method]['violation_frac_mean']*100:.2f}%", flush=True)
    
    # Save
    output_dir = Path('./exp2_results')
    output_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created output directory: {output_dir.absolute()}", flush=True)
    
    r = all_results[0]
    t = r['t_test']
    x_true = r['x_test']
    E_true = r['E_test']
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'magnus': 'green', 'hnn': 'blue', 'node': 'red'}
    labels = {'magnus': 'Magnus', 'hnn': 'HNN (fails)', 'node': 'Neural ODE'}
    
    ax = axes[0]
    ax.plot(x_true[:, 0], x_true[:, 1], 'k-', lw=2, label='True')
    for method in ['magnus', 'hnn', 'node']:
        ax.plot(r[method]['x_pred'][:, 0], r[method]['x_pred'][:, 1], '--', color=colors[method], lw=1.5, label=labels[method])
    ax.set_xlabel('q'); ax.set_ylabel('p'); ax.set_title('(a) Phase Space')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(t, E_true, 'k-', lw=2, label='True (decays)')
    for method in ['magnus', 'hnn', 'node']:
        ax.plot(t, r[method]['E_pred'], '-', color=colors[method], lw=1.5, label=labels[method])
    ax.axvline(x=TRAIN_T, color='gray', linestyle='--', lw=1)
    ax.set_xlabel('Time'); ax.set_ylabel('Energy'); ax.set_title('(b) Energy (should decay)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(t, x_true[:, 0], 'k-', lw=2, label='True')
    for method in ['magnus', 'hnn', 'node']:
        ax.plot(t, r[method]['x_pred'][:, 0], '--', color=colors[method], lw=1.5, label=labels[method])
    ax.axvline(x=TRAIN_T, color='gray', linestyle='--', lw=1)
    ax.set_xlabel('Time'); ax.set_ylabel('q(t)'); ax.set_title('(c) Trajectory')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 60)
    
    plt.suptitle('RLC Circuit: HNN FAILS (cannot dissipate), Magnus Works', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rlc_circuit_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'rlc_circuit_results.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plots", flush=True)
    
    # CSVs
    np.savetxt(output_dir / 'ground_truth.csv', np.column_stack([t, x_true[:, 0], x_true[:, 1], E_true]),
               delimiter=',', header='t,q,p,E', comments='')
    for method in ['magnus', 'hnn', 'node']:
        np.savetxt(output_dir / f'{method}_pred.csv', 
                   np.column_stack([t, r[method]['x_pred'][:, 0], r[method]['x_pred'][:, 1], r[method]['E_pred']]),
                   delimiter=',', header='t,q,p,E', comments='')
    
    with open(output_dir / 'summary_stats.csv', 'w') as f:
        f.write('method,rmse_mean,rmse_std,violation_frac_mean,violation_frac_std\n')
        for method in ['magnus', 'hnn', 'node']:
            m = metrics[method]
            f.write(f"{method},{m['rmse_mean']:.8f},{m['rmse_std']:.8f},{m['violation_frac_mean']:.8f},{m['violation_frac_std']:.8f}\n")
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✓ Saved CSVs", flush=True)
    print("\n" + "="*70, flush=True)
    print("✓ EXPERIMENT 2 COMPLETE", flush=True)
    print("="*70, flush=True)
    
    return all_results, metrics

if __name__ == "__main__":
    all_results, metrics = main()
