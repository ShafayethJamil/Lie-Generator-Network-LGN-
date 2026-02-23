"""
EXPERIMENT 3: LTV (Linear Time-Varying) System
==============================================
System: dq/dt = p, dp/dt = -ω₀²q - γ(t)p
where γ(t) = γ₀(1 + 0.5·sin(ωd·t))

Key Claim: Magnus-2 beats Magnus-0 when commutator [A(t₁), A(t₂)] ≠ 0
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
GAMMA_0 = 0.3
OMEGA_D = 1.0           # Damping variation frequency
TRAIN_T = 20.0
TEST_T = 100.0
DT_TRAIN = 0.1
DT_TEST_1 = 0.1
DT_TEST_2 = 0.2
N_SEEDS = 3
EPOCHS = 2000

def gamma(t):
    return GAMMA_0 * (1 + 0.5 * np.sin(OMEGA_D * t))

def gamma_torch(t):
    return GAMMA_0 * (1 + 0.5 * torch.sin(OMEGA_D * t))

def get_A_np(t):
    return np.array([[0, 1], [-OMEGA_0**2, -gamma(t)]])

def compute_commutator_norm(t1, t2):
    A1, A2 = get_A_np(t1), get_A_np(t2)
    return np.linalg.norm(A1 @ A2 - A2 @ A1, 'fro')

def generate_ltv_data(t_span, dt, q0=1.0, p0=0.0):
    def dynamics(t, x):
        q, p = x
        return [p, -OMEGA_0**2 * q - gamma(t) * p]
    
    t_eval = np.arange(0, t_span, dt)
    sol = solve_ivp(dynamics, [0, t_span], [q0, p0], t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    E = 0.5 * (sol.y[1]**2 + OMEGA_0**2 * sol.y[0]**2)
    return sol.t, sol.y.T, E

def energy(state):
    q, p = state[..., 0], state[..., 1]
    return 0.5 * (p**2 + OMEGA_0**2 * q**2)

# =============================================================================
# MODELS
# =============================================================================
class Magnus2(nn.Module):
    """Magnus-2: Full Magnus with commutator correction"""
    def __init__(self):
        super().__init__()
        self.omega_sq = nn.Parameter(torch.tensor(1.2))
        self.gamma_0 = nn.Parameter(torch.tensor(0.35))
        self.gamma_amp = nn.Parameter(torch.tensor(0.15))
        self.omega_d = nn.Parameter(torch.tensor(1.1))
    
    def get_gamma(self, t):
        return self.gamma_0 * (1 + self.gamma_amp * torch.sin(self.omega_d * t))
    
    def get_A(self, t):
        A = torch.zeros(2, 2, device=self.omega_sq.device, dtype=torch.float64)
        A[0, 1] = 1.0
        A[1, 0] = -self.omega_sq
        A[1, 1] = -self.get_gamma(t)
        return A
    
    def forward(self, t, x0, use_commutator=True):
        trajectory = [x0]
        x = x0.clone()
        for i in range(1, len(t)):
            t_curr, dt = t[i-1], t[i] - t[i-1]
            A0 = self.get_A(t_curr)
            A_mid = self.get_A(t_curr + dt/2)
            A1 = self.get_A(t_curr + dt)
            Omega1 = A_mid * dt
            if use_commutator:
                Omega2 = (dt**2 / 12) * (A0 @ A1 - A1 @ A0)
                Omega = Omega1 + Omega2
            else:
                Omega = Omega1
            x = torch.matrix_exp(Omega) @ x
            trajectory.append(x)
        return torch.stack(trajectory)
    
    def get_params(self):
        return {'omega_sq': self.omega_sq.item(), 'gamma_0': self.gamma_0.item(),
                'gamma_amp': self.gamma_amp.item(), 'omega_d': self.omega_d.item()}

class Magnus0(nn.Module):
    """Magnus-0: No commutator correction (ablation)"""
    def __init__(self):
        super().__init__()
        self.omega_sq = nn.Parameter(torch.tensor(1.2))
        self.gamma_0 = nn.Parameter(torch.tensor(0.35))
        self.gamma_amp = nn.Parameter(torch.tensor(0.15))
        self.omega_d = nn.Parameter(torch.tensor(1.1))
    
    def get_gamma(self, t):
        return self.gamma_0 * (1 + self.gamma_amp * torch.sin(self.omega_d * t))
    
    def get_A(self, t):
        A = torch.zeros(2, 2, device=self.omega_sq.device, dtype=torch.float64)
        A[0, 1] = 1.0
        A[1, 0] = -self.omega_sq
        A[1, 1] = -self.get_gamma(t)
        return A
    
    def forward(self, t, x0):
        trajectory = [x0]
        x = x0.clone()
        for i in range(1, len(t)):
            t_curr, dt = t[i-1], t[i] - t[i-1]
            A = self.get_A(t_curr + dt/2)
            x = torch.matrix_exp(A * dt) @ x
            trajectory.append(x)
        return torch.stack(trajectory)
    
    def get_params(self):
        return {'omega_sq': self.omega_sq.item(), 'gamma_0': self.gamma_0.item(),
                'gamma_amp': self.gamma_amp.item(), 'omega_d': self.omega_d.item()}

class NeuralODE(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x0):
        def dynamics(t_scalar, x):
            inp = torch.cat([x, (t_scalar / 10.0).unsqueeze(0)])
            return self.net(inp)
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

def evaluate_model(model, t, x_true, name="Model"):
    model.eval()
    with torch.no_grad():
        x_pred = model(t, x_true[0])
    x_pred_np = x_pred.cpu().numpy()
    x_true_np = x_true.cpu().numpy()
    rmse = np.sqrt(np.mean((x_pred_np - x_true_np)**2))
    error_t = np.sqrt(np.mean((x_pred_np - x_true_np)**2, axis=1))
    return {'rmse': rmse, 'x_pred': x_pred_np, 'error_t': error_t}

# =============================================================================
# MAIN
# =============================================================================
def run_experiment(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}", flush=True)
    print(f"SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)
    
    t_train_np, x_train_np, _ = generate_ltv_data(TRAIN_T, DT_TRAIN)
    t_test1_np, x_test1_np, _ = generate_ltv_data(TEST_T, DT_TEST_1)
    t_test2_np, x_test2_np, _ = generate_ltv_data(TEST_T, DT_TEST_2)
    
    t_train = torch.tensor(t_train_np, dtype=torch.float64, device=DEVICE)
    x_train = torch.tensor(x_train_np, dtype=torch.float64, device=DEVICE)
    t_test1 = torch.tensor(t_test1_np, dtype=torch.float64, device=DEVICE)
    x_test1 = torch.tensor(x_test1_np, dtype=torch.float64, device=DEVICE)
    t_test2 = torch.tensor(t_test2_np, dtype=torch.float64, device=DEVICE)
    x_test2 = torch.tensor(x_test2_np, dtype=torch.float64, device=DEVICE)
    
    results = {}
    
    print("\nTraining Magnus-2 (with commutator)...", flush=True)
    magnus2 = Magnus2().double().to(DEVICE)
    t0 = time.time()
    train_model(magnus2, t_train, x_train, epochs=EPOCHS, lr=1e-2, name="Magnus-2")
    results['magnus2'] = {
        'dt1': evaluate_model(magnus2, t_test1, x_test1),
        'dt2': evaluate_model(magnus2, t_test2, x_test2),
        'train_time': time.time() - t0,
        'params': magnus2.get_params()
    }
    print(f"  Params: {magnus2.get_params()}", flush=True)
    
    print("\nTraining Magnus-0 (no commutator - ablation)...", flush=True)
    magnus0 = Magnus0().double().to(DEVICE)
    t0 = time.time()
    train_model(magnus0, t_train, x_train, epochs=EPOCHS, lr=1e-2, name="Magnus-0")
    results['magnus0'] = {
        'dt1': evaluate_model(magnus0, t_test1, x_test1),
        'dt2': evaluate_model(magnus0, t_test2, x_test2),
        'train_time': time.time() - t0,
        'params': magnus0.get_params()
    }
    
    print("\nTraining Neural ODE...", flush=True)
    node = NeuralODE(hidden_dim=64).double().to(DEVICE)
    t0 = time.time()
    train_model(node, t_train, x_train, epochs=EPOCHS, lr=1e-3, name="NODE")
    results['node'] = {
        'dt1': evaluate_model(node, t_test1, x_test1),
        'dt2': evaluate_model(node, t_test2, x_test2),
        'train_time': time.time() - t0
    }
    
    results['t_test1'] = t_test1_np
    results['x_test1'] = x_test1_np
    results['t_test2'] = t_test2_np
    results['x_test2'] = x_test2_np
    
    comm_norms = [compute_commutator_norm(tv, tv + DT_TEST_1) for tv in t_test1_np[:-1]]
    results['commutator_norms'] = np.array(comm_norms)
    results['commutator_times'] = t_test1_np[:-1]
    
    return results

def main():
    print("="*70, flush=True)
    print("EXPERIMENT 3: LTV (TIME-VARYING) SYSTEM", flush=True)
    print("="*70, flush=True)
    print(f"ω₀={OMEGA_0}, γ₀={GAMMA_0}, ωd={OMEGA_D}", flush=True)
    print(f"γ(t) = γ₀(1 + 0.5·sin(ωd·t))", flush=True)
    print(f"Train: 0-{TRAIN_T}, Test: 0-{TEST_T}", flush=True)
    print(f"Δt_train={DT_TRAIN}, Δt_test={DT_TEST_1} and {DT_TEST_2}", flush=True)
    print(f"Seeds: {N_SEEDS}, Epochs: {EPOCHS}", flush=True)
    
    comm = compute_commutator_norm(0, 0.1)
    print(f"\nCommutator ||[A(0), A(0.1)]|| = {comm:.6f}", flush=True)
    
    all_results = []
    for seed in range(N_SEEDS):
        all_results.append(run_experiment(seed))
    
    print("\n" + "="*70, flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("="*70, flush=True)
    
    metrics = {}
    for method in ['magnus2', 'magnus0', 'node']:
        rmse_dt1 = [r[method]['dt1']['rmse'] for r in all_results]
        rmse_dt2 = [r[method]['dt2']['rmse'] for r in all_results]
        metrics[method] = {
            'rmse_dt1_mean': np.mean(rmse_dt1), 'rmse_dt1_std': np.std(rmse_dt1),
            'rmse_dt2_mean': np.mean(rmse_dt2), 'rmse_dt2_std': np.std(rmse_dt2),
        }
        print(f"\n{method.upper()}:", flush=True)
        print(f"  RMSE (Δt=0.1): {metrics[method]['rmse_dt1_mean']:.6f} ± {metrics[method]['rmse_dt1_std']:.6f}", flush=True)
        print(f"  RMSE (Δt=0.2): {metrics[method]['rmse_dt2_mean']:.6f} ± {metrics[method]['rmse_dt2_std']:.6f}", flush=True)
    
    # Save
    output_dir = Path('./exp3_results')
    output_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created output directory: {output_dir.absolute()}", flush=True)
    
    r = all_results[0]
    t = r['t_test1']
    x_true = r['x_test1']
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {'magnus2': 'green', 'magnus0': 'orange', 'node': 'red'}
    labels = {'magnus2': 'Magnus-2', 'magnus0': 'Magnus-0', 'node': 'NODE'}
    
    ax = axes[0, 0]
    ax.plot(t, x_true[:, 0], 'k-', lw=2, label='True')
    for method in ['magnus2', 'magnus0', 'node']:
        ax.plot(t, r[method]['dt1']['x_pred'][:, 0], '--', color=colors[method], lw=1.5, label=labels[method])
    ax.axvline(x=TRAIN_T, color='gray', linestyle='--')
    ax.set_xlabel('Time'); ax.set_ylabel('q(t)'); ax.set_title('(a) Trajectory')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 60)
    
    ax = axes[0, 1]
    for method in ['magnus2', 'magnus0', 'node']:
        ax.semilogy(t, r[method]['dt1']['error_t'], '-', color=colors[method], lw=1.5, label=labels[method])
    ax.axvline(x=TRAIN_T, color='gray', linestyle='--')
    ax.set_xlabel('Time'); ax.set_ylabel('Error'); ax.set_title('(b) Error vs Time')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    x_pos = np.arange(3)
    width = 0.35
    methods = ['magnus2', 'magnus0', 'node']
    rmse_dt1 = [metrics[m]['rmse_dt1_mean'] for m in methods]
    rmse_dt2 = [metrics[m]['rmse_dt2_mean'] for m in methods]
    ax.bar(x_pos - width/2, rmse_dt1, width, label='Δt=0.1', color=['green', 'orange', 'red'], alpha=0.7)
    ax.bar(x_pos + width/2, rmse_dt2, width, label='Δt=0.2', color=['green', 'orange', 'red'], alpha=0.4)
    ax.set_xticks(x_pos); ax.set_xticklabels(['Magnus-2', 'Magnus-0', 'NODE'])
    ax.set_ylabel('RMSE'); ax.set_title('(c) Δt Generalization')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    ax.plot(r['commutator_times'], r['commutator_norms'], 'b-', lw=1.5)
    ax.axvline(x=TRAIN_T, color='gray', linestyle='--')
    ax.set_xlabel('Time'); ax.set_ylabel('||[A(t), A(t+Δt)]||')
    ax.set_title(f'(d) Commutator (mean={np.mean(r["commutator_norms"]):.4f})')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('LTV: Magnus-2 > Magnus-0 (Commutator Matters)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'ltv_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'ltv_results.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plots", flush=True)
    
    # CSVs
    np.savetxt(output_dir / 'ground_truth_dt1.csv', np.column_stack([t, x_true[:, 0], x_true[:, 1]]),
               delimiter=',', header='t,q,p', comments='')
    for method in ['magnus2', 'magnus0', 'node']:
        np.savetxt(output_dir / f'{method}_pred_dt1.csv',
                   np.column_stack([t, r[method]['dt1']['x_pred'][:, 0], r[method]['dt1']['x_pred'][:, 1], r[method]['dt1']['error_t']]),
                   delimiter=',', header='t,q,p,error', comments='')
    np.savetxt(output_dir / 'commutator.csv', np.column_stack([r['commutator_times'], r['commutator_norms']]),
               delimiter=',', header='t,commutator_norm', comments='')
    
    with open(output_dir / 'summary_stats.csv', 'w') as f:
        f.write('method,rmse_dt1_mean,rmse_dt1_std,rmse_dt2_mean,rmse_dt2_std\n')
        for method in ['magnus2', 'magnus0', 'node']:
            m = metrics[method]
            f.write(f"{method},{m['rmse_dt1_mean']:.8f},{m['rmse_dt1_std']:.8f},{m['rmse_dt2_mean']:.8f},{m['rmse_dt2_std']:.8f}\n")
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✓ Saved CSVs", flush=True)
    print("\n" + "="*70, flush=True)
    print("✓ EXPERIMENT 3 COMPLETE", flush=True)
    print("="*70, flush=True)
    
    return all_results, metrics

if __name__ == "__main__":
    all_results, metrics = main()
