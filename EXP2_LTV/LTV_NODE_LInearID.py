"""
EXPERIMENT 3 : LTV System
===============================================================
System: dq/dt = p, dp/dt = -ω₀²q - γ(t)p  [UNKNOWN TO ALL MODELS]

BASELINES:
- NODE-RFF (~4500 params): Same Fourier features as LGN (isolates linearity)
- NODE-small (~200 params): Param-matched (isolates efficiency)
- NODE-raw (~4500 params): Original with scalar time
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
# SYSTEM PARAMETERS (Ground truth - COMPLETELY UNKNOWN to all models)
# =============================================================================
OMEGA_0 = 1.0
GAMMA_0 = 0.3
OMEGA_D = 1.0
TRAIN_T = 20.0
TEST_T = 100.0
DT = 0.1
N_SEEDS = 3
EPOCHS = 2000

# Shared Fourier features config
N_FREQ = 25
FREQ_MIN = 0.1
FREQ_MAX = 10.0  # Broad range, not tuned to ω_d=1

def gamma_true(t):
    return GAMMA_0 * (1 + 0.5 * np.sin(OMEGA_D * t))

def get_A_true(t):
    return np.array([[0, 1], [-OMEGA_0**2, -gamma_true(t)]])

def compute_commutator_norm(t1, t2):
    A1, A2 = get_A_true(t1), get_A_true(t2)
    return np.linalg.norm(A1 @ A2 - A2 @ A1, 'fro')

def generate_ltv_data(t_span, dt, q0=1.0, p0=0.0):
    def dynamics(t, x):
        q, p = x
        return [p, -OMEGA_0**2 * q - gamma_true(t) * p]
    
    t_eval = np.arange(0, t_span, dt)
    sol = solve_ivp(dynamics, [0, t_span], [q0, p0], t_eval=t_eval, 
                    method='RK45', rtol=1e-10, atol=1e-12)
    return sol.t, sol.y.T


def get_shared_frequencies(n_freq, freq_min, freq_max, seed=42):
    """Generate log-uniform random frequencies (shared by LGN and NODE-RFF)"""
    rng = np.random.default_rng(seed)  # Local RNG to avoid side effects
    log_freqs = rng.uniform(np.log(freq_min), np.log(freq_max), n_freq)
    return torch.tensor(np.exp(log_freqs), dtype=torch.float64)


# Global shared frequencies (same for all models)
SHARED_FREQUENCIES = get_shared_frequencies(N_FREQ, FREQ_MIN, FREQ_MAX)
contains_omega = ((SHARED_FREQUENCIES > OMEGA_D*0.9) & (SHARED_FREQUENCIES < OMEGA_D*1.1)).any().item()
print(f"Shared frequencies (log-uniform [{FREQ_MIN}, {FREQ_MAX}]):", flush=True)
print(f"  min={SHARED_FREQUENCIES.min().item():.3f}, max={SHARED_FREQUENCIES.max().item():.3f}, "
      f"contains ω_d={OMEGA_D}? {contains_omega}", flush=True)


# =============================================================================
# SHARED: Fourier Feature Extractor
# =============================================================================
class FourierFeatures(nn.Module):
    """Shared Fourier feature extractor for fair comparison"""
    def __init__(self, frequencies):
        super().__init__()
        self.register_buffer('frequencies', frequencies)
        self.n_features = 2 * len(frequencies)  # cos + sin
    
    def forward(self, t):
        """Returns [cos(ω_k t), sin(ω_k t)] for all frequencies"""
        phases = self.frequencies * t
        return torch.cat([torch.cos(phases), torch.sin(phases)])


# =============================================================================
# MODEL 1: COMPACT LGN - Random Fourier Features (~200 params) (uncostrained)
# =============================================================================
class LGNCompact(nn.Module):
    """
    LGN with Fourier features + linear readout.
    A(t) = W @ φ(t) + b, reshaped to 2x2
    
    Parameters: 2*n_freq*4 + 4 = 204 (for n_freq=25)
    """
    def __init__(self, state_dim=2, frequencies=None):
        super().__init__()
        self.state_dim = state_dim
        self.fourier = FourierFeatures(frequencies)
        
        # Linear readout: 2*n_freq -> 4 entries of A(t)
        self.readout = nn.Linear(self.fourier.n_features, state_dim * state_dim).double()
        nn.init.xavier_normal_(self.readout.weight, gain=0.3)
        nn.init.zeros_(self.readout.bias)
    
    def get_A(self, t):
        features = self.fourier(t)
        A_flat = self.readout(features)
        return A_flat.view(self.state_dim, self.state_dim)
    
    def forward(self, t, x0, use_commutator=True):
        trajectory = [x0]
        x = x0.clone()
        
        for i in range(1, len(t)):
            t_curr = t[i-1]
            dt_step = t[i] - t[i-1]
            
            A0 = self.get_A(t_curr)
            A_mid = self.get_A(t_curr + dt_step/2)
            A1 = self.get_A(t_curr + dt_step)
            
            Omega = A_mid * dt_step
            if use_commutator:
                commutator = A0 @ A1 - A1 @ A0
                Omega = Omega + (dt_step**2 / 12) * commutator
            
            x = torch.matrix_exp(Omega) @ x
            trajectory.append(x)
        
        return torch.stack(trajectory)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_learned_A(self, t_array):
        A_list = []
        with torch.no_grad():
            for t in t_array:
                t_tensor = torch.tensor(t, dtype=torch.float64, device=DEVICE)
                A_list.append(self.get_A(t_tensor).cpu().numpy())
        return np.array(A_list)


class LGNM1(LGNCompact):
    def forward(self, t, x0):
        return super().forward(t, x0, use_commutator=False)

class LGNM2(LGNCompact):
    def forward(self, t, x0):
        return super().forward(t, x0, use_commutator=True)


# =============================================================================
# MODEL 2: NODE-RFF - Same Fourier features as LGN 
# =============================================================================
class NeuralODE_RFF(nn.Module):
    """
    Neural ODE with SAME Fourier features as LGN.
    f(x, t) = MLP([x, φ(t)])
    
    This isolates LGN's advantage to: linearity + Magnus/expm
    """
    def __init__(self, state_dim=2, hidden_dim=64, frequencies=None):
        super().__init__()
        self.fourier = FourierFeatures(frequencies)
        
        input_dim = state_dim + self.fourier.n_features  # x + φ(t)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x0):
        def dynamics(t_scalar, x):
            phi_t = self.fourier(t_scalar)
            inp = torch.cat([x, phi_t])
            return self.net(inp)
        return odeint(dynamics, x0, t, method='dopri5', rtol=1e-5, atol=1e-6)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# MODEL 3: NODE-small - Parameter matched (~200 params)
# =============================================================================
class NeuralODE_Small(nn.Module):
    """
    Neural ODE with ~200 params to match LGN.
    Uses same Fourier features + small hidden layer.
    
    input_dim = 2 + 2*25 = 52
    hidden_dim = 4
    params = (52*4 + 4) + (4*2 + 2) = 212 + 10 = 222
    """
    def __init__(self, state_dim=2, hidden_dim=4, frequencies=None):  # FIXED: hidden_dim=4
        super().__init__()
        self.fourier = FourierFeatures(frequencies)
        
        input_dim = state_dim + self.fourier.n_features
        # Small network: input -> 4 -> output
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x0):
        def dynamics(t_scalar, x):
            phi_t = self.fourier(t_scalar)
            inp = torch.cat([x, phi_t])
            return self.net(inp)
        return odeint(dynamics, x0, t, method='dopri5', rtol=1e-5, atol=1e-6)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# MODEL 4: NODE-raw - Original with scalar time (for reference)
# =============================================================================
class NeuralODE_Raw(nn.Module):
    """Original NODE with scalar time input"""
    def __init__(self, state_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x0):
        def dynamics(t_scalar, x):
            inp = torch.cat([x, (t_scalar / 10.0).unsqueeze(0)])
            return self.net(inp)
        return odeint(dynamics, x0, t, method='dopri5', rtol=1e-5, atol=1e-6)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# MODEL 5: Linear-ID (constant A baseline)
# =============================================================================
def fit_linear_id(t, x):
    dt = t[1] - t[0]
    dx = np.gradient(x, dt, axis=0)
    A_T, _, _, _ = np.linalg.lstsq(x, dx, rcond=None)
    return A_T.T

def rollout_linear(A, x0, t):
    from scipy.linalg import expm
    dt = t[1] - t[0]
    expAdt = expm(A * dt)
    trajectory = [x0]
    x = x0.copy()
    for _ in range(1, len(t)):
        x = expAdt @ x
        trajectory.append(x)
    return np.array(trajectory)


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
                print(f"  {name} Ep {epoch+1}/{epochs}: Loss={loss.item():.6e}", flush=True)
        except Exception as e:
            print(f"  {name} Ep {epoch+1}: Error - {e}", flush=True)


def evaluate_model(model, t, x_true, is_numpy=False):
    if is_numpy:
        x_pred = model
        x_true_np = x_true
    else:
        model.eval()
        with torch.no_grad():
            x_pred = model(t, x_true[0])
        x_pred = x_pred.cpu().numpy()
        x_true_np = x_true.cpu().numpy()
    
    rmse = np.sqrt(np.mean((x_pred - x_true_np)**2))
    error_t = np.sqrt(np.mean((x_pred - x_true_np)**2, axis=1))
    norm_true = np.sqrt(np.mean(x_true_np**2))
    nrmse = rmse / (norm_true + 1e-10)
    
    return {'rmse': rmse, 'nrmse': nrmse, 'x_pred': x_pred, 'error_t': error_t}


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
    t_train_np, x_train_np = generate_ltv_data(TRAIN_T, DT)
    t_test_np, x_test_np = generate_ltv_data(TEST_T, DT)
    
    t_train = torch.tensor(t_train_np, dtype=torch.float64, device=DEVICE)
    x_train = torch.tensor(x_train_np, dtype=torch.float64, device=DEVICE)
    t_test = torch.tensor(t_test_np, dtype=torch.float64, device=DEVICE)
    x_test = torch.tensor(x_test_np, dtype=torch.float64, device=DEVICE)
    
    # Move shared frequencies to device
    frequencies = SHARED_FREQUENCIES.to(DEVICE)
    
    results = {}
    
    # --- LGN-M2 ---
    print(f"\nTraining LGN-M2 (Fourier + Linear, with commutator)...", flush=True)
    lgn_m2 = LGNM2(state_dim=2, frequencies=frequencies).to(DEVICE)
    print(f"  Parameters: {lgn_m2.count_params()}", flush=True)
    t0 = time.time()
    train_model(lgn_m2, t_train, x_train, epochs=EPOCHS, lr=1e-2, name="LGN-M2")
    results['lgn_m2'] = evaluate_model(lgn_m2, t_test, x_test)
    results['lgn_m2']['train_time'] = time.time() - t0
    results['lgn_m2']['n_params'] = lgn_m2.count_params()
    
    # --- LGN-M1 ---
    print(f"\nTraining LGN-M1 (Fourier + Linear, no commutator)...", flush=True)
    lgn_m1 = LGNM1(state_dim=2, frequencies=frequencies).to(DEVICE)
    print(f"  Parameters: {lgn_m1.count_params()}", flush=True)
    t0 = time.time()
    train_model(lgn_m1, t_train, x_train, epochs=EPOCHS, lr=1e-2, name="LGN-M1")
    results['lgn_m1'] = evaluate_model(lgn_m1, t_test, x_test)
    results['lgn_m1']['train_time'] = time.time() - t0
    results['lgn_m1']['n_params'] = lgn_m1.count_params()
    
    # --- NODE-RFF (same Fourier features) ---
    print(f"\nTraining NODE-RFF (same Fourier features as LGN)...", flush=True)
    node_rff = NeuralODE_RFF(state_dim=2, hidden_dim=64, frequencies=frequencies).double().to(DEVICE)
    print(f"  Parameters: {node_rff.count_params()}", flush=True)
    t0 = time.time()
    train_model(node_rff, t_train, x_train, epochs=EPOCHS, lr=1e-3, name="NODE-RFF")
    results['node_rff'] = evaluate_model(node_rff, t_test, x_test)
    results['node_rff']['train_time'] = time.time() - t0
    results['node_rff']['n_params'] = node_rff.count_params()
    
    # --- NODE-small (param-matched) ---
    print(f"\nTraining NODE-small (param-matched ~200)...", flush=True)
    node_small = NeuralODE_Small(state_dim=2, hidden_dim=4, frequencies=frequencies).double().to(DEVICE)
    print(f"  Parameters: {node_small.count_params()}", flush=True)
    t0 = time.time()
    train_model(node_small, t_train, x_train, epochs=EPOCHS, lr=1e-3, name="NODE-small")
    results['node_small'] = evaluate_model(node_small, t_test, x_test)
    results['node_small']['train_time'] = time.time() - t0
    results['node_small']['n_params'] = node_small.count_params()
    
    # --- NODE-raw (original, for reference) ---
    print(f"\nTraining NODE-raw (scalar time)...", flush=True)
    node_raw = NeuralODE_Raw(state_dim=2, hidden_dim=64).double().to(DEVICE)
    print(f"  Parameters: {node_raw.count_params()}", flush=True)
    t0 = time.time()
    train_model(node_raw, t_train, x_train, epochs=EPOCHS, lr=1e-3, name="NODE-raw")
    results['node_raw'] = evaluate_model(node_raw, t_test, x_test)
    results['node_raw']['train_time'] = time.time() - t0
    results['node_raw']['n_params'] = node_raw.count_params()
    
    # --- Linear-ID ---
    print(f"\nFitting Linear-ID...", flush=True)
    A_linear = fit_linear_id(t_train_np, x_train_np)
    x_pred_linear = rollout_linear(A_linear, x_test_np[0], t_test_np)
    results['linear_id'] = evaluate_model(x_pred_linear, t_test_np, x_test_np, is_numpy=True)
    results['linear_id']['n_params'] = 4
    
    results['t_test'] = t_test_np
    results['x_test'] = x_test_np
    
    return results


def main():
    print("="*70, flush=True)
    print("EXPERIMENT 3 (BULLETPROOF): LTV SYSTEM", flush=True)
    print("="*70, flush=True)
    print(f"\nGround truth (UNKNOWN): γ(t) = {GAMMA_0}(1 + 0.5·sin({OMEGA_D}·t))", flush=True)
    print(f"\nFAIRNESS CONTROLS:", flush=True)
    print(f"  1. Shared Fourier features (log-uniform [{FREQ_MIN}, {FREQ_MAX}])", flush=True)
    print(f"  2. NODE-RFF: same features as LGN (isolates linearity)", flush=True)
    print(f"  3. NODE-small: param-matched ~200 (isolates efficiency)", flush=True)
    print(f"\nTrain: 0-{TRAIN_T}, Test: 0-{TEST_T}", flush=True)
    print(f"Seeds: {N_SEEDS}, Epochs: {EPOCHS}", flush=True)
    
    all_results = []
    for seed in range(N_SEEDS):
        all_results.append(run_experiment(seed))
    
    # Aggregate
    print("\n" + "="*70, flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("="*70, flush=True)
    
    methods = ['lgn_m2', 'lgn_m1', 'node_rff', 'node_small', 'node_raw', 'linear_id']
    metrics = {}
    
    for method in methods:
        rmse_list = [r[method]['rmse'] for r in all_results]
        nrmse_list = [r[method]['nrmse'] for r in all_results]
        n_params = all_results[0][method]['n_params']
        
        metrics[method] = {
            'rmse_mean': np.mean(rmse_list),
            'rmse_std': np.std(rmse_list),
            'nrmse_mean': np.mean(nrmse_list),
            'nrmse_std': np.std(nrmse_list),
            'n_params': n_params
        }
        print(f"\n{method.upper()} ({n_params} params):", flush=True)
        print(f"  RMSE: {metrics[method]['rmse_mean']:.6e} ± {metrics[method]['rmse_std']:.6e}", flush=True)
    
    # Claims
    print("\n" + "="*70, flush=True)
    print("CLAIM VERIFICATION", flush=True)
    print("="*70, flush=True)
    
    # Claim 1: Commutator helps
    if metrics['lgn_m2']['rmse_mean'] < metrics['lgn_m1']['rmse_mean']:
        ratio = metrics['lgn_m1']['rmse_mean'] / metrics['lgn_m2']['rmse_mean']
        print(f"✓ CLAIM 1: Commutator gives {ratio:.2f}× improvement", flush=True)
    else:
        print(f"✗ CLAIM 1: Commutator did NOT help", flush=True)
    
    # Claim 2: Linearity helps (LGN vs NODE-RFF, same features)
    if metrics['lgn_m2']['rmse_mean'] < metrics['node_rff']['rmse_mean']:
        ratio = metrics['node_rff']['rmse_mean'] / metrics['lgn_m2']['rmse_mean']
        print(f"✓ CLAIM 2: Linear structure gives {ratio:.2f}× improvement (same features)", flush=True)
    else:
        print(f"✗ CLAIM 2: Linearity did NOT help vs NODE-RFF", flush=True)
    
    # Claim 3: Param efficiency (LGN vs NODE-small, same params)
    if metrics['lgn_m2']['rmse_mean'] < metrics['node_small']['rmse_mean']:
        ratio = metrics['node_small']['rmse_mean'] / metrics['lgn_m2']['rmse_mean']
        print(f"✓ CLAIM 3: LGN beats param-matched NODE by {ratio:.2f}×", flush=True)
    else:
        print(f"✗ CLAIM 3: Param-matched NODE wins", flush=True)
    
    # Save
    output_dir = Path('./exp3_bulletproof_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot
    r = all_results[0]
    t = r['t_test']
    x_true = r['x_test']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'lgn_m2': 'green', 'lgn_m1': 'limegreen', 
              'node_rff': 'red', 'node_small': 'orange', 'node_raw': 'darkred',
              'linear_id': 'purple'}
    
    # (a) Trajectory
    ax = axes[0, 0]
    ax.plot(t, x_true[:, 0], 'k-', lw=2, label='Truth')
    for m in ['lgn_m2', 'node_rff', 'node_small']:
        ax.plot(t, r[m]['x_pred'][:, 0], '--', color=colors[m], lw=1.5, 
               label=f'{m} ({metrics[m]["n_params"]}p)')
    ax.axvline(x=TRAIN_T, color='gray', ls='--')
    ax.set_xlim(0, 60)
    ax.legend(fontsize=8)
    ax.set_title('(a) Trajectory')
    ax.grid(True, alpha=0.3)
    
    # (b) Error
    ax = axes[0, 1]
    for m in ['lgn_m2', 'lgn_m1', 'node_rff', 'node_small']:
        ax.semilogy(t, r[m]['error_t'] + 1e-16, color=colors[m], lw=1.5, label=m)
    ax.axvline(x=TRAIN_T, color='gray', ls='--')
    ax.legend(fontsize=8)
    ax.set_title('(b) Error vs Time')
    ax.grid(True, alpha=0.3)
    
    # (c) RMSE bar chart
    ax = axes[1, 0]
    x_pos = np.arange(len(methods))
    rmse_vals = [metrics[m]['rmse_mean'] for m in methods]
    rmse_stds = [metrics[m]['rmse_std'] for m in methods]
    bars = ax.bar(x_pos, rmse_vals, yerr=rmse_stds, 
                  color=[colors[m] for m in methods], alpha=0.7, capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{m}\n({metrics[m]["n_params"]}p)' for m in methods], fontsize=7)
    ax.set_yscale('log')
    ax.set_ylabel('Test RMSE')
    ax.set_title('(c) RMSE Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # (d) Params vs RMSE
    ax = axes[1, 1]
    for m in methods:
        ax.scatter(metrics[m]['n_params'], metrics[m]['rmse_mean'], 
                  s=100, c=colors[m], label=m, zorder=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Test RMSE')
    ax.set_title('(d) Parameter Efficiency')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('LTV: Bulletproof Comparison\n(Shared Fourier features, param-matched baselines)', 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'results.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'results.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved to {output_dir}", flush=True)
    
    return metrics


if __name__ == "__main__":
    metrics = main()
