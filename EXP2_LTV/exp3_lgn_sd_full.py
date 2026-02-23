"""
LTV Experiment: LGN-SD Full Run
================================
- LGN-M1-SD (no commutator)
- LGN-M2-SD (with commutator)
- 3 seeds, 2000 epochs
- Saves trajectories for phase plots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import sys
from pathlib import Path
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(line_buffering=True)

from torchdiffeq import odeint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================
OMEGA_0 = 1.0
GAMMA_0 = 0.3
GAMMA_AMP = 0.5
OMEGA_D = 1.0
TRAIN_T = 20.0
TEST_T = 100.0
DT = 0.1

N_SEEDS = 3
EPOCHS = 2000
N_FREQ = 25

print(f"System: γ(t) = {GAMMA_0}(1 + {GAMMA_AMP}·sin({OMEGA_D}·t))", flush=True)
print(f"Train: 0-{TRAIN_T}, Test: 0-{TEST_T}, dt={DT}", flush=True)
print(f"Seeds: {N_SEEDS}, Epochs: {EPOCHS}", flush=True)

# =============================================================================
# DATA GENERATION
# =============================================================================
def gamma_true(t):
    return GAMMA_0 * (1 + GAMMA_AMP * np.sin(OMEGA_D * t))

def generate_data(t_span, dt):
    def dynamics(t, x):
        return [x[1], -OMEGA_0**2 * x[0] - gamma_true(t) * x[1]]
    t_eval = np.arange(0, t_span, dt)
    sol = solve_ivp(dynamics, [0, t_span], [1.0, 0.0], t_eval=t_eval, 
                    method='RK45', rtol=1e-10, atol=1e-12)
    return sol.t, sol.y.T

# =============================================================================
# FOURIER FEATURES (shared, fixed)
# =============================================================================
rng = np.random.default_rng(42)
FREQS = torch.tensor(np.exp(rng.uniform(np.log(0.1), np.log(10), N_FREQ)), 
                     dtype=torch.float64, device=DEVICE)

class FourierFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('freqs', FREQS)
    
    def forward(self, t):
        phases = self.freqs * t
        return torch.cat([torch.cos(phases), torch.sin(phases)])

# =============================================================================
# LGN-SD MODEL
# =============================================================================
class LGN_SD(nn.Module):
    """A(t) = S(t) - D(t), guaranteed stable"""
    def __init__(self):
        super().__init__()
        self.fourier = FourierFeatures()
        self.S_readout = nn.Linear(50, 1).double()
        self.D_readout = nn.Linear(50, 2).double()
        nn.init.xavier_normal_(self.S_readout.weight, gain=0.5)
        nn.init.xavier_normal_(self.D_readout.weight, gain=0.1)
        nn.init.zeros_(self.S_readout.bias)
        nn.init.zeros_(self.D_readout.bias)
    
    def get_A(self, t):
        f = self.fourier(t)
        s = self.S_readout(f).squeeze()
        S = torch.zeros(2, 2, dtype=torch.float64, device=f.device)
        S[0, 1] = s
        S[1, 0] = -s
        d = F.softplus(self.D_readout(f))
        return S - torch.diag(d)
    
    def forward(self, t, x0, use_commutator=True):
        traj = [x0]
        x = x0.clone()
        for i in range(1, len(t)):
            dt_step = t[i] - t[i-1]
            t_curr = t[i-1]
            
            A0 = self.get_A(t_curr)
            Am = self.get_A(t_curr + dt_step/2)
            A1 = self.get_A(t_curr + dt_step)
            
            Omega = Am * dt_step
            if use_commutator:
                comm = A0 @ A1 - A1 @ A0
                Omega = Omega + (dt_step**2 / 12) * comm
            
            x = torch.matrix_exp(Omega) @ x
            traj.append(x)
        return torch.stack(traj)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LGN_M1_SD(LGN_SD):
    def forward(self, t, x0):
        return super().forward(t, x0, use_commutator=False)


class LGN_M2_SD(LGN_SD):
    def forward(self, t, x0):
        return super().forward(t, x0, use_commutator=True)


# =============================================================================
# TRAINING
# =============================================================================
def train(model, t, x, epochs, lr, name):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.5)
    x0 = x[0]
    
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(t, x0)
        loss = torch.mean((pred - x)**2)
        
        if torch.isnan(loss):
            print(f"  {name}: NaN at epoch {ep}", flush=True)
            break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step(loss)
        
        if (ep + 1) % 400 == 0:
            print(f"  {name} Ep {ep+1}/{epochs}: Loss={loss.item():.6e}", flush=True)


def evaluate(model, t, x_true):
    model.eval()
    with torch.no_grad():
        x_pred = model(t, x_true[0]).cpu().numpy()
    x_true_np = x_true.cpu().numpy()
    
    rmse = np.sqrt(np.mean((x_pred - x_true_np)**2))
    norm = np.sqrt(np.mean(x_true_np**2))
    nrmse = rmse / norm
    
    return {'rmse': rmse, 'nrmse': nrmse, 'x_pred': x_pred}


# =============================================================================
# MAIN
# =============================================================================
def main():
    output_dir = Path('./exp3_lgn_sd_full')
    output_dir.mkdir(exist_ok=True)
    
    # Generate data once
    t_train_np, x_train_np = generate_data(TRAIN_T, DT)
    t_test_np, x_test_np = generate_data(TEST_T, DT)
    
    # Save ground truth
    np.savetxt(output_dir / 'ground_truth_train.csv', 
               np.column_stack([t_train_np, x_train_np]), 
               delimiter=',', header='t,q,p', comments='')
    np.savetxt(output_dir / 'ground_truth_test.csv', 
               np.column_stack([t_test_np, x_test_np]), 
               delimiter=',', header='t,q,p', comments='')
    print(f"✓ Saved ground truth", flush=True)
    
    t_train = torch.tensor(t_train_np, dtype=torch.float64, device=DEVICE)
    x_train = torch.tensor(x_train_np, dtype=torch.float64, device=DEVICE)
    t_test = torch.tensor(t_test_np, dtype=torch.float64, device=DEVICE)
    x_test = torch.tensor(x_test_np, dtype=torch.float64, device=DEVICE)
    
    results = {'lgn_m1_sd': [], 'lgn_m2_sd': []}
    
    for seed in range(N_SEEDS):
        print(f"\n{'='*60}", flush=True)
        print(f"SEED {seed}", flush=True)
        print(f"{'='*60}", flush=True)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # --- LGN-M1-SD ---
        print(f"\nTraining LGN-M1-SD (no commutator)...", flush=True)
        m1 = LGN_M1_SD().to(DEVICE)
        if seed == 0:
            print(f"  Params: {m1.count_params()}", flush=True)
        
        t0 = time.time()
        train(m1, t_train, x_train, EPOCHS, lr=1e-2, name="LGN-M1-SD")
        train_time = time.time() - t0
        
        res_m1 = evaluate(m1, t_test, x_test)
        res_m1['train_time'] = train_time
        results['lgn_m1_sd'].append(res_m1)
        print(f"  Test RMSE: {res_m1['rmse']:.6f}, NRMSE: {res_m1['nrmse']*100:.2f}%", flush=True)
        
        # Save trajectory
        np.savetxt(output_dir / f'lgn_m1_sd_pred_seed{seed}.csv',
                   np.column_stack([t_test_np, res_m1['x_pred']]),
                   delimiter=',', header='t,q,p', comments='')
        
        # --- LGN-M2-SD ---
        print(f"\nTraining LGN-M2-SD (with commutator)...", flush=True)
        m2 = LGN_M2_SD().to(DEVICE)
        
        t0 = time.time()
        train(m2, t_train, x_train, EPOCHS, lr=1e-2, name="LGN-M2-SD")
        train_time = time.time() - t0
        
        res_m2 = evaluate(m2, t_test, x_test)
        res_m2['train_time'] = train_time
        results['lgn_m2_sd'].append(res_m2)
        print(f"  Test RMSE: {res_m2['rmse']:.6f}, NRMSE: {res_m2['nrmse']*100:.2f}%", flush=True)
        
        # Save trajectory
        np.savetxt(output_dir / f'lgn_m2_sd_pred_seed{seed}.csv',
                   np.column_stack([t_test_np, res_m2['x_pred']]),
                   delimiter=',', header='t,q,p', comments='')
    
    # ==========================================================================
    # AGGREGATE
    # ==========================================================================
    print(f"\n{'='*60}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    
    metrics = {}
    for method in ['lgn_m1_sd', 'lgn_m2_sd']:
        rmse_list = [r['rmse'] for r in results[method]]
        nrmse_list = [r['nrmse'] for r in results[method]]
        
        metrics[method] = {
            'rmse_mean': float(np.mean(rmse_list)),
            'rmse_std': float(np.std(rmse_list)),
            'nrmse_mean': float(np.mean(nrmse_list)),
            'nrmse_std': float(np.std(nrmse_list)),
            'n_params': 153
        }
        
        print(f"\n{method.upper()}:", flush=True)
        print(f"  RMSE:  {metrics[method]['rmse_mean']:.6f} ± {metrics[method]['rmse_std']:.6f}", flush=True)
        print(f"  NRMSE: {metrics[method]['nrmse_mean']*100:.2f}% ± {metrics[method]['nrmse_std']*100:.2f}%", flush=True)
    
    # Commutator improvement
    if metrics['lgn_m2_sd']['rmse_mean'] < metrics['lgn_m1_sd']['rmse_mean']:
        ratio = metrics['lgn_m1_sd']['rmse_mean'] / metrics['lgn_m2_sd']['rmse_mean']
        print(f"\n✓ Commutator gives {ratio:.2f}× improvement", flush=True)
    else:
        print(f"\n✗ Commutator did not help", flush=True)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ All results saved to {output_dir}", flush=True)
    
    return metrics


if __name__ == "__main__":
    main()
