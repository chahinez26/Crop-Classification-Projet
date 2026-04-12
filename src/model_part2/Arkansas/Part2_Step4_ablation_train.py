"""
PART 2 — ÉTAPE 4 : Entraînement des 5 configurations d'ablation
================================================================
C1 : S2 seulement          → MCTNet baseline
C2 : S2 + Climat  (3 cov)  → temp_mean, precip_total, solar_mean
C3 : S2 + Sol     (3 cov)  → soil_ph, soil_oc, soil_texture
C4 : S2 + Topo    (2 cov)  → elevation, landforms
C5 : S2 + Tout    (8 cov)  → toutes covariables
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os, time, json

from step5_mctnet import count_parameters
from Part2_Step3_mctnetcov import build_model

# ── Chemins ────────────────────────────────────────────────────────────────
COV_FILE    = r"data\merged_npz\arkansas\ARK_covariates.npz"
MODEL_DIR   = r"Coutputs\results\Arkansas\part2\models"
RESULTS_DIR = r"outputs\results\Arkansas\part2\results"
FIG_DIR     = r"outputs\figures\Arkansas\part2"

BATCH_SIZE = 32
EPOCHS     = 200
LR         = 0.001
PATIENCE   = 40
D_COV      = 16
SEED       = 42

CONFIGS = {
    'C1_S2only'  : {'n_cov': 0, 'cov_keys': [],
                    'label': 'S2 only (baseline)', 'color': '#7f7f7f'},
    'C2_Climate' : {'n_cov': 3, 'cov_keys': ['cov_clim'],
                    'label': 'S2 + Climat',         'color': '#1f77b4'},
    'C3_Soil'    : {'n_cov': 3, 'cov_keys': ['cov_soil'],
                    'label': 'S2 + Sol',             'color': '#8c564b'},
    'C4_Topo'    : {'n_cov': 2, 'cov_keys': ['cov_topo'],
                    'label': 'S2 + Topo',            'color': '#bcbd22'},
    'C5_All'     : {'n_cov': 8, 'cov_keys': ['cov_clim','cov_soil','cov_topo'],
                    'label': 'S2 + Tout',            'color': '#e377c2'},
}

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,     exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)


def load_data():
    d = np.load(COV_FILE, allow_pickle=True)
    def t(a, dtype=torch.float32):
        return torch.tensor(np.array(a), dtype=dtype)
    splits = {}
    for sp in ['train', 'val', 'test']:
        idx = d[f'{sp}_idx']
        splits[sp] = {
            'X'       : t(d['X_all'][idx]),
            'mask'    : t(d['mask_all'][idx]),
            'y'       : t(d['y_all'][idx], dtype=torch.long),
            'cov_clim': t(d['cov_clim'][idx]),
            'cov_soil': t(d['cov_soil'][idx]),
            'cov_topo': t(d['cov_topo'][idx]),
        }
    print(f"  train={len(splits['train']['y'])}  "
          f"val={len(splits['val']['y'])}  "
          f"test={len(splits['test']['y'])}")
    return splits


def get_cov(sp_data, cov_keys):
    if not cov_keys:
        return None
    parts = [sp_data[k] for k in cov_keys]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def make_loaders(splits, cfg):
    loaders = {}
    for sp, data in splits.items():
        cov   = get_cov(data, cfg['cov_keys'])
        dummy = torch.zeros(len(data['y']), 1)
        ds = TensorDataset(data['X'], data['mask'], data['y'],
                           cov if cov is not None else dummy)
        loaders[sp] = DataLoader(
            ds, batch_size=BATCH_SIZE if sp == 'train' else 256,
            shuffle=(sp == 'train'), num_workers=0)
    return loaders


def run_epoch(model, loader, criterion, optimizer, device, use_cov, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_ok = tot_n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, m, y, cov in loader:
            X, m, y = X.to(device), m.to(device), y.to(device)
            if train: optimizer.zero_grad()
            out  = model(X, m, cov.to(device)) if use_cov else model(X, m)
            loss = criterion(out, y)
            if train: loss.backward(); optimizer.step()
            tot_loss += loss.item() * len(y)
            tot_ok   += (out.argmax(1) == y).sum().item()
            tot_n    += len(y)
    return tot_loss / tot_n, tot_ok / tot_n


def compute_metrics(model, loader, device, use_cov, nc=5):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, m, y, cov in loader:
            out = model(X.to(device), m.to(device),
                        cov.to(device)) if use_cov else model(X.to(device), m.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y.numpy())
    yt, yp = np.array(trues), np.array(preds)
    oa = float((yt == yp).mean())
    N  = len(yt)
    cm = np.zeros((nc, nc), dtype=np.int64)
    for t, p in zip(yt, yp): cm[t, p] += 1
    kappa = (N * np.trace(cm) - (cm.sum(1) * cm.sum(0)).sum()) / \
            (N**2 - (cm.sum(1) * cm.sum(0)).sum())
    f1s = []
    for i in range(nc):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        p  = tp/(tp+fp) if tp+fp > 0 else 0.
        r  = tp/(tp+fn) if tp+fn > 0 else 0.
        f1s.append(2*p*r/(p+r) if p+r > 0 else 0.)
    return {'OA': round(oa, 4), 'Kappa': round(float(kappa), 4),
            'F1': round(float(np.mean(f1s)), 4)}


def train_config(cname, cfg, splits, device):
    print(f"\n{'='*58}\n  {cname} : {cfg['label']}\n{'='*58}")
    loaders  = make_loaders(splits, cfg)
    model    = build_model(cfg['n_cov'], D_COV, device)
    use_cov  = cfg['n_cov'] > 0
    print(f"  Paramètres : {count_parameters(model):,}")

    crit  = nn.CrossEntropyLoss()
    optim_ = optim.Adam(model.parameters(), lr=LR)

    hist  = {'tl':[], 'ta':[], 'vl':[], 'va':[]}
    best_vl = float('inf'); pat = 0; best_ep = 0
    mpath   = os.path.join(MODEL_DIR, f'{cname}.pth')

    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'s':>3}")
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        te = time.time()
        tl, ta = run_epoch(model, loaders['train'], crit, optim_, device, use_cov)
        vl, va = run_epoch(model, loaders['val'],   crit, None,   device, use_cov, False)
        for k, v in zip(['tl','ta','vl','va'], [tl,ta,vl,va]):
            hist[k].append(v)

        mk = ''
        if vl < best_vl:
            best_vl, best_ep, pat = vl, ep, 0
            torch.save({'ep': ep, 'state': model.state_dict(), 'vl': vl, 'va': va,
                        'n_params': count_parameters(model),
                        'cfg': {k: v for k, v in cfg.items() if k != 'color'}}, mpath)
            mk = ' ←'
        else:
            pat += 1

        if ep % 10 == 0 or mk or ep == 1:
            print(f"  {ep:>4}  {tl:8.4f}  {ta*100:6.2f}%  "
                  f"{vl:8.4f}  {va*100:6.2f}%  {time.time()-te:2.1f}s{mk}")
        if pat >= PATIENCE:
            print(f"\n  ⏹ Early stop ep {ep} (best={best_ep})")
            break

    print(f"\n  ⏱ {time.time()-t0:.0f}s | best ep {best_ep} | "
          f"val_acc={hist['va'][best_ep-1]*100:.2f}%")
    return hist, mpath, best_ep


def main():
    print("="*58)
    print("Part 2 — Étape 4 : Ablation (5 configurations)")
    print("="*58 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")
    splits = load_data()

    all_results, all_hist = {}, {}

    for cname, cfg in CONFIGS.items():
        hist, mpath, best_ep = train_config(cname, cfg, splits, device)
        all_hist[cname] = hist

        ckpt  = torch.load(mpath, map_location=device)
        model = build_model(cfg['n_cov'], D_COV, device)
        model.load_state_dict(ckpt['state'])
        loaders = make_loaders(splits, cfg)
        metrics = compute_metrics(model, loaders['test'], device, cfg['n_cov'] > 0)
        all_results[cname] = {'label': cfg['label'], 'color': cfg['color'],
                               'n_params': ckpt['n_params'],
                               'best_epoch': best_ep, **metrics}
        print(f"\n  TEST → OA={metrics['OA']:.4f}  "
              f"Kappa={metrics['Kappa']:.4f}  F1={metrics['F1']:.4f}")

    # Résumé
    print(f"\n\n{'='*65}")
    print("RÉSUMÉ ABLATION — Arkansas (Part 2)")
    print(f"{'='*65}")
    print(f"  {'Config':25s}  {'OA':>7}  {'Kappa':>7}  {'F1':>7}  "
          f"{'ΔOA':>7}  {'Params':>9}")
    print(f"  {'-'*62}")
    base = all_results['C1_S2only']['OA']
    best_k = max(all_results, key=lambda c: all_results[c]['OA'])
    for c, r in all_results.items():
        mk = ' ★' if c == best_k else ''
        print(f"  {r['label']:25s}  {r['OA']:7.4f}  {r['Kappa']:7.4f}  "
              f"{r['F1']:7.4f}  {r['OA']-base:+7.4f}  {r['n_params']:9,}{mk}")
    print(f"  {'Papier MCTNet':25s}  {'0.968':>7}  {'0.951':>7}  {'0.933':>7}")
    print(f"\n  ★ Best : {all_results[best_k]['label']}  "
          f"(Δ={all_results[best_k]['OA']-base:+.4f})")

    # Sauvegarde
    res_f = os.path.join(RESULTS_DIR, 'ablation_results.json')
    with open(res_f, 'w') as f:
        json.dump(all_results, f, indent=2)
    hist_f = os.path.join(RESULTS_DIR, 'ablation_histories.npz')
    np.savez(hist_f, **{f'{c}_{k}': np.array(v)
             for c, h in all_hist.items() for k, v in h.items()})
    print(f"\n✅ {res_f}\n✅ {hist_f}")
    print("\n   → Lancer : python Part2_Step5_ablation_results.py")


if __name__ == "__main__":
    main()
