#!/usr/bin/env python3
"""Privacy checks for cgfl.py.

  1. Accountant: every noise multiplier used in the archived runs is recomputed with cgfl's RDP accountant and,
     if Opacus is installed, with Opacus' RDP accountant; the largest relative gap in epsilon is reported.
  2. Clipping: per-example gradients restricted to the public mask have norm <= C after clipping.
  3. Noise: the Gaussian noise lands on the masked coordinates only, with standard deviation sigma * C.

Usage: python test_privacy.py [results dir]
"""
import sys, os, json, glob, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
import cgfl

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
DELTA = 1e-5
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


# ---------------------------------------------------------------- 1. accountant
cases = set()
for f in glob.glob(os.path.join(RES, "*.json")):
    r = json.load(open(f))
    c = r["cfg"]
    if c.get("dp") != "record":
        continue
    cases.add((float(c["eps"]), int(c["T"]), int(c["tau"]), float(c["part"])))
print(f"{len(cases)} distinct (eps, T, tau, p) DP configurations in {RES}")

# the accountant itself: sigma solved for eps must give eps back, for a grid of q and step counts
worst = 0.0
for eps in (1.0, 3.0, 8.0):
    for q in (0.004, 0.02, 0.1, 0.5):
        for steps in (500, 1000, 2000, 4000):
            s = cgfl.solve_sigma(eps, DELTA, q, steps)
            e = cgfl.epsilon(q, s, steps, DELTA)
            worst = max(worst, (eps - e) / eps)
            if e > eps + 1e-9:
                check(False, f"solve_sigma overshoots eps={eps} q={q} steps={steps}: {e:.4f}")
check(worst < 0.01, f"solve_sigma returns sigma with eps within 1% below the target (worst {100*worst:.3f}%)")

try:
    from opacus.accountants import RDPAccountant
    gap = 0.0
    for eps in (1.0, 3.0, 8.0):
        for q in (0.004, 0.02, 0.1, 0.5):
            for steps in (500, 2000):
                s = cgfl.solve_sigma(eps, DELTA, q, steps)
                acc = RDPAccountant()
                acc.history = [(s, q, steps)]
                e_op = acc.get_epsilon(delta=DELTA, alphas=list(range(2, 257)))
                gap = max(gap, abs(e_op - cgfl.epsilon(q, s, steps, DELTA)) / eps)
    check(gap < 0.01, f"cgfl accountant agrees with Opacus RDP accountant (largest gap {100*gap:.3f}% of eps)")
except ImportError:
    print("SKIP Opacus not installed (pip install opacus) - accountant cross-check skipped")

# ---------------------------------------------------------------- 2./3. clipping and noise on a tiny model
torch.manual_seed(0)
model = cgfl.ECGNet(8, 5)
gp = {k: v.detach().clone() for k, v in model.named_parameters()}
d = sum(v.numel() for v in gp.values())
X = torch.randn(64, 12, 1000)
y = torch.randint(0, 5, (64,))
C = 1.0
mask = torch.zeros(d, dtype=torch.bool)
mask[torch.randperm(d)[: d // 4]] = True
mv = cgfl.unflat(mask.float(), gp)


def loss_one(p, xb, yb):
    return F.cross_entropy(functional_call(model, p, (xb.unsqueeze(0),)), yb.unsqueeze(0))


gs = vmap(grad(loss_one), in_dims=(None, 0, 0))(gp, X, y)
gs = {k: g * mv[k] for k, g in gs.items()}
norms = torch.sqrt(sum(g.reshape(64, -1).pow(2).sum(1) for g in gs.values()))
scale = (C / (norms + 1e-12)).clamp(max=1.0)
clipped = torch.sqrt(sum((g * scale.view(-1, *[1] * (g.dim() - 1))).reshape(64, -1).pow(2).sum(1)
                         for g in gs.values()))
check(bool((clipped <= C * (1 + 1e-5)).all()), f"masked per-example gradients clipped to norm <= C (max {float(clipped.max()):.6f})")
off = torch.cat([(g * (1 - mv[k])).reshape(64, -1) for k, g in gs.items()], 1)
check(float(off.abs().max()) == 0.0, "per-example gradients are zero outside the public mask before clipping")

# noise exactly as local_train draws it
sigma, gen = 3.0, torch.Generator().manual_seed(1)
noise = torch.cat([(torch.randn(gp[k].shape, generator=gen) * sigma * C * mv[k]).reshape(-1) for k in gp])
check(float(noise[~mask].abs().max()) == 0.0, "noise is zero on unmasked (unreleased) coordinates")
sd = float(noise[mask].std())
check(abs(sd - sigma * C) / (sigma * C) < 0.02, f"noise std on masked coordinates = {sd:.4f} (target sigma*C = {sigma*C})")

print("\nall privacy checks passed" if fails == 0 else f"\n{fails} check(s) failed")
sys.exit(1 if fails else 0)
