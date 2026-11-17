#!/usr/bin/env python3
"""Compare freshly re-run results with the archived ones (same run IDs).

Usage: python repro_check.py <archived results dir> <re-run results dir>
GPU kernels are not bit-wise deterministic, so AUC is compared with a tolerance; privacy (sigma),
data volume and uplink time are deterministic given the seed and must match.
Archived Top-k runs stored 32-bit indices; they are rescaled to ceil(log2 d)-bit indices as in the paper.
"""
import sys, os, json, glob, math

OLD, NEW = sys.argv[1:3]
TOL_AUC = 0.02
bad = 0
for f in sorted(glob.glob(os.path.join(NEW, "*.json"))):
    rid = os.path.basename(f)
    g = os.path.join(OLD, rid)
    if not os.path.exists(g):
        print(f"{rid}: no archived counterpart"); continue
    n, o = json.load(open(f)), json.load(open(g))
    c = o["cfg"]
    fac = 1.0
    if c.get("mask") == "topk" and c.get("method") not in ("fedavg", "qsgd", "qsgd_cap"):
        fac = (32 + math.ceil(math.log2(o["d"]))) / 64.0
    checks = [("test AUC", n["test"]["auc"], o["test"]["auc"], TOL_AUC),
              ("sigma", n["sigma_mean"], o["sigma_mean"], 1e-6),
              ("data MB", n["comm_mb"], fac * o["comm_mb"], 1e-3 * max(1, o["comm_mb"])),
              ("uplink s", n["wallclock_uplink_s"], fac * o["wallclock_uplink_s"], 1e-3 * max(1, o["wallclock_uplink_s"]))]
    for name, a, b, tol in checks:
        ok = abs(a - b) <= tol
        bad += not ok
        print(f"{rid[:-5]} {c.get('method'):9s} dp={c.get('dp'):6s} seed={c.get('seed')}  {name:9s} "
              f"re-run={a:10.4f}  archived={b:10.4f}  {'OK' if ok else 'DIFFERS'}")
print("\nall checks passed" if bad == 0 else f"\n{bad} check(s) differ")
