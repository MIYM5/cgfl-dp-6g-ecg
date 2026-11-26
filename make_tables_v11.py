"""LaTeX tables and figure data for the version 1.1 experiments.

Usage:  python make_tables_v11.py analysis_v11.json results <out_dir>
Writes <out_dir>/tables/{protocol,recent,remedies,csi,e2e,health,cinc20}.tex and <out_dir>/figs/data/maskq.csv.
All numbers come from analysis_v11.json, except the per-institution CinC-2021 AUCs, read from the run files.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

AJ, RES, OUT = sys.argv[1:4]
sys.argv = sys.argv[:1]
import cgfl  # noqa: E402

A = json.load(open(AJ))
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figs", "data"), exist_ok=True)


def w(name, s):
    open(os.path.join(OUT, "tables", name), "w").write(s)


def pm(v, k=3):
    return f"{v['mean']:.{k}f}$\\pm${v['sd']:.{k}f}"


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def sgn(x, k=3):
    x = 0.0 if abs(x) < 0.5 * 10 ** -k else x
    return f"${x:+.{k}f}$".replace("+0.000", "0.000")


# ---------------------------------------------------------------- 20-seed protocol
P = A["protocol"]
rows = []
for st, r in P.items():
    rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{st}}}}}\\\\")
    for m in ("DP-FedAvg", "Fixed ratio", "Capacity map", "Noise-aware"):
        cmp = {"Capacity map": r["cap_vs_fixed"], "Noise-aware": r["na_vs_fixed"]}.get(m)
        if cmp:
            ci = f"{sgn(cmp['diff'])} [{sgn(cmp['ci_t'][0])}, {sgn(cmp['ci_t'][1])}]"
            ni = pv(cmp["p_ni_holm"]) + (" \\checkmark" if cmp["p_ni_holm"] < 0.05 else "")
        else:
            ci, ni = "--", "--"
        name = "\\textbf{Noise-aware}" if m == "Noise-aware" else m
        rows.append(f"{name} & {pm(r[m])} & {r[m]['mb']['mean']:.0f} & {r[m]['uplink']['mean']:.1f} & {ci} & {ni} \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
w("protocol.tex", r"""\begin{table*}[!htbp]\centering
\caption{Main comparison with 20 seeds under record-level DP ($\varepsilon=3$). $\Delta$: paired difference in test macro-AUC against the fixed ratio, with its 95\% $t$ interval. NI: Holm-adjusted $p$-value of the one-sided paired $t$-test of non-inferiority, $H_0\colon\Delta\le-0.01$ (\checkmark: non-inferior at level $0.05$; the Wilcoxon signed-rank test agrees in every case). All sparsified methods send the same data volume.}\label{tab:protocol}
\setlength{\tabcolsep}{4pt}\begin{tabular}{lccccc}\toprule
Method & Test AUC & Data (MB) & Uplink (s) & $\Delta$ vs.\ fixed [95\% CI] & NI $p$\\\midrule
""".replace("{8}", "{6}") + "\n".join(rows).replace("{8}", "{6}") + "\n\\end{tabular}\\end{table*}\n")

# ---------------------------------------------------------------- recent baselines
B = A["baselines"]
lab = {"default": "PTB-XL, default", "tuned": "PTB-XL, tuned ($\\tau=5$)", "harsh": "PTB-XL, blockage + interference",
       "CinC-2021": "CinC~2021"}
rows = []
for st, r in B.items():
    rows.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{lab[st]}}}}}\\\\")
    for m, v in r.items():
        name = "\\textbf{Noise-aware}" if m == "Noise-aware" else m.replace("et al.", "\\emph{et al.}")
        n = f"$^{{\\dagger}}$" if v["n"] != 5 else ""
        up = f"{v['uplink']['mean']:,.0f}".replace(",", "{,}") if v["uplink"]["mean"] > 1000 else f"{v['uplink']['mean']:.1f}"
        rows.append(f"{name}{n} & {pm(v)} & {v['mb']['mean']:.0f} & {up} \\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
w("recent.tex", r"""\begin{table}[!htbp]\centering
\caption{Comparison with two recent methods adapted to the same record-level DP setting ($\varepsilon=3$) and communication budget (5 seeds; $^\dagger$3 seeds). Wei \emph{et al.}~\cite{wei2024gradient}: per-client drift-plus-penalty retention with a random mask fixed before training and a 1-bit-per-parameter mask; Cheng \emph{et al.}~\cite{cheng2022differentially}: fixed ratio, Top-$k$ after noise with the BLUR regulariser. Hyperparameters of both selected on validation AUC.}\label{tab:recent}
\setlength{\tabcolsep}{4pt}{\small\begin{tabular}{lccc}\toprule
Method & Test AUC & Data (MB) & Uplink (s)\\\midrule
""" + "\n".join(rows) + "\n\\end{tabular}}\\end{table}\n")

# ---------------------------------------------------------------- remedies
R = A["remedies"]
order = ("Fixed ratio", "Fixed ratio + momentum", "Capacity map", "Damped normalisation", "Capacity + momentum",
         "Size-aware floor", "Noise-aware")
hdr = {"default": "Default", "harsh": "Blockage + interf.", "N100": "$N=100$"}
rows = []
for m in order:
    cells = []
    for st in ("default", "harsh", "N100"):
        v = R[st][m]
        g = f" ({v['gain']:+.3f})" if "gain" in v else ""
        cells.append(f"{v['mean']:.3f}{g}")
    name = "\\textbf{Noise-aware}" if m == "Noise-aware" else m
    rows.append(f"{name} & " + " & ".join(cells) + " \\\\" + ("\\midrule" if m in ("Fixed ratio + momentum", "Capacity map") else ""))
w("remedies.tex", r"""\begin{table}[!htbp]\centering
\caption{Remedies for the accuracy cost of the capacity map under record-level DP (test macro-AUC, 3 seeds; in parentheses, the change relative to the plain capacity map). Damped normalisation uses $\mu=0.5$, server momentum $\beta=0.9$ with server learning rate $0.1$, both selected on validation AUC. All variants send the same data volume.}\label{tab:remedies}
\setlength{\tabcolsep}{3pt}{\small\begin{tabular}{lccc}\toprule
Variant & """ + " & ".join(hdr.values()) + r"""\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")

# ---------------------------------------------------------------- CSI / HARQ
C = A["csi"]
rows = []
for e in ("0.0", "0.05", "0.1"):
    f, c = C[f"{e}/fixed"], C[f"{e}/capacity"]
    sv = 100 * (1 - c["uplink"]["mean"] / f["uplink"]["mean"])
    lbl = "perfect" if e == "0.0" else f"$e^2={e}$"
    rows.append(f"{lbl} & {pm(f)} & {pm(c)} & {f['uplink']['mean']:.1f} & {c['uplink']['mean']:.1f} & {sv:.0f}\\% & "
                f"{100 * f['retx'] / f['sent']:.1f}\\% & {100 * f['dropped'] / f['sent']:.1f}\\% \\\\")
w("csi.tex", r"""\begin{table}[!htbp]\centering
\caption{Imperfect CSI with HARQ (5 seeds). The rate is chosen from an MMSE channel estimate with error variance $e^2$ and a $1$\,dB back-off; an update sent above the true rate is retransmitted on fresh fading (at most $3$ times) and otherwise dropped. Retx./dropped: share of the $2{,}000$ scheduled updates, equal for both policies.}\label{tab:csi}
\setlength{\tabcolsep}{2.5pt}{\small\begin{tabular}{lccccccc}\toprule
CSI & \multicolumn{2}{c}{Test AUC} & \multicolumn{2}{c}{Uplink (s)} & Saving & Retx. & Dropped\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5} & Fixed & Capacity & Fixed & Capacity & & & \\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")

# ---------------------------------------------------------------- end-to-end and bandwidth sweep
E = A["e2e"]
rows = []
for W, r in E.items():
    f, c, n, d = r["Fixed ratio"], r["Capacity map"], r["Noise-aware"], r["DP-FedAvg"]
    rows.append(f"{W.replace(' MHz', '')} & {d['total']:.0f} & {f['total']:.0f} & {c['total']:.0f} & {n['total']:.0f} & "
                f"{100 * n['saving_uplink']:.0f}\\% & {100 * n['saving_total']:.0f}\\% \\\\")
dl = list(E.values())[0]["Fixed ratio"]["downlink"]
cp = list(E.values())[0]["Fixed ratio"]["comp"]
w("e2e.tex", r"""\begin{table}[!htbp]\centering
\caption{End-to-end time over 200 rounds (s, mean of 20 seeds, record-level DP) versus the per-hospital uplink sub-band $W_{\mathrm u}$, at the same transmit power. Each total adds the local computation (""" + f"{cp:.0f}" + r"""\,s; $0.38$\,s per round on a T4 GPU) and the downlink broadcast of the dense model (""" + f"{dl:.1f}" + r"""\,s; $40$\,dBm over $100$\,MHz at the weakest scheduled rate). Savings of the noise-aware rule relative to the fixed ratio.}\label{tab:e2e}
\setlength{\tabcolsep}{3.5pt}{\small\begin{tabular}{ccccccc}\toprule
$W_{\mathrm u}$ (MHz) & DP-FedAvg & Fixed & Capacity & Noise-aware & Uplink saving & Total saving\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")

# ---------------------------------------------------------------- healthcare-facing metrics (44 labels)
H = A["health"]
rows = []
for m in ("DP-FedAvg", "Fixed ratio", "Capacity map", "Noise-aware"):
    v, h = H[m], H["hospital"][m]
    name = "\\textbf{Noise-aware}" if m == "Noise-aware" else m
    rows.append(f"{name} & {pm(v['auc'])} & {v['rare']['mean']:.3f} & {v['frequent']['mean']:.3f} & "
                f"{v['f1_05_same']['mean']:.3f} & {v['f1_tuned']['mean']:.3f} & {v['n_tuned']} & "
                f"{h['worst']:.3f} & {h['spread']:.3f} \\\\")
w("health.tex", r"""\begin{table*}[!htbp]\centering
\caption{Clinically oriented metrics under record-level DP ($\varepsilon=3$). Left: fine-grained 44-statement task (20 seeds); rare: the """ + str(H["n_rare"]) + r""" statements with fewer than $100$ training records. Macro-F1 at threshold $0.5$ and at per-label thresholds chosen on the validation set (on the $n$ seeds with saved probabilities). Right: single-label task, test AUC re-weighted to the label mix of each hospital; worst hospital and best--worst spread, averaged over seeds.}\label{tab:health}
\setlength{\tabcolsep}{4pt}\begin{tabular}{lcccccc|cc}\toprule
& \multicolumn{6}{c|}{44 diagnostic statements} & \multicolumn{2}{c}{Per-hospital AUC}\\
\cmidrule(lr){2-7}\cmidrule(lr){8-9}
Method & Macro-AUC & Rare & Frequent & F1@0.5 & F1, tuned & $n$ & Worst & Spread\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

# ---------------------------------------------------------------- CinC-2021, patient-wise split, 20 seeds, per institution
BASE = vars(cgfl.defaults().parse_args(["--lr_dp", "0.5"]))
DP = dict(dp="record", mask="public")
NA = {"method": "noiseaware", **A["tune"]["noiseaware"]["pick"], **DP}


def get(o):
    f = os.path.join(RES, cgfl.run_id(argparse.Namespace(**{**BASE, **o}))[0] + ".json")
    return json.load(open(f)) if os.path.exists(f) else None


SITES = ("CPSC", "CPSC-Extra", "Chapman", "Georgia", "Ningbo", "PTB-XL")
rows, nk = [], None
for m, o in (("DP-FedAvg", dict(method="fedavg", dp="record")), ("Fixed ratio", dict(method="fixed", **DP)),
             ("Capacity map", dict(method="capacity", **DP)), ("Noise-aware", NA)):
    Rr = [r for r in (get({**o, **cgfl.CINC, "seed": s}) for s in range(1, 21)) if r]
    nk = Rr[0]["test"]["site_nk"]
    site = [np.mean([r["test"]["site_auc"][s] for r in Rr]) for s in SITES]
    v = dict(mean=np.mean([r["test"]["auc"] for r in Rr]), sd=np.std([r["test"]["auc"] for r in Rr], ddof=1))
    name = "\\textbf{Noise-aware}" if m == "Noise-aware" else m
    rows.append(f"{name} & {pm(v)} & {np.mean([r['wallclock_uplink_s'] for r in Rr]):.1f} & "
                + " & ".join(f"{x:.3f}" for x in site) + " \\\\")
sig = [r for r in (get({**NA, **cgfl.CINC, "seed": s}) for s in range(1, 21)) if r]
hdr = " & ".join(f"{s.replace('Chapman', 'Chap.').replace('CPSC-Extra', 'Extra').replace('Georgia', 'Georg.')} ({nk[s]})"
                 for s in SITES)
w("cinc20.tex", r"""\begin{table*}[!htbp]\centering
\caption{PhysioNet/CinC~2021 with six real institutions as clients, patient-wise split of the PTB-XL institution (record-level DP, $\varepsilon=3$, 20 seeds; 19 diagnoses, macro-AUC over the diagnoses each institution annotates). Right: test AUC per institution (training records in parentheses).}\label{tab:cinc}
\setlength{\tabcolsep}{3.2pt}\resizebox{\textwidth}{!}{\begin{tabular}{lcc|cccccc}\toprule
Method & Test AUC & Uplink (s) & """ + hdr + r"""\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table*}\n")

# ---------------------------------------------------------------- mask quality curve
M = A["maskq"]
with open(os.path.join(OUT, "figs", "data", "maskq.csv"), "w") as fh:
    fh.write("s,pub_default,rnd_default,pub_tuned,rnd_tuned\n")
    for s in M["default"]:
        fh.write(f"{s},{M['default'][s]['public']:.4f},{M['default'][s]['random']:.4f},"
                 f"{M['tuned'][s]['public']:.4f},{M['tuned'][s]['random']:.4f}\n")
print("tables and maskq.csv written to", OUT)
