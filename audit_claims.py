import sys, os, json, argparse, numpy as np
from scipy import stats
sys.path.insert(0, sys.argv[1]); import cgfl
RES = sys.argv[2]
BASE = vars(cgfl.defaults().parse_args(["--lr_dp", "0.5"]))
def R(o):
    rid, _ = cgfl.run_id(argparse.Namespace(**{**BASE, **o})); return json.load(open(os.path.join(RES, rid + ".json")))
DP = dict(dp="record", mask="public")
pairs = []
settings = [({}, [1,2,3,4,5])] + [(dict(eps=e, T=T), [1,2,3]) for e in [1,3,8] for T in [100,200,400]] + \
           [(dict(N=50),[1,2,3]),(dict(N=100),[1,2,3]),(dict(alpha_dir=0.1),[1,2,3]),(dict(alpha_dir=1.0),[1,2,3])] + \
           [(dict(scenario=s),[1,2,3]) for s in ["blockage","interf","blockage_interf"]]
for o, seeds in settings:
    for s in seeds:
        f, c = R(dict(method="fixed", **DP, **o, seed=s)), R(dict(method="capacity", **DP, **o, seed=s))
        pairs.append((o, s, f, c))
red = np.array([1 - c["wallclock_uplink_s"] / f["wallclock_uplink_s"] for _, _, f, c in pairs])
dauc = np.array([c["test"]["auc"] - f["test"]["auc"] for _, _, f, c in pairs])
dmb = np.array([c["comm_mb"] / f["comm_mb"] - 1 for _, _, f, c in pairs])
print("pairs:", len(pairs))
print("uplink reduction: all positive?", (red > 0).all(), "| pooled (sum-based) %.1f%%" % (100 * (1 - sum(c["wallclock_uplink_s"] for *_, c in pairs) / sum(f["wallclock_uplink_s"] for _, _, f, _ in pairs))),
      "| mean %.1f%% median %.1f%% range %.0f-%.0f%%" % (100 * red.mean(), 100 * np.median(red), 100 * red.min(), 100 * red.max()))
print("sign test p =", stats.binomtest(int((red > 0).sum()), len(red)).pvalue)
print("data volume ratio cap/fixed - 1: max |.| = %.4f" % np.abs(dmb).max())
ci = stats.t.interval(.95, len(dauc) - 1, loc=dauc.mean(), scale=stats.sem(dauc))
print("AUC diff mean %.4f  95%% CI [%.3f, %.3f]  Wilcoxon p = %.4f" % (dauc.mean(), ci[0], ci[1], stats.wilcoxon(dauc).pvalue))
# saving formula 1 - smin/sbar vs simulation (main setting)
for o, s, f, c in pairs[:5]:
    print("  seed", s, "s_bar %.3f -> predicted %.1f%%, simulated %.1f%%" % (c["s_bar"], 100 * (1 - 0.1 / c["s_bar"]), 100 * (1 - c["wallclock_uplink_s"] / f["wallclock_uplink_s"])))
# sigma per T at eps=3 (text: 4.8, 6.6, 9.3)
for T in [100, 200, 400]:
    print("sigma eps=3 T=%d: %.2f" % (T, np.mean([R(dict(method="capacity", **DP, eps=3, T=T, seed=s))["sigma_mean"] for s in [1,2,3]])))
