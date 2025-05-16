python3 - << 'PYCODE'
import glob, json

entries = []
for fn in glob.glob("sweep_summary_*.json"):
    d = json.load(open(fn))
    entries.append((d["exp_name"], d["mean_auc"]))

# sort descending by mean_auc
for exp, auc in sorted(entries, key=lambda x: x[1], reverse=True):
    print(f"{auc:.3f}\t{exp}")
PYCODE