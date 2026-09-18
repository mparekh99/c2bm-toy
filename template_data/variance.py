import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# CSV_PATH  = "/N/u/mihparek/BigRed200/c2bm-toy/template_data/labeled_episodes_raw.csv"
CSV_PATH = "/N/slate/mihparek/Fail2/evals_1.5/target/output-trails/labels_template_v2/labeled_episodes_raw.csv"
TASK_NODE = "success"

df = pd.read_csv(CSV_PATH).dropna(how="all").dropna(axis=1, how="all")
print(f"Loaded {len(df)} episodes\n")

concept_cols = [c for c in df.columns if c not in ["episode_idx"]]

variance     = df[concept_cols].var()
counts_true  = df[concept_cols].sum()
counts_false = len(df) - counts_true

print("="*60)
print("VARIANCE + CHI-SQUARE vs TASK NODE")
print("="*60)

results = []
for c in concept_cols:
    if c == TASK_NODE:
        continue
    v  = variance[c]
    t  = int(counts_true[c])
    f  = int(counts_false[c])
    ct = pd.crosstab(df[c], df[TASK_NODE])
    if ct.shape == (2, 2):
        _, p, _, _ = chi2_contingency(ct)
    else:
        p = 1.0
    results.append((c, v, t, f, p))

results.sort(key=lambda x: x[1], reverse=True)

for c, v, t, f, p in results:
    var_q = "✅" if v >= 0.1  else "⚠️ " if v >= 0.05 else "❌"
    chi_q = "✅" if p < 0.05 else "⚠️ " if p < 0.1  else "❌"
    print(f"{var_q} var={v:.3f}  {chi_q} p={p:.4f}  (1s={t:2d}, 0s={f:2d})  {c}")

print("\n→ Recommended keep_concepts (var>=0.1 AND p<0.1):")
keep = [c for c, v, t, f, p in results if v >= 0.1 and p < 0.1]
print(keep)