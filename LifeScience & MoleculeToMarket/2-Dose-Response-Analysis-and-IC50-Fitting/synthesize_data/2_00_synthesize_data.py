"""
video2_00_synthesize_data.py
============================
Generates five synthetic dose-response datasets for the IC50 fitting notebook.
Run this script ONCE to create the CSV files in the ./data/ folder.

Five compounds are created:
  A - clean sigmoid (ground truth for fitting)
  B - noisy sigmoid (realistic lab scatter)
  C - hook effect at high concentration (cytotoxicity artifact)
  D - incomplete curve (IC50 outside tested range)
  E - inactive compound (flat line, no inhibition)

Usage:
  python video2_00_synthesize_data.py

Output:
  data/compound_A_clean.csv
  data/compound_B_noisy.csv
  data/compound_C_hook.csv
  data/compound_D_incomplete.csv
  data/compound_E_inactive.csv
"""

import numpy as np
import pandas as pd
import os

# ---- reproducibility ----
np.random.seed(42)
print("=" * 60)
print("Synthesizing dose-response data for Video 2")
print("=" * 60)

# ---- output directory ----
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"\nOutput directory: {OUT_DIR}")

# ---- 4-parameter logistic model ----
def four_pl(conc, bottom, top, ic50, hill):
    """4PL dose-response model: response = bottom + (top-bottom)/(1+(ic50/conc)^hill)"""
    return bottom + (top - bottom) / (1.0 + (ic50 / conc) ** hill)

# ---- concentration range (10 points per decade, 4 decades) ----
CONC = np.logspace(-2, 2, 10)   # 0.01 to 100 µM, 10 points

print(f"\nConcentration range: {CONC.min():.3f} to {CONC.max():.0f} µM ({len(CONC)} points)")

# ============================================================
# Compound A — clean sigmoid (ideal reference curve)
# ============================================================
print("\n[A] Compound A: clean sigmoid")
print("    Parameters: IC50=1.0 µM, Hill=1.5, Top=100%, Bottom=0%")
resp_A = four_pl(CONC, bottom=0, top=100, ic50=1.0, hill=1.5)
resp_A += np.random.normal(0, 1.5, size=len(CONC))   # very small noise
resp_A = np.clip(resp_A, -5, 105)
df_A = pd.DataFrame({"concentration_uM": CONC, "response_pct": resp_A})
df_A.to_csv(os.path.join(OUT_DIR, "compound_A_clean.csv"), index=False)
print(f"    Saved {len(df_A)} rows → compound_A_clean.csv")

# ============================================================
# Compound B — noisy sigmoid (realistic lab variability)
# ============================================================
print("\n[B] Compound B: noisy sigmoid")
print("    Parameters: IC50=1.0 µM, Hill=1.5, noise=8%")
resp_B = four_pl(CONC, bottom=0, top=100, ic50=1.0, hill=1.5)
resp_B += np.random.normal(0, 8.0, size=len(CONC))   # large realistic noise
resp_B = np.clip(resp_B, -15, 115)
df_B = pd.DataFrame({"concentration_uM": CONC, "response_pct": resp_B})
df_B.to_csv(os.path.join(OUT_DIR, "compound_B_noisy.csv"), index=False)
print(f"    Saved {len(df_B)} rows → compound_B_noisy.csv")

# ============================================================
# Compound C — hook effect (drop at high concentration)
# ============================================================
print("\n[C] Compound C: hook effect")
print("    IC50=0.5 µM, then cytotoxicity dip above 10 µM")
resp_C = four_pl(CONC, bottom=0, top=100, ic50=0.5, hill=1.5)
# Hook: add a cytotoxicity dip at concentrations above 10 µM
hook_mask = CONC > 10
resp_C[hook_mask] += (CONC[hook_mask] - 10) * -3   # artificial dip
resp_C += np.random.normal(0, 2.0, size=len(CONC))
df_C = pd.DataFrame({"concentration_uM": CONC, "response_pct": resp_C})
df_C.to_csv(os.path.join(OUT_DIR, "compound_C_hook.csv"), index=False)
print(f"    Saved {len(df_C)} rows → compound_C_hook.csv")

# ============================================================
# Compound D — incomplete curve (IC50 beyond tested range)
# ============================================================
print("\n[D] Compound D: incomplete curve")
print("    IC50=50 µM (way above tested range of 0.01-100 µM)")
resp_D = four_pl(CONC, bottom=0, top=100, ic50=50.0, hill=1.5)
resp_D += np.random.normal(0, 3.0, size=len(CONC))
resp_D = np.clip(resp_D, -5, 110)
df_D = pd.DataFrame({"concentration_uM": CONC, "response_pct": resp_D})
df_D.to_csv(os.path.join(OUT_DIR, "compound_D_incomplete.csv"), index=False)
print(f"    Saved {len(df_D)} rows → compound_D_incomplete.csv")

# ============================================================
# Compound E — inactive (flat line, no effect)
# ============================================================
print("\n[E] Compound E: inactive compound")
print("    No dose-response relationship, all responses ~100%")
resp_E = np.ones(len(CONC)) * 98.0
resp_E += np.random.normal(0, 4.0, size=len(CONC))
resp_E = np.clip(resp_E, 85, 115)
df_E = pd.DataFrame({"concentration_uM": CONC, "response_pct": resp_E})
df_E.to_csv(os.path.join(OUT_DIR, "compound_E_inactive.csv"), index=False)
print(f"    Saved {len(df_E)} rows → compound_E_inactive.csv")

print("\n" + "=" * 60)
print("All 5 synthetic datasets generated successfully!")
print("Next: open video2_01_dose_response_ic50.ipynb")
print("=" * 60)
