"""Print the R² of every model and lead time as a Markdown table (from the saved evaluation CSVs).

    python scripts/forecast_table.py
"""
import pandas as pd

LEADS = [("Nowcast", ""), ("Day 1 (+24 h)", "_h24"), ("Day 2 (+48 h)", "_h48"), ("Day 3 (+72 h)", "_h72")]
MODELS = [("LSTM", "lstm"), ("Neural ODE", "neural_ode"), ("PINN", "pinn")]

rows = ["| Lead time | Model | CO | NOx | NO₂ |", "|---|---|---|---|---|"]
for lead, sfx in LEADS:
    for label, key in MODELS:
        r2 = pd.read_csv(f"evaluation/{key}/{key}_metrics{sfx}.csv").set_index("Parameter")["R2"]
        rows.append(f"| {lead} | {label} | {r2['CO']:.2f} | {r2['NOx']:.2f} | {r2['NO2']:.2f} |")
print("\n".join(rows))
