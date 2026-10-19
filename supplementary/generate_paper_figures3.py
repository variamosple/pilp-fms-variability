import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("generated/benchmark_results_summary.csv")

# -------------------------------------------------------------------------
# FIGURA H3: Degradación por Ruido y Observaciones Faltantes
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

h3_df = df[df["hypothesis"] == "H3"].copy()
# Filtrar condiciones a n=100
h3_n100 = h3_df[h3_df["sample_size"] == 100].sort_values(by=["noise_level", "missingness_rate"])

# Caso base: n=100, noise=0, miss=0
base_row = df[(df["hypothesis"] == "H2") & (df["sample_size"] == 100) & (df["method"] == "PILP_SLIPCOVER_Structure")].iloc[0]

conditions = ["Baseline (0%)", "Noise 5%", "Noise 10%", "Missing 15%"]
brier_vals = [base_row["brier_mean"], 
              h3_df[(h3_df["noise_level"] == 0.05) & (h3_df["sample_size"] == 100)]["brier_mean"].values[0],
              h3_df[(h3_df["noise_level"] == 0.10) & (h3_df["sample_size"] == 100)]["brier_mean"].values[0],
              h3_df[(h3_df["missingness_rate"] == 0.15) & (h3_df["sample_size"] == 100)]["brier_mean"].values[0]]

brier_err = [base_row["brier_sd"] * 1.96 / np.sqrt(30),
             h3_df[(h3_df["noise_level"] == 0.05) & (h3_df["sample_size"] == 100)]["brier_sd"].values[0] * 1.96 / np.sqrt(30),
             h3_df[(h3_df["noise_level"] == 0.10) & (h3_df["sample_size"] == 100)]["brier_sd"].values[0] * 1.96 / np.sqrt(30),
             h3_df[(h3_df["missingness_rate"] == 0.15) & (h3_df["sample_size"] == 100)]["brier_sd"].values[0] * 1.96 / np.sqrt(30)]

f1_vals = [base_row["clause_f1_mean"],
           h3_df[(h3_df["noise_level"] == 0.05) & (h3_df["sample_size"] == 100)]["clause_f1_mean"].values[0],
           h3_df[(h3_df["noise_level"] == 0.10) & (h3_df["sample_size"] == 100)]["clause_f1_mean"].values[0],
           h3_df[(h3_df["missingness_rate"] == 0.15) & (h3_df["sample_size"] == 100)]["clause_f1_mean"].values[0]]

x = np.arange(len(conditions))

# Panel 1: Brier score con 95% CI
ax1.bar(x, brier_vals, yerr=brier_err, capsize=5, color="#3B82F6", edgecolor="#1D4ED8", alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(conditions, fontsize=9)
ax1.set_ylabel("Empirical Brier Score (Lower is better)", fontsize=9.5)
ax1.set_title("(a) Predictive Error under Perturbations", fontsize=10.5, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Panel 2: F1 score
ax2.plot(x, f1_vals, marker="o", color="#EF4444", lw=2, markersize=7)
for i, txt in enumerate(f1_vals):
    ax2.annotate(f"{txt:.4f}", (x[i], f1_vals[i] + 0.015), ha="center", fontsize=8.5)
ax2.set_xticks(x)
ax2.set_xticklabels(conditions, fontsize=9)
ax2.set_ylabel("Rule-Signature F1 Score (Higher is better)", fontsize=9.5)
ax2.set_ylim(0.1, 0.5)
ax2.set_title("(b) Structural Clause Recovery", fontsize=10.5, fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig("figures/fig_h3_robustness.pdf", bbox_inches="tight")
fig.savefig("figures/fig_h3_robustness.png", bbox_inches="tight")
plt.close(fig)
print("[OK] Generada figura H3: figures/fig_h3_robustness.pdf")


# -------------------------------------------------------------------------
# FIGURA H4: Escalabilidad Computacional (Runtime vs N)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)

h4_df = df[df["hypothesis"] == "H4"].sort_values(by="sample_size")
sizes = h4_df["sample_size"].values
runtimes = h4_df["runtime_sec_mean"].values

ax.plot(sizes, runtimes, marker="s", color="#8B5CF6", lw=2.2, markersize=8, label="SLIPCOVER runtime")
for s, rt in zip(sizes, runtimes):
    ax.annotate(f"{rt:.2f}s", (s, rt + 0.3), ha="center", fontsize=8.5)

ax.set_xlabel("Training Runs per Included Product (n)", fontsize=9.5)
ax.set_ylabel("Mean Subprocess Execution Time (seconds)", fontsize=9.5)
ax.set_title("H4: Computational Scalability across Training Sizes", fontsize=10.5, fontweight="bold")
ax.set_xticks(sizes)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_ylim(2, 10)

plt.tight_layout()
fig.savefig("figures/fig_h4_scalability.pdf", bbox_inches="tight")
fig.savefig("figures/fig_h4_scalability.png", bbox_inches="tight")
plt.close(fig)
print("[OK] Generada figura H4: figures/fig_h4_scalability.pdf")


# -------------------------------------------------------------------------
# FIGURA H5: Soporte a Decisiones de Mantenimiento
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

h5_df = df[df["hypothesis"] == "H5"].sort_values(by="method")
methods = ["BetaBinomial", "ParameterLPAD", "SLIPCOVER", "UnmaskedLPAD"]
agrees = [h5_df[h5_df["method"] == "BetaBinomial_Product"]["maint_top1_agree_rate"].values[0] * 100,
          h5_df[h5_df["method"] == "ParameterOnly_LPAD"]["maint_top1_agree_rate"].values[0] * 100,
          h5_df[h5_df["method"] == "PILP_SLIPCOVER_Structure"]["maint_top1_agree_rate"].values[0] * 100,
          h5_df[h5_df["method"] == "PILP_Unmasked_Ablation"]["maint_top1_agree_rate"].values[0] * 100]

regrets = [h5_df[h5_df["method"] == "BetaBinomial_Product"]["maint_regret_mean"].values[0] * 1000,
           h5_df[h5_df["method"] == "ParameterOnly_LPAD"]["maint_regret_mean"].values[0] * 100,
           h5_df[h5_df["method"] == "PILP_SLIPCOVER_Structure"]["maint_regret_mean"].values[0] * 1000,
           h5_df[h5_df["method"] == "PILP_Unmasked_Ablation"]["maint_regret_mean"].values[0] * 1000]

x = np.arange(len(methods))

# Panel 1: Top-1 Agreement Rate (%)
bars1 = ax1.bar(x, agrees, color="#10B981", edgecolor="#047857", width=0.55)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=9)
ax1.set_ylabel("Top-1 Station Agreement Rate (%)", fontsize=9.5)
ax1.set_title("(a) Maintenance Agreement with Oracle", fontsize=10.5, fontweight="bold")
ax1.set_ylim(0, 100)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", fontsize=8.5)

# Panel 2: Regret (x10^-3)
bars2 = ax2.bar(x, regrets, color="#F59E0B", edgecolor="#B45309", width=0.55)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=9)
ax2.set_ylabel(r"Expected Regret ($\times 10^{-3}$)", fontsize=9.5)
ax2.set_title("(b) Expected Decision Regret", fontsize=10.5, fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.5)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}", ha="center", fontsize=8.5)

plt.tight_layout()
fig.savefig("figures/fig_h5_maintenance.pdf", bbox_inches="tight")
fig.savefig("figures/fig_h5_maintenance.png", bbox_inches="tight")
plt.close(fig)
print("[OK] Generada figura H5: figures/fig_h5_maintenance.pdf")