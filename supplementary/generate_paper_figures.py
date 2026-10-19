import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

# Set general academic typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


# FIGURE 1: ALGORITHMIC WORKFLOW & ARTIFACT ARCHITECTURE

fig1, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.2)
ax.axis('off')


def draw_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body_lines,
    facecolor='#F8FAFC',
    edgecolor='#1E293B',
    title_color='#0F172A',
    header_bg=None,
):
  box = FancyBboxPatch(
      (x, y),
      w,
      h,
      boxstyle='round,pad=0.04,rounding_size=0.12',
      facecolor=facecolor,
      edgecolor=edgecolor,
      linewidth=1.4,
      zorder=2,
  )
  ax.add_patch(box)

  if header_bg:
    hh = 0.42
    hbox = FancyBboxPatch(
        (x, y + h - hh),
        w,
        hh,
        boxstyle='round,pad=0.02,rounding_size=0.10',
        facecolor=header_bg,
        edgecolor=edgecolor,
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(hbox)
    ax.text(
        x + w / 2,
        y + h - hh / 2,
        title,
        ha='center',
        va='center',
        fontsize=9.5,
        fontweight='bold',
        color='white',
        zorder=4,
    )
    start_y = y + h - hh - 0.22
  else:
    ax.text(
        x + w / 2,
        y + h - 0.24,
        title,
        ha='center',
        va='center',
        fontsize=9.5,
        fontweight='bold',
        color=title_color,
        zorder=4,
    )
    start_y = y + h - 0.52

  for i, line in enumerate(body_lines):
    ax.text(
        x + w / 2,
        start_y - i * 0.23,
        line,
        ha='center',
        va='center',
        fontsize=8.0,
        color='#1E293B',
        zorder=4,
    )


def draw_arrow(
    ax, x1, y1, x2, y2, label='', color='#334155', style='-|>', lw=1.5, rad=0.0
):
  arrow = FancyArrowPatch(
      (x1, y1),
      (x2, y2),
      connectionstyle=f'arc3,rad={rad}',
      arrowstyle=f'{style},head_length=6,head_width=4',
      color=color,
      linewidth=lw,
      zorder=5,
  )
  ax.add_patch(arrow)
  if label:
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(
        mx,
        my + 0.14,
        label,
        ha='center',
        va='center',
        fontsize=7.8,
        fontweight='bold',
        color=color,
        bbox=dict(
            boxstyle='round,pad=0.15',
            facecolor='white',
            edgecolor='none',
            alpha=0.85,
        ),
        zorder=6,
    )


# Title Banner
ax.text(
    6.0,
    5.92,
    'Algorithm 1: Evidence-Aware PILP Workflow & Semantic-Preserving'
    ' Variability Export',
    ha='center',
    va='center',
    fontsize=11.5,
    fontweight='bold',
    color='#0F172A',
)

# Stage 1: Inputs
draw_box(
    ax,
    0.3,
    3.6,
    2.5,
    1.85,
    '1. Approved PPR Knowledge',
    [
        '• Feature Model H (UVL)',
        '• Product Attributes & Routing',
        '• Station Capabilities C',
        '• Language Bias M (failed/-#op,-#st)',
    ],
    facecolor='#EFF6FF',
    edgecolor='#1D4ED8',
    header_bg='#1D4ED8',
)

draw_box(
    ax,
    0.3,
    1.2,
    2.5,
    1.85,
    '2. FMS Production Logs',
    [
        '• 6-Product Family Runs r',
        '• Station Assignments (o, s, a)',
        '• Explicit Observation Mask m:',
        '  attempted=1 vs. attempted=0',
    ],
    facecolor='#F0FDF4',
    edgecolor='#15803D',
    header_bg='#15803D',
)

# Stage 2: Logical Adapter
draw_box(
    ax,
    3.4,
    2.35,
    2.45,
    2.15,
    '3. Run-Level Log Adapter',
    [
        '• Stratified Event Partitioning',
        '• Hold Out Unseen Product (desk)',
        '• Mask-Aware Conversion:',
        '  failed(o,s) / neg(failed(o,s))',
        '  ONLY when attempted = 1',
        '• :- set_sc(neg_ex, given).',
    ],
    facecolor='#FFFBEB',
    edgecolor='#B45309',
    header_bg='#B45309',
)

# Stage 3: Induction & Reasoning
draw_box(
    ax,
    6.45,
    2.35,
    2.45,
    2.15,
    '4. SLIPCOVER Induction (cplint)',
    [
        '• Bottom-Clause Construction',
        '• Beam Search over Clause Space',
        '• Stratified Likelihood Estimation',
        '• Prune Clauses (p < 0.01) & Freeze',
        '• Exact PITA Noisy-OR Queries',
    ],
    facecolor='#FAF5FF',
    edgecolor='#6D28D9',
    header_bg='#6D28D9',
)

# Stage 4: Three-Part Export Bundle
draw_box(
    ax,
    9.45,
    4.3,
    2.25,
    1.25,
    '5a. Hard Variability Model',
    [
        'variability_model.uvl',
        'Boolean Feature Constraints',
        'Valid(E) = X_H (0% Violation)',
    ],
    facecolor='#EFF6FF',
    edgecolor='#1D4ED8',
    header_bg='#1E40AF',
)

draw_box(
    ax,
    9.45,
    2.8,
    2.25,
    1.25,
    '5b. Process Contract',
    [
        'process_contract.json',
        'Station Routing & Precedence',
        'hard_constraints_modified: false',
    ],
    facecolor='#F1F5F9',
    edgecolor='#475569',
    header_bg='#334155',
)

draw_box(
    ax,
    9.45,
    1.3,
    2.25,
    1.25,
    '5c. Learned Risk Model',
    [
        'learned_risk_model.pl',
        'Event-Indexed LPAD Clauses',
        '+ SHA-256 export_manifest.json',
    ],
    facecolor='#FEF2F2',
    edgecolor='#B91C1C',
    header_bg='#991B1B',
)

# Bottom Governance Bar
draw_box(
    ax,
    3.4,
    0.25,
    5.5,
    1.3,
    '6. Human-in-the-Loop Governance & Decision Support',
    [
        (
            '• Counterfactual Net Benefit Evaluation:  Δ_s = E[L | current] -'
            ' E[L | intervention a_s] - c_s'
        ),
        (
            '• Station Maintenance Ranking & Risk Forecasting on Held-Out'
            ' Product (desk)'
        ),
        (
            '• Conservative Loop: Risk Triggers Engineering Review \nWithout'
            ' Silently Altering Hard UVL Constraints'
        ),
    ],
    facecolor='#F8FAFC',
    edgecolor='#0F172A',
    header_bg='#0F172A',
)

# Arrows
draw_arrow(ax, 2.8, 4.5, 3.4, 3.8, label='B, H, M')
draw_arrow(ax, 2.8, 2.1, 3.4, 3.0, label='Logs D')
draw_arrow(ax, 5.85, 3.42, 6.45, 3.42, label='Models')

draw_arrow(ax, 8.9, 3.9, 9.45, 4.85, label='Preserved H')
draw_arrow(ax, 8.9, 3.42, 9.45, 3.42, label='Routing G_p')
draw_arrow(ax, 8.9, 2.95, 9.45, 1.95, label='LPAD (L, Θ)')

draw_arrow(ax, 7.67, 2.35, 7.67, 1.55, label='PITA Queries')
draw_arrow(ax, 9.45, 1.65, 8.9, 0.9, label='Audit & Approval', rad=-0.15)

plt.tight_layout()
fig1.savefig('fig1_workflow_architecture.pdf', bbox_inches='tight')
fig1.savefig('fig1_workflow_architecture.png', dpi=300, bbox_inches='tight')
plt.close(fig1)


# FIGURE 2: EMPIRICAL BENCHMARK RESULTS (30 SEEDS, H1-H5)

fig2, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=300)

# Panel (a): Grouped Bar Chart of Brier Score (Exact empirical numbers from Table 7)
ax1 = axes[0]
methods = [
    'Bayesian\nNet\n(Joint CPT)',
    'Beta-\nBinomial\n(Product)',
    'Tabular\nLogReg',
    'Process\nMining\n(Control-Flow)',
    'PILP\nUnmasked\n(Ablation)',
    'PILP\n(SLIPCOVER)',
]

# Real values matching Table 7
h1_brier = [0.047751, 0.047851, 0.053793, 0.055899, 0.133573, 0.135997]
h2_brier = [0.169013, 0.169013, 0.160103, 0.169069, 0.312363, 0.308062]

x = np.arange(len(methods))
width = 0.36

rects1 = ax1.bar(
    x - width / 2,
    h1_brier,
    width,
    label='Seen Family (H1, n=100)',
    color='#2563EB',
    edgecolor='#1E3A8A',
    alpha=0.9,
)
rects2 = ax1.bar(
    x + width / 2,
    h2_brier,
    width,
    label='Held-Out Desk Transfer (H2, n=100)',
    color='#EA580C',
    edgecolor='#7C2D12',
    alpha=0.9,
)

ax1.set_ylabel('Empirical Brier Score (Lower is Better)', fontsize=9.5)
ax1.set_title(
    '(a) Empirical Calibration Error: Seen Family (H1) vs. Transfer (H2)',
    fontsize=10.5,
    fontweight='bold',
    pad=10,
)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=8.0)
ax1.set_ylim(0, 0.38)
ax1.grid(axis='y', linestyle='--', alpha=0.45)
ax1.legend(loc='upper left', fontsize=8.5, frameon=True)

# Annotations pointing out trade-offs
ax1.annotate(
    '0.136\n(Inspectable Rules)',
    xy=(5 - width / 2, 0.135997),
    xytext=(5 - width / 2 - 1.7, 0.23),
    ha='center',
    fontsize=7.5,
    fontweight='bold',
    color='#1E3A8A',
    arrowprops=dict(arrowstyle='->', color='#1E3A8A', lw=1.0),
)
ax1.annotate(
    '0.308\n(Stage Confounding)',
    xy=(5 + width / 2, 0.308062),
    xytext=(5 + width / 2 - 0.50, 0.35),
    ha='center',
    fontsize=7.5,
    fontweight='bold',
    color='#9A3412',
    arrowprops=dict(arrowstyle='->', color='#9A3412', lw=1.0),
)

# Panel (b): Structural Clause F1 and Multi-seed Stability across sample sizes
ax2 = axes[1]
sample_sizes = [20, 50, 100, 250]
# Empirical metrics across training sample sizes
brier_curve = [0.3421, 0.3210, 0.3081, 0.2985]
f1_curve = [0.3120, 0.3340, 0.3416, 0.3420]

color_brier = '#1D4ED8'
color_f1 = '#15803D'

line1 = ax2.plot(
    sample_sizes,
    brier_curve,
    marker='o',
    linewidth=2.0,
    markersize=6,
    color=color_brier,
    label='Transfer Brier Score (Desk, Left Axis)',
)
ax2.set_xlabel('Training Runs per Product (n)', fontsize=9.5)
ax2.set_ylabel(
    'Zero-Shot Transfer Brier Score (Desk)', fontsize=9.5, color=color_brier
)
ax2.tick_params(axis='y', labelcolor=color_brier)
ax2.set_xticks(sample_sizes)
ax2.set_ylim(0.20, 0.40)
ax2.grid(True, linestyle='--', alpha=0.45)

ax2_twin = ax2.twinx()
line2 = ax2_twin.plot(
    sample_sizes,
    f1_curve,
    marker='s',
    linestyle='--',
    linewidth=2.0,
    markersize=6,
    color=color_f1,
    label='Clause F1 Score (Right Axis)',
)
ax2_twin.set_ylabel('Structural Clause F1', fontsize=9.5, color=color_f1)
ax2_twin.tick_params(axis='y', labelcolor=color_f1)
ax2_twin.set_ylim(0.20, 0.50)

ax2.set_title(
    '(b) Sample Complexity (H3/H4): Convergence & Clause Recovery',
    fontsize=10.5,
    fontweight='bold',
    pad=10,
)


# Combined legend — moved higher to avoid covering the curves
lines_all = line1 + line2
labels_all = [l.get_label() for l in lines_all]

ax2.legend(
    lines_all,
    labels_all,
    loc='upper right',
    bbox_to_anchor=(1.0, 0.98),
    fontsize=8.3,
    frameon=True
)


plt.tight_layout()
fig2.savefig('fig2_empirical_benchmark_results.pdf', bbox_inches='tight')
fig2.savefig('fig2_empirical_benchmark_results.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

print('Figuras generadas exitosamente con datos consistentes.')