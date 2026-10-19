import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------------------
# FIGURA SECCIÓN 4.4: Ciclo de vida y transformación de eventos a reglas LPAD
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 3.8), dpi=300)
ax.axis("off")

# Definir cajas del pipeline
box_props = dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#475569", linewidth=1.2)
highlight_props = dict(boxstyle="round,pad=0.5", facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.4)
lpad_props = dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.4)
query_props = dict(boxstyle="round,pad=0.5", facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.4)

ax.text(0.12, 0.5, 
        "1. Event Log Entry\n"
        "───────────────────────\n"
        "run_id: r42\n"
        "product: table\n"
        "material: heavy_wood\n"
        "geometry: compact\n"
        "operation: cutting\n"
        "station: s1\n"
        "attempted: 1, failed: 1",
        va="center", ha="center", family="monospace", fontsize=8.5, bbox=box_props)

ax.text(0.38, 0.5,
        "2. Logical Interpretation\n"
        "───────────────────────\n"
        "begin(model(ev_42)).\n"
        "material(heavy_wood).\n"
        "geometry(compact).\n"
        "assigned(cutting, s1).\n"
        "failed(cutting, s1).\n"
        "end(model(ev_42)).",
        va="center", ha="center", family="monospace", fontsize=8.5, bbox=highlight_props)

ax.text(0.65, 0.5,
        "3. Learned LPAD Rule\n"
        "───────────────────────\n"
        "failed(Station) : 1.0000 :-\n"
        "    assigned(cutting, Station),\n"
        "    material(heavy_wood).\n"
        "\n"
        "/* Induced via SLIPCOVER */",
        va="center", ha="center", family="monospace", fontsize=8.5, bbox=lpad_props)

ax.text(0.90, 0.5,
        "4. Probabilistic Query\n"
        "───────────────────────\n"
        "?- prob(failed(s1),\n"
        "        [product(desk)],\n"
        "        Prob).\n"
        "\n"
        "Output: P(F) = 0.3695\n"
        "Evaluated via PITA BDD",
        va="center", ha="center", family="monospace", fontsize=8.5, bbox=query_props)

# Flechas conectivas
arrow = dict(arrowstyle="->", color="#334155", lw=1.8, mutation_scale=15)
ax.annotate("", xy=(0.24, 0.5), xytext=(0.26, 0.5), arrowprops=arrow)
ax.annotate("", xy=(0.50, 0.5), xytext=(0.52, 0.5), arrowprops=arrow)
ax.annotate("", xy=(0.77, 0.5), xytext=(0.79, 0.5), arrowprops=arrow)

plt.tight_layout()
fig_path_44 = "figures/fig_event_to_interpretation.pdf"
fig.savefig(fig_path_44, bbox_inches="tight")
fig.savefig("figures/fig_event_to_interpretation.png", bbox_inches="tight")
plt.close(fig)
print(f"[OK] Generada figura Sección 4.4: {fig_path_44}")


# -------------------------------------------------------------------------
# FIGURA SECCIÓN 5.2: Matriz de Rutas de la Línea de Productos FMS
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=300)

products = ["chair", "stool", "table", "bench", "armchair", "desk (held-out)"]
stations = ["s1 (Cutting)", "s2 (Drilling)", "s3 (Assembly)", "s4 (Sanding)", "s5 (Varnishing)"]

# Matriz de operaciones: None si no pasa por la estación
# Formato: (Operación, Compartida/Específica)
routes_matrix = [
    # chair
    ["Cutting", "Leg drilling", "Seat assembly", "Sanding (opt)", "Varnishing"],
    # stool
    ["Cutting", "Leg drilling", "Seat assembly", None, "Varnishing"],
    # table
    ["Cutting", "Leg drilling", "Top assembly", "Sanding (opt)", "Varnishing"],
    # bench
    ["Cutting", "Leg drilling", "Seat assembly", "Sanding (opt)", "Varnishing"],
    # armchair
    ["Cutting", "Leg drilling", "Seat assembly", "Sanding (opt)", "Varnishing"],
    # desk
    ["Cutting", "Leg drilling", "Top assembly", "Sanding (opt)", "Varnishing"],
]

# Configuración de grilla
ax.set_xlim(-0.5, len(stations) - 0.5)
ax.set_ylim(-0.5, len(products) - 0.5)
ax.set_xticks(range(len(stations)))
ax.set_xticklabels(stations, fontsize=10, fontweight="bold")
ax.set_yticks(range(len(products)))
ax.set_yticklabels(products, fontsize=10, fontweight="bold")
ax.invert_yaxis()

for i, prod in enumerate(products):
    is_held_out = (i == 5)
    for j, st in enumerate(stations):
        op = routes_matrix[i][j]
        if op is None:
            # Estación saltada (bypass)
            rect = patches.Rectangle((j - 0.42, i - 0.35), 0.84, 0.7, linewidth=1,
                                     edgecolor="#CBD5E1", facecolor="#F1F5F9", linestyle=":")
            ax.add_patch(rect)
            ax.text(j, i, "Bypassed", ha="center", va="center", color="#94A3B8", fontsize=8.5, style="italic")
        else:
            # Colores según tipo de operación y partición
            if is_held_out:
                edge_col = "#DC2626"
                face_col = "#FEE2E2"
                text_col = "#991B1B"
            elif op == "Top assembly":
                edge_col = "#7C3AED"
                face_col = "#EDE9FE"
                text_col = "#5B21B6"
            elif op.startswith("Sanding"):
                edge_col = "#D97706"
                face_col = "#FEF3C7"
                text_col = "#92400E"
            else:
                edge_col = "#2563EB"
                face_col = "#EFF6FF"
                text_col = "#1E40AF"

            rect = patches.Rectangle((j - 0.42, i - 0.35), 0.84, 0.7, linewidth=1.3,
                                     edgecolor=edge_col, facecolor=face_col)
            ax.add_patch(rect)
            ax.text(j, i, op, ha="center", va="center", color=text_col, fontsize=9, fontweight="semibold")


# Destacar la fila de Desk (Held-out)
rect_desk = patches.Rectangle(
    (-0.48, 4.55), 4.96, 1.00,
    linewidth=1.8,
    edgecolor="#DC2626",
    facecolor="none",
    linestyle="--",
    zorder=5,
    clip_on=False
)
ax.add_patch(rect_desk)

# Ampliar el margen inferior para separar la anotación
# de las etiquetas de las estaciones
ax.set_ylim(5.95, -0.5)

# Anotación roja debajo de la fila, sin superponer las cajas
ax.text(
    2.0, 5.72,  # Antes: 5.62
    "Held-out Product (Excluded from Training, Zero-Shot Evaluation for H2)",
    ha="center",
    va="center",
    color="#DC2626",
    fontsize=8,
    fontweight="bold",
    clip_on=False
)

ax.set_title(
    "Manufacturing Product Line Workstation Routings and Station Allocation",
    fontsize=11,
    pad=12,
    fontweight="bold"
)

plt.tight_layout()

fig_path_52 = "figures/fig_fms_routes.pdf"
fig.savefig(fig_path_52, bbox_inches="tight", dpi=300)
fig.savefig(
    "figures/fig_fms_routes.png",
    bbox_inches="tight",
    dpi=300
)
plt.close(fig)

print(f"[OK] Generada figura Sección 5.2: {fig_path_52}")
