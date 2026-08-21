import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"

# Source: result/archive/0307/.../comparison_kg_vs_rl_vs_itr_vs_mole_strict.json (aemo_vic h=24 row)
# Verified 2026-07-03, same source already used for figures/figure4.jpg
rows = [
    ("iTransformer", "574.10", False),
    ("RL-QMS", "559.31", False),
    ("MoLE", "542.81", False),
    ("LightGBM（表现最好的传统单模型）", "542.81", False),
    ("本方法（KG）", "310.77", True),
]

fig, ax = plt.subplots(figsize=(6.8, 2.3))
ax.axis("off")

col_labels = ["Model", "MAE (AEMO VIC, h=24)"]
table_data = [[name, val] for name, val, _ in rows]
n_cols = 2
n_rows = len(rows) + 1

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    colLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.9)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("none")
    cell.set_linewidth(0)
    cell.set_text_props(ha="center")

for c in range(n_cols):
    tbl[(0, c)].set_text_props(weight="bold")
    tbl[(0, c)].visible_edges = "TB"
    tbl[(0, c)].set_edgecolor("black")
    tbl[(0, c)].set_linewidth(1.3)
    tbl[(n_rows - 1, c)].visible_edges = "B"
    tbl[(n_rows - 1, c)].set_edgecolor("black")
    tbl[(n_rows - 1, c)].set_linewidth(1.3)

for idx, (name, val, is_best) in enumerate(rows):
    r = idx + 1
    if is_best:
        tbl[(r, 0)].set_text_props(weight="bold")
        tbl[(r, 1)].set_text_props(weight="bold")

plt.tight_layout()
out = "/tmp/claude-1000/-home-jia----Modelcombine/27777934-db0d-4fa4-a18b-7a82d9ec3e9a/scratchpad/evidence_table.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
