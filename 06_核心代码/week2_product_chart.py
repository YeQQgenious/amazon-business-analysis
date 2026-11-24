from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / "outputs"

# 读取前 1%、5%、10% 商品的评分占比
concentration = pd.read_csv(
    output_dir / "product_rating_concentration.csv"
).sort_values("top_product_pct")

print("绘图数据：")
print(concentration.to_string(index=False))

labels = [
    f"Top {value:g}%"
    for value in concentration["top_product_pct"]
]

shares = concentration["rating_share_pct"].tolist()

# 绘制柱状图
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    labels,
    shares,
    color=["#73A9CF", "#468DBB", "#2878B5"]
)

for bar, share in zip(bars, shares):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{share:.2f}%",
        ha="center",
        va="bottom"
    )

ax.set_title("Concentration of Ratings Across Products")
ax.set_xlabel("Products ranked by number of ratings")
ax.set_ylabel("Cumulative share of ratings (%)")
ax.set_ylim(0, 100)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 提醒：三个分组相互包含，不能将占比相加
fig.text(
    0.5,
    0.02,
    "Groups overlap: Top 1% is included in Top 5% and Top 10%.",
    ha="center",
    fontsize=9,
    color="#555555"
)

fig.tight_layout(rect=[0, 0.06, 1, 1])

image_path = output_dir / "product_rating_concentration.png"
fig.savefig(image_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n图表已保存至：", image_path)