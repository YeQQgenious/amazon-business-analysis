from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. 读取之前保存的商品评分数量表
output_dir = Path("/Users/ethen/CS/Amazon/mydata/outputs")
file_path = output_dir / "product_rating_counts.csv"

df = pd.read_csv(
    file_path,
    dtype={"product_id": "string"}
)

counts = pd.to_numeric(df["rating_count"], errors="raise")

# 每个商品应对应一行，评分数量应为正整数
if df["product_id"].isna().any() or df["product_id"].duplicated().any():
    raise ValueError("商品编号存在缺失或重复，请检查输入表。")

if (
    counts.isna().any()
    or not np.isfinite(counts).all()
    or (counts <= 0).any()
    or (counts % 1 != 0).any()
):
    raise ValueError("评分数量存在缺失或不是正整数，请检查输入表。")


# 2. 按评分数量从少到多排序
sorted_counts = np.sort(counts.to_numpy(dtype=np.float64))

product_count = len(sorted_counts)
total_ratings = sorted_counts.sum()

if product_count == 0:
    raise ValueError("输入表没有商品记录。")

print("商品总数：", product_count)
print("评分总数：", int(total_ratings))

# 确认使用的是此前生成的全量汇总表
if product_count != 476002 or int(total_ratings) != 7824482:
    raise ValueError("汇总数量与此前全量结果不一致，请确认输入文件。")


# 3. 计算洛伦兹曲线
# 加入起点：0% 的商品对应 0% 的评分
cumulative_product_share = np.arange(product_count + 1) / product_count

cumulative_rating_share = np.concatenate([
    [0.0],
    np.cumsum(sorted_counts) / total_ratings
])


# 4. 计算基尼系数
# 此公式要求评分数量已经按升序排列
ranks = np.arange(1, product_count + 1, dtype=np.float64)

gini = (
    2 * np.sum(ranks * sorted_counts)
    / (product_count * total_ratings)
    - (product_count + 1) / product_count
)

print(f"商品评分数量的基尼系数：{gini:.6f}")


# 5. 保存统计结果
summary = pd.DataFrame({
    "product_count": [product_count],
    "total_ratings": [int(total_ratings)],
    "gini_coefficient": [gini]
})

summary_path = output_dir / "product_rating_gini_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


# 6. 绘制洛伦兹曲线
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    cumulative_product_share * 100,
    cumulative_rating_share * 100,
    color="#2878B5",
    linewidth=2,
    label=f"Lorenz curve (Gini = {gini:.4f})"
)

ax.plot(
    [0, 100],
    [0, 100],
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label="Equal distribution"
)

ax.set_title("Lorenz Curve of Product Rating Counts")
ax.set_xlabel("Cumulative share of products (%)\n"
              "(sorted by rating count, ascending)")
ax.set_ylabel("Cumulative share of ratings (%)")

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.2)
ax.legend(loc="upper left")

fig.tight_layout()

chart_path = output_dir / "product_rating_lorenz_curve.png"
fig.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("统计结果已保存至：", summary_path)
print("图表已保存至：", chart_path)