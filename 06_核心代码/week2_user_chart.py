from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 读取之前保存的统计结果
output_dir = Path(__file__).resolve().parent / "outputs"

distribution = pd.read_csv(
    output_dir / "user_rating_distribution.csv"
)

counts = distribution["rating_count"]
users = distribution["user_count"]

# 按每位用户的评分条数分组
group_users = [
    int(users[counts == 1].sum()),
    int(users[counts.between(2, 5)].sum()),
    int(users[counts.between(6, 10)].sum()),
    int(users[counts >= 11].sum())
]

labels = ["1", "2–5", "6–10", "11+"]

total_users = int(users.sum())

# 核对分组是否遗漏或重复计算用户
assert sum(group_users) == total_users, "分组数量与用户总数不一致"

percentages = [
    number / total_users * 100
    for number in group_users
]

# 保存绘图使用的数据
chart_data = pd.DataFrame({
    "ratings_per_user": labels,
    "user_count": group_users,
    "user_share_pct": percentages
})

chart_data.to_csv(
    output_dir / "user_rating_groups.csv",
    index=False,
    encoding="utf-8-sig"
)

# 使用英文图中文字，暂时避免中文字体缺失问题
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    labels,
    percentages,
    color=["#2878B5", "#73A9CF", "#73A9CF", "#73A9CF"]
)

# 在每根柱子上标注百分比
for bar, percentage in zip(bars, percentages):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{percentage:.2f}%",
        ha="center",
        va="bottom"
    )

ax.set_title("User Distribution by Number of Ratings")
ax.set_xlabel("Number of ratings per user")
ax.set_ylabel("Share of users (%)")
ax.set_ylim(0, max(percentages) + 12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

image_path = output_dir / "user_rating_distribution.png"
fig.savefig(image_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("分组结果：")
print(chart_data.to_string(index=False))

print("\n图表已保存至：", image_path)