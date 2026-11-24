from pathlib import Path
import sqlite3
import pandas as pd


# 1. 连接已有数据库，只读查询
db_path = Path("/Users/ethen/CS/Amazon/mydata/ratings_check.sqlite")
output_dir = Path("/Users/ethen/CS/Amazon/mydata/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

if not db_path.exists():
    raise FileNotFoundError(f"找不到数据库：{db_path}")

conn = sqlite3.connect(
    db_path.as_uri() + "?mode=ro",
    uri=True
)

try:
    # 2. 按 UTC 年月统计评分数量
    print("正在统计每个月的评分数量，请稍等……")

    monthly = pd.read_sql_query(
        """
        SELECT
            strftime(
                '%Y-%m',
                CAST(timestamp AS INTEGER),
                'unixepoch'
            ) AS month,
            COUNT(*) AS rating_count
        FROM ratings
        GROUP BY month
        ORDER BY month
        """,
        conn
    )

finally:
    conn.close()


# 3. 核对结果
if monthly["month"].isna().any():
    raise ValueError("存在无法转换为年月的时间戳，请检查。")

total_ratings = int(monthly["rating_count"].sum())

if total_ratings != 7824482:
    raise ValueError(
        f"评分总数与此前结果不一致：{total_ratings}"
    )

monthly["rating_share_pct"] = (
    monthly["rating_count"] / total_ratings * 100
)

monthly["cumulative_count"] = monthly["rating_count"].cumsum()

monthly["cumulative_share_pct"] = (
    monthly["cumulative_count"] / total_ratings * 100
)


# 4. 汇总年度分布
monthly["year"] = monthly["month"].str[:4]

yearly = (
    monthly.groupby("year", as_index=False)["rating_count"]
    .sum()
)

yearly["rating_share_pct"] = (
    yearly["rating_count"] / total_ratings * 100
)

print("\n年度评分分布：")
print(yearly.round(2).to_string(index=False))

print("\n最后 24 个月的评分分布：")
print(
    monthly[
        [
            "month",
            "rating_count",
            "rating_share_pct",
            "cumulative_share_pct"
        ]
    ]
    .tail(24)
    .round(2)
    .to_string(index=False)
)


# 5. 寻找接近 80% / 20% 的候选月份边界
# 每个候选日期都是某月第一天：
# 日期之前为训练集，该日期及之后为测试集
candidates = monthly[["month"]].copy()

candidates["cutoff_date_utc"] = candidates["month"] + "-01"

candidates["train_count"] = (
    monthly["cumulative_count"] - monthly["rating_count"]
)

candidates["test_count"] = (
    total_ratings - candidates["train_count"]
)

candidates["train_share_pct"] = (
    candidates["train_count"] / total_ratings * 100
)

candidates["test_share_pct"] = (
    candidates["test_count"] / total_ratings * 100
)

# 排除训练集或测试集为空的情况
candidates = candidates[
    (candidates["train_count"] > 0)
    & (candidates["test_count"] > 0)
].copy()

candidates["distance_from_80"] = (
    candidates["train_share_pct"] - 80
).abs()

closest = (
    candidates.sort_values(
        ["distance_from_80", "cutoff_date_utc"]
    )
    .head(5)
    .sort_values("cutoff_date_utc")
)

candidate_columns = [
    "cutoff_date_utc",
    "train_count",
    "train_share_pct",
    "test_count",
    "test_share_pct"
]

print("\n接近 80% 训练 / 20% 测试的候选日期：")
print(
    closest[candidate_columns]
    .round(2)
    .to_string(index=False)
)


# 6. 保存结果
monthly.to_csv(
    output_dir / "week3_monthly_rating_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

yearly.to_csv(
    output_dir / "week3_yearly_rating_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

closest[candidate_columns].to_csv(
    output_dir / "week3_time_split_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n时间分布及候选日期已保存至：", output_dir)
print("本次仅统计分布，尚未正式划分训练集和测试集。")