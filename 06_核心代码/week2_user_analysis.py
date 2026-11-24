from pathlib import Path
import sqlite3
import pandas as pd

db_path = Path(
    "/Users/ethen/CS/Amazon/mydata/ratings_check.sqlite"
)

# 只读连接，不修改数据库
connection = sqlite3.connect(
    db_path.as_uri() + "?mode=ro",
    uri=True
)

try:
    print("正在统计每个用户的评分数量……", flush=True)

    # 先计算每个用户有多少条评分，
    # 再统计具有相同评分数量的用户有多少
    sql = """
    SELECT
        rating_count,
        COUNT(*) AS user_count
    FROM (
        SELECT
            user_id,
            COUNT(*) AS rating_count
        FROM ratings
        GROUP BY user_id
    )
    GROUP BY rating_count
    ORDER BY rating_count
    """

    user_distribution = pd.read_sql_query(sql, connection)

finally:
    connection.close()

# 用户总数
total_users = int(user_distribution["user_count"].sum())

# 从分布反算评分总数，用于核对
total_ratings = int(
    (
        user_distribution["rating_count"]
        * user_distribution["user_count"]
    ).sum()
)

# 只有一条评分的用户
single_rating_users = int(
    user_distribution.loc[
        user_distribution["rating_count"] == 1,
        "user_count"
    ].sum()
)

single_rating_pct = single_rating_users / total_users * 100

print("\n用户总数：", total_users)
print("核对评分总数：", total_ratings)
print("只有一条评分的用户数：", single_rating_users)
print(f"只有一条评分的用户占比：{single_rating_pct:.2f}%")
print(f"平均每位用户的评分数：{total_ratings / total_users:.2f}")

print("\n用户评分数量分布（前 20 档）：")
print(user_distribution.head(20).to_string(index=False))


# 创建结果文件夹，与当前脚本放在同一目录下
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

# 保存用户评分条数分布
user_distribution.to_csv(
    output_dir / "user_rating_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

# 保存关键指标
user_summary = pd.DataFrame({
    "metric": [
        "total_users",
        "total_ratings",
        "single_rating_users",
        "single_rating_user_pct",
        "mean_ratings_per_user"
    ],
    "value": [
        total_users,
        total_ratings,
        single_rating_users,
        single_rating_pct,
        total_ratings / total_users
    ]
})

user_summary.to_csv(
    output_dir / "user_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n用户分析结果已保存至：", output_dir)