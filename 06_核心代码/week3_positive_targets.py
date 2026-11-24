from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import pandas as pd


db_path = Path("/Users/ethen/CS/Amazon/mydata/ratings_check.sqlite")
output_dir = Path("/Users/ethen/CS/Amazon/mydata/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

cutoff = int(
    datetime(2014, 1, 1, tzinfo=timezone.utc).timestamp()
)

if not db_path.exists():
    raise FileNotFoundError(f"找不到数据库：{db_path}")

conn = sqlite3.connect(
    db_path.as_uri() + "?mode=ro",
    uri=True
)

try:
    # 1. 读取训练期出现过的编号
    print("正在读取训练期用户和商品……")

    train_users = {
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT user_id
            FROM ratings
            WHERE CAST(timestamp AS INTEGER) < ?
            """,
            (cutoff,)
        )
    }

    train_products = {
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT product_id
            FROM ratings
            WHERE CAST(timestamp AS INTEGER) < ?
            """,
            (cutoff,)
        )
    }

    # 2. 分批读取测试期 4～5 星评分
    labels = {
        "known_user_known_product": "已见用户、已见商品",
        "new_user_known_product": "新用户、已见商品",
        "known_user_new_product": "已见用户、新商品",
        "new_user_new_product": "新用户、新商品"
    }

    category_counts = {key: 0 for key in labels}

    # 每位测试正反馈用户：
    # 全部正反馈目标数量，以及训练期已见商品目标数量
    user_target_counts = {}
    positive_products = set()
    processed = 0

    print("正在统计测试期正反馈目标……")

    chunks = pd.read_sql_query(
        """
        SELECT user_id, product_id
        FROM ratings
        WHERE CAST(timestamp AS INTEGER) >= ?
          AND CAST(rating AS REAL) >= 4
          AND CAST(rating AS REAL) <= 5
        """,
        conn,
        params=(cutoff,),
        chunksize=100000
    )

    for chunk in chunks:
        for user_id, product_id in chunk.itertuples(
            index=False, name=None
        ):
            known_user = user_id in train_users
            known_product = product_id in train_products

            if known_user and known_product:
                category = "known_user_known_product"
            elif not known_user and known_product:
                category = "new_user_known_product"
            elif known_user and not known_product:
                category = "known_user_new_product"
            else:
                category = "new_user_new_product"

            category_counts[category] += 1
            positive_products.add(product_id)

            if user_id not in user_target_counts:
                user_target_counts[user_id] = [0, 0]

            user_target_counts[user_id][0] += 1

            if known_product:
                user_target_counts[user_id][1] += 1

        processed += len(chunk)

        if processed % 500000 == 0:
            print(f"已统计 {processed:,} 条正反馈……")

finally:
    conn.close()


if processed == 0:
    raise ValueError("没有找到测试期正反馈，请检查。")

if sum(category_counts.values()) != processed:
    raise ValueError("四类正反馈合计不一致。")


# 3. 保存用户层面的评估目标信息
user_targets = pd.DataFrame(
    [
        {
            "user_id": user_id,
            "known_user": user_id in train_users,
            "positive_target_count": counts[0],
            "known_product_target_count": counts[1]
        }
        for user_id, counts in user_target_counts.items()
    ]
)

user_targets["new_product_target_count"] = (
    user_targets["positive_target_count"]
    - user_targets["known_product_target_count"]
)

user_targets["has_known_product_target"] = (
    user_targets["known_product_target_count"] > 0
)

total_users = len(user_targets)

known_user_count = int(user_targets["known_user"].sum())
new_user_count = total_users - known_user_count

has_known_target_count = int(
    user_targets["has_known_product_target"].sum()
)

only_new_target_count = total_users - has_known_target_count

warm_evaluable_count = int(
    (
        user_targets["known_user"]
        & user_targets["has_known_product_target"]
    ).sum()
)


# 4. 正反馈记录的四类分布
category_summary = pd.DataFrame([
    {
        "category": key,
        "category_name": labels[key],
        "positive_count": count,
        "positive_share_pct": count / processed * 100
    }
    for key, count in category_counts.items()
])

print("\n测试期正反馈概况：")
print(f"4～5 星评分数量：{processed:,}")
print(f"占全部测试评分：{processed / 1708604 * 100:.2f}%")
print(f"具有正反馈的不同用户数：{total_users:,}")
print(f"正反馈涉及的不同商品数：{len(positive_products):,}")

print("\n正反馈记录的四类分布：")
print(
    category_summary[
        ["category_name", "positive_count", "positive_share_pct"]
    ].round(2).to_string(index=False)
)


# 5. 用户覆盖情况
coverage = pd.DataFrame([
    {
        "metric": "具有测试正反馈的全部用户",
        "user_count": total_users
    },
    {
        "metric": "其中：训练期已见用户",
        "user_count": known_user_count
    },
    {
        "metric": "其中：训练期未见用户",
        "user_count": new_user_count
    },
    {
        "metric": "至少一个目标商品在训练期出现过",
        "user_count": has_known_target_count
    },
    {
        "metric": "全部目标商品均未在训练期出现",
        "user_count": only_new_target_count
    },
    {
        "metric": "已见用户且至少有一个已见商品目标",
        "user_count": warm_evaluable_count
    }
])

coverage["share_of_positive_users_pct"] = (
    coverage["user_count"] / total_users * 100
)

print("\n正反馈用户的覆盖情况：")
print(coverage.round(2).to_string(index=False))


# 6. 保存结果
user_targets.to_csv(
    output_dir / "week3_positive_user_targets.csv",
    index=False,
    encoding="utf-8-sig"
)

category_summary.to_csv(
    output_dir / "week3_positive_cold_start_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

coverage.to_csv(
    output_dir / "week3_positive_user_coverage.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n统计完成，结果已保存。原始数据库未修改。")