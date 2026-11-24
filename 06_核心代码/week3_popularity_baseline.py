from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
from itertools import groupby
import sqlite3
import csv
import math


# ---------- 1. 固定实验设置 ----------
base_dir = Path("/Users/ethen/CS/Amazon/mydata")
db_path = base_dir / "ratings_check.sqlite"
output_dir = base_dir / "outputs"

# 上一步已经生成的用户目标统计表
target_path = output_dir / "week3_positive_user_targets.csv"

output_dir.mkdir(parents=True, exist_ok=True)

if not db_path.exists() or not target_path.exists():
    raise FileNotFoundError("数据库或用户目标统计表不存在，请检查路径。")

cutoff = int(
    datetime(2014, 1, 1, tzinfo=timezone.utc).timestamp()
)

K = 10

# 第 1～10 名的折扣权重
discounts = [
    1 / math.log2(rank + 1)
    for rank in range(1, K + 1)
]


# ---------- 2. 读取需要评估的用户编号 ----------
with target_path.open("r", encoding="utf-8-sig", newline="") as file:
    eval_users = {
        row["user_id"]
        for row in csv.DictReader(file)
    }

if len(eval_users) != 870128:
    raise ValueError("评估用户数量与上一步不一致。")

print(f"本次评估用户：{len(eval_users):,}")


conn = sqlite3.connect(
    db_path.as_uri() + "?mode=ro",
    uri=True
)

try:
    # ---------- 3. 仅使用训练期数据建立热门榜 ----------
    print("\n正在建立训练期热门榜……")

    # 候选范围：全部训练期商品
    # 热度：训练期 4～5 星评分数量
    # 正反馈为 0 的训练期商品仍保留在榜尾
    popularity_rows = conn.execute(
        """
        SELECT
            product_id,
            SUM(
                CASE
                    WHEN CAST(rating AS REAL) >= 4
                     AND CAST(rating AS REAL) <= 5
                    THEN 1 ELSE 0
                END
            ) AS positive_count
        FROM ratings
        WHERE CAST(timestamp AS INTEGER) < ?
        GROUP BY product_id
        ORDER BY positive_count DESC, product_id ASC
        """,
        (cutoff,)
    ).fetchall()

    ranked_products = [row[0] for row in popularity_rows]
    train_products = set(ranked_products)

    if len(train_products) != 406064:
        raise ValueError("训练期商品数量与此前结果不一致。")

    with (output_dir / "week3_popularity_ranking.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["product_id", "train_positive_count"])
        writer.writerows(popularity_rows)

    del popularity_rows

    # ---------- 4. 保存评估用户的训练期历史 ----------
    # 所有星级都计入“已经评分过”，不只是 4～5 星
    print("正在读取评估用户的训练期历史……")

    histories = {}

    cursor = conn.execute(
        """
        SELECT user_id, product_id
        FROM ratings
        WHERE CAST(timestamp AS INTEGER) < ?
        """,
        (cutoff,)
    )

    for user_id, product_id in cursor:
        if user_id in eval_users:
            if user_id not in histories:
                histories[user_id] = set()
            histories[user_id].add(product_id)

    if len(histories) != 269398:
        raise ValueError("具有训练期历史的评估用户数量不一致。")

    print(f"其中有训练期历史的用户：{len(histories):,}")

    # 释放不再需要的编号集合
    del eval_users

    # ---------- 5. 准备汇总容器 ----------
    # 每组保存：用户数、Recall之和、NDCG之和、命中用户数
    aggregates = defaultdict(lambda: [0, 0.0, 0.0, 0])
    recommendation_counts = Counter()

    def add_result(group_name, recall, ndcg, hits):
        values = aggregates[group_name]
        values[0] += 1
        values[1] += recall
        values[2] += ndcg
        values[3] += int(hits > 0)

    def history_group(count):
        if count == 0:
            return "history_0"
        if count == 1:
            return "history_1"
        if count <= 5:
            return "history_2_5"
        if count <= 10:
            return "history_6_10"
        return "history_11_plus"

    # ---------- 6. 按用户读取测试正反馈并评估 ----------
    # 按用户排序后逐个处理，不把全部测试目标同时放入内存
    print("\n正在计算推荐指标……")

    test_cursor = conn.execute(
        """
        SELECT user_id, product_id
        FROM ratings
        WHERE CAST(timestamp AS INTEGER) >= ?
          AND CAST(rating AS REAL) >= 4
          AND CAST(rating AS REAL) <= 5
        ORDER BY user_id, product_id
        """,
        (cutoff,)
    )

    user_result_path = output_dir / "week3_popularity_user_metrics.csv"

    processed_users = 0
    processed_targets = 0

    with user_result_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "user_id",
            "train_history_count",
            "positive_target_count",
            "known_product_target_count",
            "recommendation_count",
            "hits_at_10",
            "recall_at_10",
            "ndcg_at_10"
        ])

        for user_id, rows in groupby(
            test_cursor, key=lambda row: row[0]
        ):
            targets = {row[1] for row in rows}
            seen = histories.get(user_id, set())
            history_count = len(seen)

            # 此数据此前已确认用户—商品组合无重复
            if targets & seen:
                raise ValueError("训练历史与测试目标出现重叠。")

            known_target_count = sum(
                product_id in train_products
                for product_id in targets
            )

            # 排除训练期已评分商品，再取前 10 个
            recommendations = []

            for product_id in ranked_products:
                if product_id not in seen:
                    recommendations.append(product_id)

                if len(recommendations) == K:
                    break

            relevance = [
                int(product_id in targets)
                for product_id in recommendations
            ]

            hits = sum(relevance)
            recall = hits / len(targets)

            dcg = sum(
                value * discounts[index]
                for index, value in enumerate(relevance)
            )

            # 所有正反馈按同等相关性处理
            # 理想排序的分母包括新商品目标
            idcg = sum(discounts[:min(K, len(targets))])
            ndcg = dcg / idcg

            add_result("all_positive_users", recall, ndcg, hits)

            if history_count == 0:
                add_result("new_users", recall, ndcg, hits)
            else:
                add_result("known_users", recall, ndcg, hits)

            add_result(
                history_group(history_count), recall, ndcg, hits
            )

            if history_count > 0 and known_target_count > 0:
                add_result(
                    "known_users_with_known_targets",
                    recall, ndcg, hits
                )

            recommendation_counts.update(recommendations)

            writer.writerow([
                user_id,
                history_count,
                len(targets),
                known_target_count,
                len(recommendations),
                hits,
                recall,
                ndcg
            ])

            processed_users += 1
            processed_targets += len(targets)

            if processed_users % 100000 == 0:
                print(f"已评估 {processed_users:,} 个用户……")

finally:
    conn.close()


# ---------- 7. 核对评估范围 ----------
if processed_users != 870128:
    raise ValueError("实际评估用户数与预期不一致。")

if processed_targets != 1287793:
    raise ValueError("实际正反馈目标数量与预期不一致。")

if aggregates["known_users_with_known_targets"][0] != 246891:
    raise ValueError("补充评估组人数与预期不一致。")


# ---------- 8. 保存并打印分组结果 ----------
group_labels = {
    "all_positive_users": "全部正反馈用户",
    "new_users": "新用户",
    "known_users": "已见用户",
    "known_users_with_known_targets": "已见用户且有已见商品目标",
    "history_0": "训练历史为 0 条",
    "history_1": "训练历史为 1 条",
    "history_2_5": "训练历史为 2～5 条",
    "history_6_10": "训练历史为 6～10 条",
    "history_11_plus": "训练历史为 11 条及以上"
}

summary_path = output_dir / "week3_popularity_summary.csv"

with summary_path.open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "group",
        "group_name",
        "user_count",
        "recall_at_10",
        "ndcg_at_10",
        "hit_rate_at_10"
    ])

    print("\n热门推荐基线结果：")

    for group, label in group_labels.items():
        count, recall_sum, ndcg_sum, hit_users = aggregates[group]

        if count == 0:
            print(f"\n{label}：没有符合条件的用户")
            continue

        mean_recall = recall_sum / count
        mean_ndcg = ndcg_sum / count
        hit_rate = hit_users / count

        writer.writerow([
            group, label, count,
            mean_recall, mean_ndcg, hit_rate
        ])

        print(f"\n【{label}】用户数：{count:,}")
        print(f"Recall@10：{mean_recall:.6f}")
        print(f"NDCG@10：{mean_ndcg:.6f}")
        print(f"至少命中一次的用户占比：{hit_rate:.2%}")


# ---------- 9. 商品推荐覆盖 ----------
recommended_product_count = len(recommendation_counts)
catalog_coverage = recommended_product_count / len(train_products)

with (output_dir / "week3_popularity_item_exposure.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(["product_id", "recommendation_count"])
    writer.writerows(recommendation_counts.most_common())

with (output_dir / "week3_popularity_coverage.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "candidate_product_count",
        "recommended_product_count",
        "catalog_coverage"
    ])
    writer.writerow([
        len(train_products),
        recommended_product_count,
        catalog_coverage
    ])

print("\n商品推荐覆盖：")
print(f"训练期候选商品数：{len(train_products):,}")
print(f"至少被推荐过一次的商品数：{recommended_product_count:,}")
print(f"商品覆盖率：{catalog_coverage:.6%}")

print("\n核对通过，结果已保存至：", output_dir)
print("原始数据库未修改。")