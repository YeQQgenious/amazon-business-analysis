from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
from itertools import groupby
import csv
import math
import sqlite3

import numpy as np
import pandas as pd


# ---------- 1. 路径与固定配置 ----------
base_dir = Path("/Users/ethen/CS/Amazon/mydata")
output_dir = base_dir / "outputs"
model_dir = output_dir / "itemcf"
db_path = base_dir / "ratings_check.sqlite"

cutoff = int(
    datetime(2014, 1, 1, tzinfo=timezone.utc).timestamp()
)

K = 10
discounts = [1 / math.log2(rank + 1) for rank in range(1, K + 1)]

required_files = [
    db_path,
    model_dir / "product_index.csv",
    model_dir / "itemcf_neighbors_k50.npz",
    output_dir / "week3_positive_user_targets.csv",
    output_dir / "week3_popularity_ranking.csv",
    output_dir / "week3_popularity_summary.csv"
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件：{path}")


# ---------- 2. 读取商品映射和模型 ----------
products = pd.read_csv(
    model_dir / "product_index.csv",
    dtype={"product_id": "string"}
).sort_values("product_index")

if not np.array_equal(
    products["product_index"].to_numpy(),
    np.arange(len(products))
):
    raise ValueError("商品索引不是连续编号。")

product_ids = products["product_id"].tolist()
product_to_index = {
    product_id: index
    for index, product_id in enumerate(product_ids)
}

with np.load(model_dir / "itemcf_neighbors_k50.npz") as model:
    neighbor_indices = model["neighbor_indices"]
    neighbor_scores = model["neighbor_scores"]
    neighbor_counts = model["neighbor_counts"]

if (
    len(products) != 406064
    or neighbor_indices.shape != (406064, 50)
    or neighbor_scores.shape != neighbor_indices.shape
    or neighbor_counts.shape != (406064,)
):
    raise ValueError("邻居模型与商品映射规模不一致。")


# ---------- 3. 读取热门榜和评估用户 ----------
# 使用与热门基线完全相同的榜单
popularity = pd.read_csv(
    output_dir / "week3_popularity_ranking.csv",
    dtype={"product_id": "string"}
)

if (
    len(popularity) != len(products)
    or popularity["product_id"].duplicated().any()
    or set(popularity["product_id"]) != set(product_ids)
):
    raise ValueError("热门榜与模型候选商品不一致。")

popular_indices = [
    product_to_index[product_id]
    for product_id in popularity["product_id"]
]

with (output_dir / "week3_positive_user_targets.csv").open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    eval_users = {
        row["user_id"]
        for row in csv.DictReader(file)
    }

if len(eval_users) != 870128:
    raise ValueError("评估用户数量不一致。")


# ---------- 4. 读取评估用户的全部历史与正反馈历史 ----------
conn = sqlite3.connect(
    db_path.as_uri() + "?mode=ro",
    uri=True
)

try:
    print("正在读取评估用户的训练历史……")

    all_histories = {}
    positive_histories = {}

    cursor = conn.execute(
        """
        SELECT user_id, product_id, CAST(rating AS REAL)
        FROM ratings
        WHERE CAST(timestamp AS INTEGER) < ?
        """,
        (cutoff,)
    )

    for user_id, product_id, rating in cursor:
        if user_id not in eval_users:
            continue

        index = product_to_index[product_id]

        if user_id not in all_histories:
            all_histories[user_id] = set()

        all_histories[user_id].add(index)

        if 4 <= rating <= 5:
            if user_id not in positive_histories:
                positive_histories[user_id] = set()

            positive_histories[user_id].add(index)

    if len(all_histories) != 269398:
        raise ValueError("训练期已见用户数量不一致。")

    del eval_users

    print(f"有训练历史的评估用户：{len(all_histories):,}")
    print(
        "有训练期正反馈的评估用户："
        f"{len(positive_histories):,}"
    )


    # ---------- 5. 准备分组汇总 ----------
    # 用户数、Recall之和、NDCG之和、命中用户数、
    # 获得个性化推荐的用户数、个性化推荐位置总数
    aggregates = defaultdict(lambda: [0, 0.0, 0.0, 0, 0, 0])
    exposures = Counter()
    # 记录两类推荐位置的展示数和命中数
    source_stats = defaultdict(
    lambda: {
        "personalized_slots": 0,
        "fallback_slots": 0,
        "personalized_hits": 0,
        "fallback_hits": 0
    }
)

    def add_result(name, recall, ndcg, hits, personal_count):
        values = aggregates[name]
        values[0] += 1
        values[1] += recall
        values[2] += ndcg
        values[3] += int(hits > 0)
        values[4] += int(personal_count > 0)
        values[5] += personal_count

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


    # ---------- 6. 按用户生成推荐并评估 ----------
    print("\n正在生成协同过滤推荐并评估……")

    test_cursor = conn.execute(
        """
        SELECT user_id, product_id
        FROM ratings
        WHERE CAST(timestamp AS INTEGER) >= ?
          AND CAST(rating AS REAL) BETWEEN 4 AND 5
        ORDER BY user_id, product_id
        """,
        (cutoff,)
    )

    processed_users = 0
    processed_targets = 0

    with (output_dir / "week3_itemcf_user_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "user_id",
            "train_history_count",
            "train_positive_history_count",
            "positive_target_count",
            "known_product_target_count",
            "personalized_count",
            "fallback_count",
            "hits_at_10",
            "recall_at_10",
            "ndcg_at_10"
        ])

        for user_id, rows in groupby(
            test_cursor, key=lambda row: row[0]
        ):
            targets = {row[1] for row in rows}

            seen = all_histories.get(user_id, set())
            liked = positive_histories.get(user_id, set())

            # 汇总相似度得分
            scores = defaultdict(float)

            # 排序遍历保证浮点累加顺序可复现
            for source_index in sorted(liked):
                count = int(neighbor_counts[source_index])

                for position in range(count):
                    candidate = int(
                        neighbor_indices[source_index, position]
                    )

                    if candidate in seen:
                        continue

                    scores[candidate] += float(
                        neighbor_scores[source_index, position]
                    )

            # 相似度之和从高到低；并列按商品索引排序
            recommendations = sorted(
                scores,
                key=lambda index: (-scores[index], index)
            )[:K]

            personal_count = len(recommendations)
            chosen = set(recommendations)

            # 个性化不足 10 个时，按原热门榜补齐
            if len(recommendations) < K:
                for candidate in popular_indices:
                    if candidate in seen or candidate in chosen:
                        continue

                    recommendations.append(candidate)
                    chosen.add(candidate)

                    if len(recommendations) == K:
                        break

            if len(recommendations) != K:
                raise ValueError("未生成完整的 10 个推荐商品。")

            if chosen & seen:
                raise ValueError("推荐中包含训练期已评分商品。")

            known_target_count = sum(
                product_id in product_to_index
                for product_id in targets
            )

            # 检查训练历史与测试目标不存在重叠
            if any(
                product_to_index.get(product_id, -1) in seen
                for product_id in targets
            ):
                raise ValueError("训练历史与测试目标重叠。")

            relevance = [
                int(product_ids[index] in targets)
                for index in recommendations
            ]

            hits = sum(relevance)
            recall = hits / len(targets)

            dcg = sum(
                value * discounts[position]
                for position, value in enumerate(relevance)
            )

            # 保留所有目标，包括训练期未见商品
            idcg = sum(discounts[:min(K, len(targets))])
            ndcg = dcg / idcg

            groups = [
                "all_positive_users",
                "new_users" if not seen else "known_users",
                history_group(len(seen))
            ]

            if seen and known_target_count > 0:
                groups.append("known_users_with_known_targets")

            
            # 推荐列表前 personal_count 个位置来自协同过滤，
            # 后面的剩余位置来自热门补齐。
            personalized_hits = sum(relevance[:personal_count])
            fallback_hits = sum(relevance[personal_count:])
            if personalized_hits + fallback_hits != hits:
                raise ValueError("两类命中数的合计与总命中数不一致。")

            for group in groups:
                add_result(
                    group, recall, ndcg, hits, personal_count
                )
                source_stats[group]["personalized_slots"] += personal_count
                source_stats[group]["fallback_slots"] += K - personal_count

                source_stats[group]["personalized_hits"] += personalized_hits
                source_stats[group]["fallback_hits"] += fallback_hits


            exposures.update(recommendations)

            writer.writerow([
                user_id,
                len(seen),
                len(liked),
                len(targets),
                known_target_count,
                personal_count,
                K - personal_count,
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


# ---------- 7. 核对范围 ----------
if processed_users != 870128 or processed_targets != 1287793:
    raise ValueError("评估用户数或目标数与基线不一致。")

if aggregates["known_users_with_known_targets"][0] != 246891:
    raise ValueError("补充评估组用户数不一致。")


# ---------- 8. 保存汇总 ----------
labels = {
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

summary_rows = []

for group, label in labels.items():
    count, recall_sum, ndcg_sum, hit_users, personal_users, slots = (
        aggregates[group]
    )

    if count == 0:
        continue

    summary_rows.append({
        "group": group,
        "group_name": label,
        "user_count": count,
        "recall_at_10": recall_sum / count,
        "ndcg_at_10": ndcg_sum / count,
        "hit_rate_at_10": hit_users / count,
        "personalized_user_pct": personal_users / count * 100,
        "full_fallback_user_pct":
            (count - personal_users) / count * 100,
        "personalized_slot_pct": slots / (count * K) * 100
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    output_dir / "week3_itemcf_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# ---------- 9. 与热门基线比较 ----------
baseline = pd.read_csv(
    output_dir / "week3_popularity_summary.csv"
)

comparison = summary.merge(
    baseline[
        ["group", "user_count", "recall_at_10", "ndcg_at_10"]
    ],
    on="group",
    suffixes=("_itemcf", "_baseline"),
    validate="one_to_one"
)

if (
    len(comparison) != len(summary)
    or not (
        comparison["user_count_itemcf"]
        == comparison["user_count_baseline"]
    ).all()
):
    raise ValueError("两种方法的分组评估人数不一致。")

for metric in ["recall_at_10", "ndcg_at_10"]:
    comparison[f"{metric}_difference"] = (
        comparison[f"{metric}_itemcf"]
        - comparison[f"{metric}_baseline"]
    )

    comparison[f"{metric}_relative_change_pct"] = (
        comparison[f"{metric}_difference"]
        / comparison[f"{metric}_baseline"].replace(0, np.nan)
        * 100
    )

# 新用户完全使用同一热门榜，指标应与基线一致。
# 基线 CSV 保存的是完整精度，不使用终端四舍五入值。
new_user_row = comparison[
    comparison["group"] == "new_users"
].iloc[0]

for metric in ["recall_at_10", "ndcg_at_10"]:
    if not np.isclose(
        new_user_row[f"{metric}_itemcf"],
        new_user_row[f"{metric}_baseline"],
        rtol=1e-9,
        atol=1e-12
    ):
        raise ValueError("新用户结果与热门基线不一致，请检查兜底逻辑。")

comparison.to_csv(
    output_dir / "week3_itemcf_vs_popularity.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n协同过滤＋热门补齐，与热门基线对比：")

for _, row in comparison.iterrows():
    print(f"\n【{row['group_name']}】")
    print(f"用户数：{int(row['user_count_itemcf']):,}")

    print(
        f"Recall@10："
        f"{row['recall_at_10_baseline']:.6f}"
        f" → {row['recall_at_10_itemcf']:.6f}"
    )
    print(
        "Recall 相对变化："
        f"{row['recall_at_10_relative_change_pct']:+.2f}%"
    )

    print(
        f"NDCG@10："
        f"{row['ndcg_at_10_baseline']:.6f}"
        f" → {row['ndcg_at_10_itemcf']:.6f}"
    )
    print(
        "NDCG 相对变化："
        f"{row['ndcg_at_10_relative_change_pct']:+.2f}%"
    )

    print(
        "至少获得一个个性化推荐的用户占比："
        f"{row['personalized_user_pct']:.2f}%"
    )
    print(
        "完全依赖热门兜底的用户占比："
        f"{row['full_fallback_user_pct']:.2f}%"
    )


# ---------- 10. 保存推荐覆盖 ----------
coverage = pd.DataFrame([{
    "candidate_product_count": len(product_ids),
    "recommended_product_count": len(exposures),
    "catalog_coverage": len(exposures) / len(product_ids)
}])

coverage.to_csv(
    output_dir / "week3_itemcf_coverage.csv",
    index=False,
    encoding="utf-8-sig"
)

with (output_dir / "week3_itemcf_item_exposure.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(["product_id", "recommendation_count"])

    for index, count in exposures.most_common():
        writer.writerow([product_ids[index], count])

print("\n商品推荐覆盖：")
print(f"候选商品数：{len(product_ids):,}")
print(f"至少被推荐一次的商品数：{len(exposures):,}")
print(f"商品覆盖率：{len(exposures) / len(product_ids):.6%}")

print("\n核对通过，结果已保存。原始数据库未修改。")

# ---------- 11. 命中来源汇总 ----------
source_rows = []

for group, label in labels.items():
    user_count = aggregates[group][0]

    if user_count == 0:
        continue

    stats = source_stats[group]

    personal_slots = stats["personalized_slots"]
    fallback_slots = stats["fallback_slots"]

    personal_hits = stats["personalized_hits"]
    fallback_hits = stats["fallback_hits"]

    total_slots = personal_slots + fallback_slots
    total_hits = personal_hits + fallback_hits

    if total_slots != user_count * K:
        raise ValueError(f"{label} 的推荐位置总数不一致。")

    source_rows.append({
        "group": group,
        "group_name": label,
        "user_count": user_count,

        "personalized_slots": personal_slots,
        "fallback_slots": fallback_slots,

        "personalized_hits": personal_hits,
        "fallback_hits": fallback_hits,
        "total_hits": total_hits,

        # 每 100 个推荐位置中，有多少命中测试正反馈目标
        "personalized_hits_per_100_slots": (
            personal_hits / personal_slots * 100
            if personal_slots else float("nan")
        ),
        "fallback_hits_per_100_slots": (
            fallback_hits / fallback_slots * 100
            if fallback_slots else float("nan")
        ),

        # 全部命中中，两类来源分别贡献多少
        "personalized_share_of_hits_pct": (
            personal_hits / total_hits * 100
            if total_hits else float("nan")
        ),
        "fallback_share_of_hits_pct": (
            fallback_hits / total_hits * 100
            if total_hits else float("nan")
        ),

        # 按本组全部用户平均，可与之前的平均命中数衔接
        "personalized_hits_per_user": personal_hits / user_count,
        "fallback_hits_per_user": fallback_hits / user_count
    })

source_summary = pd.DataFrame(source_rows)

source_summary.to_csv(
    output_dir / "week3_itemcf_hit_sources.csv",
    index=False,
    encoding="utf-8-sig"
)


def display_number(value):
    if pd.isna(value):
        return "不适用（分母为 0）"
    return f"{value:.4f}"


print("\n========== 命中来源统计 ==========")

for row in source_rows:
    print(f"\n【{row['group_name']}】")
    print(f"用户数：{row['user_count']:,}")

    print(
        f"个性化推荐位置数：{row['personalized_slots']:,}"
    )
    print(
        f"热门补齐位置数：{row['fallback_slots']:,}"
    )

    print(
        f"个性化位置命中数：{row['personalized_hits']:,}"
    )
    print(
        f"热门补齐位置命中数：{row['fallback_hits']:,}"
    )

    print(
        "每 100 个个性化位置的命中数："
        + display_number(
            row["personalized_hits_per_100_slots"]
        )
    )
    print(
        "每 100 个热门补齐位置的命中数："
        + display_number(
            row["fallback_hits_per_100_slots"]
        )
    )

    print(
        "个性化位置贡献的命中占比（%）："
        + display_number(
            row["personalized_share_of_hits_pct"]
        )
    )
    print(
        "热门补齐贡献的命中占比（%）："
        + display_number(
            row["fallback_share_of_hits_pct"]
        )
    )

print(
    "\n命中来源结果已保存至：",
    output_dir / "week3_itemcf_hit_sources.csv"
)