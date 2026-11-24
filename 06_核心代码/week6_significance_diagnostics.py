from pathlib import Path
import numpy as np
import pandas as pd

# 1. 已有结果文件所在目录
output_dir = Path("/Users/ethen/CS/Amazon/mydata/outputs")

baseline_path = output_dir / "week3_popularity_user_metrics.csv"
itemcf_path = output_dir / "week3_itemcf_user_metrics.csv"

# 2. 读取前检查文件
for path in (baseline_path, itemcf_path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")

baseline = pd.read_csv(
    baseline_path,
    encoding="utf-8-sig",
    dtype={"user_id": "string"},
)
itemcf = pd.read_csv(
    itemcf_path,
    encoding="utf-8-sig",
    dtype={"user_id": "string"},
)

# 3. 检查必要字段、用户编号与指标
required = [
    "user_id",
    "train_history_count",
    "positive_target_count",
    "known_product_target_count",
    "recall_at_10",
    "ndcg_at_10",
]

for name, data in [("热门基线", baseline), ("ItemCF", itemcf)]:
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{name}缺少字段：{missing}")

    if data[required].isna().any().any():
        raise ValueError(f"{name}存在缺失值，请先检查")

    if data["user_id"].str.strip().eq("").any():
        raise ValueError(f"{name}存在空用户编号")

    if data["user_id"].duplicated().any():
        raise ValueError(f"{name}存在重复用户编号，不能直接配对")

    for metric in ["recall_at_10", "ndcg_at_10"]:
        values = pd.to_numeric(data[metric], errors="raise")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError(f"{name}的{metric}存在异常值")
        data[metric] = values

# 4. 按用户编号配对，不能直接按行号相减
paired = baseline.merge(
    itemcf,
    on="user_id",
    how="outer",
    suffixes=("_baseline", "_itemcf"),
    validate="one_to_one",
    indicator=True,
)

if not paired["_merge"].eq("both").all():
    raise ValueError("两份结果的用户集合不同，请先核对评估范围")

# 同一用户的历史与评估目标数量应一致
for column in [
    "train_history_count",
    "positive_target_count",
    "known_product_target_count",
]:
    if not paired[f"{column}_baseline"].eq(
        paired[f"{column}_itemcf"]
    ).all():
        raise ValueError(f"两份结果的{column}不一致")

paired = paired.drop(columns="_merge")

# 5. 计算每个用户的指标差值：ItemCF减热门基线
paired["recall_diff"] = (
    paired["recall_at_10_itemcf"]
    - paired["recall_at_10_baseline"]
)
paired["ndcg_diff"] = (
    paired["ndcg_at_10_itemcf"]
    - paired["ndcg_at_10_baseline"]
)

# 6. 输出读取和配对检查结果
known = paired["train_history_count_baseline"] > 0

print("读取与配对检查通过")
print(f"热门基线用户数：{len(baseline):,}")
print(f"ItemCF用户数：{len(itemcf):,}")
print(f"成功配对用户数：{len(paired):,}")
print(f"新用户数：{(~known).sum():,}")
print(f"已知用户数：{known.sum():,}")

for name, data in [
    ("全部用户", paired),
    ("已知用户", paired.loc[known]),
]:
    print(f"\n{name}")
    print(f"热门基线 Recall@10：{data['recall_at_10_baseline'].mean():.4%}")
    print(f"ItemCF Recall@10：{data['recall_at_10_itemcf'].mean():.4%}")
    print(f"Recall平均差值：{data['recall_diff'].mean() * 100:.4f} 个百分点")
    print(f"NDCG平均差值：{data['ndcg_diff'].mean():.6f}")

print("\n本步骤仅完成数据读取与配对，尚未进行显著性检验。")



# ==========================================================
# 第二部分：统计显著性分析
# 依赖前面已经生成的 paired 数据
# 差值方向统一为：ItemCF + 热门补齐 - 热门基线
# ==========================================================

from datetime import datetime
import json
from scipy import stats

BOOTSTRAP_TIMES = 5000
RANDOM_SEED = 2026
ALPHA = 0.05


def bootstrap_mean_ci(differences, seed):
    """按用户差值进行Bootstrap，返回平均差值的95%置信区间。"""
    differences = np.asarray(differences, dtype=float)

    if len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("差值样本不足或包含非有限值")

    # 相同差值合并为频数，不进行四舍五入，也不丢弃零差值。
    # 对均值而言，此方法与逐用户有放回抽样具有相同的抽样分布，
    # 但不需要创建“5000 × 用户数”的巨大数组。
    values, counts = np.unique(differences, return_counts=True)
    probabilities = counts / counts.sum()
    n = len(differences)

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(BOOTSTRAP_TIMES)

    for i in range(BOOTSTRAP_TIMES):
        sampled_counts = rng.multinomial(n, probabilities)
        bootstrap_means[i] = np.dot(sampled_counts, values) / n

    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values):
    """Holm校正，允许不同检验之间存在依赖。"""
    p_values = np.asarray(p_values, dtype=float)

    if not np.isfinite(p_values).all():
        raise ValueError("检验产生无效p值，请检查数据")

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    m = len(sorted_p)

    adjusted_sorted = np.maximum.accumulate(
        sorted_p * np.arange(m, 0, -1)
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


groups = [
    ("all_positive_users", "全部用户", paired),
    (
        "known_users",
        "已知用户",
        paired.loc[paired["train_history_count_baseline"] > 0],
    ),
]

metrics = [
    ("recall_at_10", "Recall@10", 100.0, "百分点"),
    ("ndcg_at_10", "NDCG@10", 1.0, "原始分值"),
]

results = []

print("\n开始统计显著性分析……", flush=True)

for group_id, group_name, data in groups:
    for metric, metric_name, scale, unit in metrics:
        baseline_values = data[f"{metric}_baseline"].to_numpy(
            dtype=float
        )
        itemcf_values = data[f"{metric}_itemcf"].to_numpy(
            dtype=float
        )
        differences = itemcf_values - baseline_values

        print(
            f"正在分析：{group_name} / {metric_name}",
            flush=True,
        )

        if len(differences) < 2:
            raise ValueError("用户数不足，无法检验")

        # 全部差值为0时不调用t检验，避免产生NaN。
        if np.all(differences == 0):
            t_statistic, p_value = 0.0, 1.0
        elif np.std(differences, ddof=1) == 0:
            raise ValueError("所有差值相同且非零，需单独检查")
        else:
            test = stats.ttest_rel(
                itemcf_values,
                baseline_values,
                alternative="two-sided",
                nan_policy="raise",
            )
            t_statistic = float(test.statistic)
            p_value = float(test.pvalue)

        ci_low, ci_high = bootstrap_mean_ci(
            differences,
            seed=RANDOM_SEED + len(results),
        )

        results.append({
            "group": group_id,
            "group_name": group_name,
            "metric": metric_name,
            "user_count": len(data),
            "baseline_mean_raw": baseline_values.mean(),
            "itemcf_mean_raw": itemcf_values.mean(),
            "mean_difference_raw": differences.mean(),
            "difference_unit": unit,
            "mean_difference": differences.mean() * scale,
            "bootstrap_ci95_low": ci_low * scale,
            "bootstrap_ci95_high": ci_high * scale,
            "t_statistic": t_statistic,
            "p_value_raw": p_value,
        })

result_table = pd.DataFrame(results)
result_table["p_value_holm"] = holm_adjust(
    result_table["p_value_raw"].to_numpy()
)
result_table["significant_holm_005"] = (
    result_table["p_value_holm"] < ALPHA
)

# 保存到独立的新目录，不覆盖前期结果。
result_dir = output_dir / (
    "week6_significance_"
    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
result_dir.mkdir(parents=True, exist_ok=False)

csv_path = result_dir / "significance_results.csv"
result_table.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig",
    float_format="%.15g",
)

settings = {
    "difference_direction": "itemcf_minus_popularity",
    "bootstrap_method": "paired user differences, percentile CI",
    "bootstrap_resamples": BOOTSTRAP_TIMES,
    "base_random_seed": RANDOM_SEED,
    "seed_rule": "base_random_seed + result row index",
    "confidence_level": 0.95,
    "test": "two-sided paired t-test",
    "multiple_testing": "Holm correction across all four tests",
    "alpha": ALPHA,
    "baseline_file": str(baseline_path),
    "itemcf_file": str(itemcf_path),
    "limitations": [
        "用户作为近似独立抽样单位",
        "推断以当前训练模型和数据划分为条件",
        "Bootstrap区间是逐项95%区间，不是多重比较校正后的同时区间",
        "统计显著性不代表实际效果改善或因果关系",
        "浮点计算下p值可能下溢为0，不应在报告中写p=0",
    ],
}
with (result_dir / "analysis_settings.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(settings, file, ensure_ascii=False, indent=2)


def format_p(value):
    # 报告不要将极小p值写成0。
    return "<0.0001" if value < 0.0001 else f"{value:.6f}"


print("\n========== 分析结果 ==========")

for row in result_table.to_dict("records"):
    print(f"\n{row['group_name']} / {row['metric']}")
    print(
        f"平均差值：{row['mean_difference']:.6f} "
        f"{row['difference_unit']}"
    )
    print(
        "Bootstrap 95%置信区间："
        f"[{row['bootstrap_ci95_low']:.6f}, "
        f"{row['bootstrap_ci95_high']:.6f}]"
    )
    print(f"原始p值：{format_p(row['p_value_raw'])}")
    print(f"Holm校正p值：{format_p(row['p_value_holm'])}")

    if row["significant_holm_005"]:
        direction = (
            "低于"
            if row["mean_difference_raw"] < 0
            else "高于"
        )
        print(
            f"结论：当前检验下，ItemCF＋热门补齐显著{direction}热门基线。"
        )
    else:
        print("结论：当前检验未发现统计显著差异。")

print("\n注意：置信区间来自Bootstrap，p值来自配对t检验。")
print("上述结论以当前模型、数据划分及用户近似独立假设为条件。")
print("本步骤尚未完成表现不佳的原因诊断。")
print(f"\n结果已保存至：{result_dir}")




# ==========================================================
# 第三部分：ItemCF表现不佳的量化诊断
# 使用前面已有的 paired、output_dir、result_dir
# ==========================================================

print("\n开始量化诊断……")

history = paired["train_history_count_baseline"]

history_groups = [
    ("0条（新用户）", history == 0),
    ("1条", history == 1),
    ("2～5条", history.between(2, 5)),
    ("6～10条", history.between(6, 10)),
    ("11条及以上", history >= 11),
]

# ---------- 1. 按历史长度比较两种推荐方法 ----------
history_rows = []

for label, mask in history_groups:
    data = paired.loc[mask]
    if data.empty:
        continue

    history_rows.append({
        "历史长度分组": label,
        "用户数": len(data),
        "热门Recall百分比":
            data["recall_at_10_baseline"].mean() * 100,
        "ItemCF加补齐Recall百分比":
            data["recall_at_10_itemcf"].mean() * 100,
        "Recall差值_百分点":
            data["recall_diff"].mean() * 100,
        "热门NDCG":
            data["ndcg_at_10_baseline"].mean(),
        "ItemCF加补齐NDCG":
            data["ndcg_at_10_itemcf"].mean(),
        "NDCG差值":
            data["ndcg_diff"].mean(),
        "完全热门补齐用户占比_百分比":
            data["personalized_count"].eq(0).mean() * 100,
    })

history_table = pd.DataFrame(history_rows)

if history_table["用户数"].sum() != len(paired):
    raise ValueError("历史分组未覆盖全部用户，请检查历史数量")

# ---------- 2. 核对并整理命中来源 ----------
source_path = output_dir / "week3_itemcf_hit_sources.csv"
sources = pd.read_csv(source_path, encoding="utf-8-sig")

if sources["group"].duplicated().any():
    raise ValueError("命中来源文件存在重复分组")

sources = sources.set_index("group")

analysis_groups = [
    ("all_positive_users", "全部用户", paired),
    ("new_users", "新用户", paired.loc[history == 0]),
    ("known_users", "已知用户", paired.loc[history > 0]),
]

source_rows = []

for group_id, label, data in analysis_groups:
    if group_id not in sources.index:
        raise ValueError(f"命中来源文件缺少分组：{group_id}")

    row = sources.loc[group_id]

    # 检查推荐列表长度
    if not (
        data["personalized_count"] + data["fallback_count"]
    ).eq(10).all():
        raise ValueError(f"{label}存在推荐数量不为10的记录")

    # 与当前逐用户结果核对
    checks = {
        "user_count": len(data),
        "personalized_slots": data["personalized_count"].sum(),
        "fallback_slots": data["fallback_count"].sum(),
        "total_hits": data["hits_at_10_itemcf"].sum(),
    }

    for field, expected in checks.items():
        if row[field] != expected:
            raise ValueError(
                f"{label}的{field}与逐用户结果不一致，"
                "请检查是否混用了不同批次文件"
            )

    personal_slots = int(row["personalized_slots"])
    fallback_slots = int(row["fallback_slots"])
    personal_hits = int(row["personalized_hits"])
    fallback_hits = int(row["fallback_hits"])
    total_hits = int(row["total_hits"])

    if personal_hits + fallback_hits != total_hits:
        raise ValueError(f"{label}的两类命中数之和不正确")

    for source_name, slots, hits in [
        ("ItemCF个性化", personal_slots, personal_hits),
        ("热门补齐", fallback_slots, fallback_hits),
    ]:
        if not 0 <= hits <= slots:
            raise ValueError(f"{label}的{source_name}命中数异常")

        source_rows.append({
            "用户分组": label,
            "推荐来源": source_name,
            "用户数": len(data),
            "推荐位置数": slots,
            "位置占比_百分比":
                slots / (len(data) * 10) * 100,
            "命中数": hits,
            "每100个推荐位置的命中数":
                hits / slots * 100 if slots else np.nan,
            "占本组全部命中的比例_百分比":
                hits / total_hits * 100 if total_hits else np.nan,
        })

source_table = pd.DataFrame(source_rows)

# ---------- 3. 新商品目标占比 ----------
target_rows = []

for group_id, label, data in analysis_groups:
    total = data["positive_target_count_baseline"]
    known_targets = data["known_product_target_count_baseline"]

    if (
        (total <= 0).any()
        or (known_targets < 0).any()
        or (known_targets > total).any()
    ):
        raise ValueError(f"{label}的目标数量异常")

    total_count = int(total.sum())
    known_count = int(known_targets.sum())
    new_count = total_count - known_count

    target_rows.append({
        "用户分组": label,
        "用户数": len(data),
        "测试正反馈目标总数": total_count,
        "训练期已出现商品目标数": known_count,
        "训练期未出现商品目标数": new_count,
        "新商品目标占比_百分比":
            new_count / total_count * 100,
        "全部目标均为新商品的用户数":
            int(known_targets.eq(0).sum()),
    })

target_table = pd.DataFrame(target_rows)

# ---------- 4. 保存三张表 ----------
tables = [
    ("diagnostic_history_groups.csv", history_table),
    ("diagnostic_hit_sources.csv", source_table),
    ("diagnostic_new_item_targets.csv", target_table),
]

for filename, table in tables:
    table.to_csv(
        result_dir / filename,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

# 保存解释口径，防止把相关性写成因果关系
notes = [
    "本分析仅复用已有结果，没有重新训练或调整模型。",
    "历史长度包含训练期全部星级评分，不仅是正反馈。",
    "Recall差值单位为百分点；NDCG差值为原始分值。",
    "每100个推荐位置的命中数不是Recall@10。",
    "来源分析中的热门补齐是混合方案的一部分，不是独立热门基线。",
    "两类推荐来源的用户构成和排名位置不同，不能直接作因果比较。",
    "目标数量按用户—商品目标累计，不是全局去重商品数。",
    "新商品候选限制同时影响两种方法，不能单独解释方法间差距。",
    "全部用户包含新用户与已知用户，三个分组不可再次相加。",
    "本诊断尚未证明邻居质量或其他因素是性能下降的原因。",
]

with (result_dir / "diagnostic_notes.txt").open(
    "w", encoding="utf-8"
) as file:
    file.write("\n".join(notes))

# ---------- 5. 显示结果 ----------
for title, table in [
    ("历史长度分组", history_table),
    ("推荐命中来源", source_table),
    ("新商品目标占比", target_table),
]:
    print(f"\n========== {title} ==========")
    print(table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
        na_rep="不适用",
    ))

print("\n量化诊断完成。")
print(f"结果文件夹：{result_dir}")
print("请结合诊断说明解释结果，不将分组差异直接视为因果关系。")