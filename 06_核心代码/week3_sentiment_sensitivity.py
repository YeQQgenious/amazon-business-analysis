from pathlib import Path
from importlib.metadata import version
import json

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


output_dir = Path("/Users/ethen/CS/Amazon/mydata/outputs")
input_path = output_dir / "review_sensitivity_weights.csv"

if not input_path.exists():
    raise FileNotFoundError(
        f"找不到三组权重文件：{input_path}。"
        "请先检查文件，不要改用词频表。"
    )

df = pd.read_csv(
    input_path,
    dtype={
        "clean_text_id": "string",
        "review_content_clean": "string"
    },
    keep_default_na=False
)

required = {
    "clean_text_id",
    "review_content_clean",
    "weight_A",
    "weight_B",
    "weight_C"
}

if not required.issubset(df.columns):
    raise ValueError(f"输入缺少字段：{required - set(df.columns)}")

if len(df) != 1194 or df["clean_text_id"].duplicated().any():
    raise ValueError("清洗文本数量或编号唯一性不一致。")

if df["review_content_clean"].str.strip().eq("").any():
    raise ValueError("存在空白正文。")

expected_totals = {"A": 1465, "B": 1212, "C": 1194}

for group, expected in expected_totals.items():
    column = f"weight_{group}"
    weights = pd.to_numeric(df[column], errors="raise")

    if (
        not np.isfinite(weights).all()
        or (weights <= 0).any()
        or (weights % 1 != 0).any()
        or int(weights.sum()) != expected
    ):
        raise ValueError(f"{group} 组权重核对失败。")

    df[column] = weights.astype("int64")

if not df["weight_C"].eq(1).all():
    raise ValueError("C 组每段唯一清洗正文应只计一次。")


# 1. 使用完整清洗正文，不去停用词，不转为词语列表
analyzer = SentimentIntensityAnalyzer()

scores = []

print("正在计算 1,194 段清洗正文的情感分数……")

for number, text in enumerate(df["review_content_clean"], start=1):
    result = analyzer.polarity_scores(text)
    scores.append(result["compound"])

    if number % 200 == 0:
        print(f"已处理 {number} 段……")

df["compound"] = scores
df["text_char_count"] = df["review_content_clean"].str.len()

# VADER 常用分类阈值
# compound 是情感倾向分数，不是概率
df["sentiment_label"] = np.select(
    [
        df["compound"] >= 0.05,
        df["compound"] <= -0.05
    ],
    [
        "positive",
        "negative"
    ],
    default="neutral"
)


# 2. 按三组权重分别汇总
summary_rows = []

for group in ["A", "B", "C"]:
    weights = df[f"weight_{group}"].to_numpy()
    total = int(weights.sum())

    positive_count = int(
        weights[df["sentiment_label"].eq("positive")].sum()
    )
    neutral_count = int(
        weights[df["sentiment_label"].eq("neutral")].sum()
    )
    negative_count = int(
        weights[df["sentiment_label"].eq("negative")].sum()
    )

    if positive_count + neutral_count + negative_count != total:
        raise ValueError("情感分类数量合计不一致。")

    summary_rows.append({
        "group": group,
        "weighted_text_count": total,
        "mean_compound": np.average(
            df["compound"], weights=weights
        ),
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "positive_pct": positive_count / total * 100,
        "neutral_pct": neutral_count / total * 100,
        "negative_pct": negative_count / total * 100,

        # 检查分数是否大量接近边界值 ±1
        "extreme_score_pct": np.average(
            df["compound"].abs() >= 0.95,
            weights=weights
        ) * 100
    })

summary = pd.DataFrame(summary_rows).set_index("group")


# 3. 输出组间变化
difference_rows = []

for before, after in [("A", "B"), ("B", "C"), ("A", "C")]:
    difference_rows.append({
        "comparison": f"{after}_minus_{before}",
        "mean_compound_difference": (
            summary.loc[after, "mean_compound"]
            - summary.loc[before, "mean_compound"]
        ),
        "positive_difference_pp": (
            summary.loc[after, "positive_pct"]
            - summary.loc[before, "positive_pct"]
        ),
        "neutral_difference_pp": (
            summary.loc[after, "neutral_pct"]
            - summary.loc[before, "neutral_pct"]
        ),
        "negative_difference_pp": (
            summary.loc[after, "negative_pct"]
            - summary.loc[before, "negative_pct"]
        )
    })

differences = pd.DataFrame(difference_rows)


# 4. 保存评分及汇总
df.to_csv(
    output_dir / "week3_sentiment_text_scores.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.reset_index().to_csv(
    output_dir / "week3_sentiment_group_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

differences.to_csv(
    output_dir / "week3_sentiment_group_differences.csv",
    index=False,
    encoding="utf-8-sig"
)


# 5. 导出人工核查材料
# 每种标签抽取最多3段，再补充临界分数和长文本
samples = []

for label in ["positive", "neutral", "negative"]:
    pool = df[df["sentiment_label"] == label]

    if not pool.empty:
        samples.append(
            pool.sample(n=min(3, len(pool)), random_state=42)
        )

boundary_distance = np.minimum(
    (df["compound"] - 0.05).abs(),
    (df["compound"] + 0.05).abs()
)

samples.append(df.loc[boundary_distance.nsmallest(3).index])
samples.append(df.nlargest(3, "text_char_count"))

review_sample = (
    pd.concat(samples)
    .drop_duplicates("clean_text_id")
)

with (output_dir / "week3_sentiment_manual_check.txt").open(
    "w", encoding="utf-8"
) as file:
    for _, row in review_sample.iterrows():
        file.write(
            f"\n{'=' * 60}\n"
            f"文本编号：{row['clean_text_id']}\n"
            f"自动标签：{row['sentiment_label']}\n"
            f"compound：{row['compound']}\n"
            f"字符数：{row['text_char_count']}\n"
            f"A/B/C权重：{row['weight_A']}/"
            f"{row['weight_B']}/{row['weight_C']}\n\n"
            f"{row['review_content_clean']}\n"
        )

settings = {
    "method": "VADER whole-cleaned-text baseline",
    "vader_version": version("vaderSentiment"),
    "positive_threshold": 0.05,
    "negative_threshold": -0.05,
    "analysis_unit": "concatenated cleaned text, not individual review",
    "unique_clean_text_count": len(df),
    "expected_group_totals": expected_totals
}

with (output_dir / "week3_sentiment_settings.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(settings, file, ensure_ascii=False, indent=2)


# 6. 显示结果
print("\n三组情感分析初步结果：")
print(summary.round(4).to_string())

print("\n组间变化：")
print(differences.round(4).to_string(index=False))

print("\n说明：pp 表示占比变化的百分点。")
print("extreme_score_pct 表示 |compound| >= 0.95 的文本占比。")
print("结果和人工核查材料已保存，原始数据未修改。")