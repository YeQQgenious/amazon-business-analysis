from pathlib import Path
from datetime import datetime
import csv
import json
import platform

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# ==================== 设置 ====================

BASE = Path("/Users/ethen/CS/Amazon/mydata")
INPUT = BASE / "outputs/review_clean_unique_texts.csv"

TOPIC_COUNTS = list(range(2, 11))
MODEL_SEEDS = [42, 123, 2026]
SPLIT_SEED = 42
VALIDATION_SIZE = 0.20
MAX_ITER = 50

OUT = BASE / "outputs" / (
    "week5_lda_selection_"
    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)

VECTORIZER_SETTINGS = {
    "lowercase": True,
    "stop_words": "english",
    "token_pattern": r"(?u)\b[a-zA-Z]{3,}\b",
    "min_df": 5,
    "max_df": 0.90,
    "max_features": 5000,
}


def save_csv(filename, fields, records):
    with (OUT / filename).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


# ==================== 读取与检查 ====================

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    required = {"clean_text_id", "review_content_clean"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("输入文件缺少预期字段，请检查表头。")

    raw_rows = list(reader)

records = [
    {
        "clean_text_id": row["clean_text_id"].strip(),
        "review_content_clean": row["review_content_clean"].strip(),
    }
    for row in raw_rows
    if row["review_content_clean"].strip()
]

ids = [r["clean_text_id"] for r in records]
texts = [r["review_content_clean"] for r in records]

if any(not item for item in ids):
    raise ValueError("存在空文本编号，请先检查。")

if len(ids) != len(set(ids)):
    raise ValueError("存在重复文本编号，请先检查。")

if len(texts) != len(set(texts)):
    raise ValueError("存在完全重复文本，请先检查，避免跨集合泄漏。")

if len(records) < 50:
    raise ValueError("有效文本数量过少，请检查输入文件。")

OUT.mkdir(parents=True, exist_ok=False)

print(f"原始记录数：{len(raw_rows)}")
print(f"非空文本数：{len(records)}")


# ==================== 固定训练/验证划分 ====================

indices = np.arange(len(records))

train_idx, val_idx = train_test_split(
    indices,
    test_size=VALIDATION_SIZE,
    random_state=SPLIT_SEED,
    shuffle=True,
)

train_texts = [texts[i] for i in train_idx]
val_texts = [texts[i] for i in val_idx]

# 重要：词表只在训练集上建立
vectorizer = CountVectorizer(**VECTORIZER_SETTINGS)

X_train_raw = vectorizer.fit_transform(train_texts)
X_val_raw = vectorizer.transform(val_texts)

words = vectorizer.get_feature_names_out()

# 删除转换后没有有效词的记录，并保存排除名单
train_keep = np.asarray(X_train_raw.sum(axis=1)).ravel() > 0
val_keep = np.asarray(X_val_raw.sum(axis=1)).ravel() > 0

split_rows = []

for subset, subset_idx, keep_mask in [
    ("train", train_idx, train_keep),
    ("validation", val_idx, val_keep),
]:
    for index, keep in zip(subset_idx, keep_mask):
        split_rows.append({
            "clean_text_id": records[index]["clean_text_id"],
            "split": subset,
            "included": bool(keep),
            "exclusion_reason": "" if keep else "no_retained_tokens",
        })

save_csv(
    "data_split.csv",
    ["clean_text_id", "split", "included", "exclusion_reason"],
    split_rows,
)

train_valid_idx = train_idx[train_keep]
val_valid_idx = val_idx[val_keep]

X_train = X_train_raw[train_keep]
X_val = X_val_raw[val_keep]

if X_train.shape[0] < 10 or X_val.shape[0] == 0:
    raise ValueError("预处理后有效文本不足，请检查词表设置。")

print(f"训练集有效文本数：{X_train.shape[0]}")
print(f"验证集有效文本数：{X_val.shape[0]}")
print(f"训练集排除数：{int((~train_keep).sum())}")
print(f"验证集排除数：{int((~val_keep).sum())}")
print(f"词表大小：{len(words)}")
print("开始比较主题数，共训练27个模型。", flush=True)


# ==================== 模型比较 ====================

run_results = []
summary_results = []

for k in TOPIC_COUNTS:
    print(f"\n===== 主题数 K={k} =====", flush=True)
    k_runs = []

    for seed in MODEL_SEEDS:
        print(f"正在运行 seed={seed}……", flush=True)

        model = LatentDirichletAllocation(
            n_components=k,
            learning_method="batch",
            max_iter=MAX_ITER,
            evaluate_every=-1,
            random_state=seed,
            n_jobs=1,

            # 固定先验，避免默认先验随K一起改变
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
        )

        model.fit(X_train)

        train_perplexity = float(model.perplexity(X_train))
        val_perplexity = float(model.perplexity(X_val))

        if not np.isfinite([train_perplexity, val_perplexity]).all():
            raise ValueError(f"K={k}, seed={seed}产生无效困惑度。")

        result = {
            "n_topics": k,
            "seed": seed,
            "train_perplexity": train_perplexity,
            "validation_perplexity": val_perplexity,
            "iterations": int(model.n_iter_),
        }

        run_results.append(result)
        k_runs.append(result)

        print(
            f"训练集困惑度：{train_perplexity:.2f}；"
            f"验证集困惑度：{val_perplexity:.2f}",
            flush=True,
        )

        # 每次运行都保存关键词，方便检查随机初始化影响
        keyword_rows = []

        for topic in range(k):
            weights = model.components_[topic]
            top = weights.argsort()[::-1][:12]

            keyword_rows.append({
                "topic_id": topic + 1,
                "keywords": ", ".join(words[top]),
                "manual_topic_name": "",
            })

        save_csv(
            f"k{k}_seed{seed}_keywords.csv",
            ["topic_id", "keywords", "manual_topic_name"],
            keyword_rows,
        )

        # 固定用seed=42导出人工核查材料，
        # 不挑选困惑度最低的一次作为展示结果
        if seed == MODEL_SEEDS[0]:
            probabilities = model.transform(X_val)
            dominant = probabilities.argmax(axis=1)

            examples = []
            assignments = []

            for topic in range(k):
                members = np.flatnonzero(dominant == topic)
                ranked = members[
                    np.argsort(
                        -probabilities[members, topic],
                        kind="stable",
                    )
                ][:3]

                for rank, local_index in enumerate(ranked, start=1):
                    source_index = val_valid_idx[local_index]
                    record = records[source_index]

                    examples.append({
                        "topic_id": topic + 1,
                        "rank": rank,
                        "clean_text_id": record["clean_text_id"],
                        "topic_probability": float(
                            probabilities[local_index, topic]
                        ),
                        "review_content_clean":
                            record["review_content_clean"],
                        "manual_topic_fit": "",
                        "manual_notes": "",
                    })

            for local_index, source_index in enumerate(val_valid_idx):
                item = {
                    "clean_text_id":
                        records[source_index]["clean_text_id"],
                    "dominant_topic":
                        int(dominant[local_index]) + 1,
                }

                for topic in range(k):
                    item[f"topic_{topic + 1}_probability"] = float(
                        probabilities[local_index, topic]
                    )

                assignments.append(item)

            save_csv(
                f"k{k}_seed{seed}_validation_examples.csv",
                [
                    "topic_id", "rank", "clean_text_id",
                    "topic_probability", "review_content_clean",
                    "manual_topic_fit", "manual_notes",
                ],
                examples,
            )

            save_csv(
                f"k{k}_seed{seed}_validation_topics.csv",
                ["clean_text_id", "dominant_topic"]
                + [
                    f"topic_{topic + 1}_probability"
                    for topic in range(k)
                ],
                assignments,
            )

    train_values = [r["train_perplexity"] for r in k_runs]
    val_values = [r["validation_perplexity"] for r in k_runs]

    summary_results.append({
        "n_topics": k,
        "train_perplexity_mean": float(np.mean(train_values)),
        "validation_perplexity_mean": float(np.mean(val_values)),
        "validation_perplexity_std": float(
            np.std(val_values, ddof=1)
        ),
        "manual_interpretability": "",
        "manual_topic_overlap": "",
        "manual_decision": "",
    })

    # 每完成一个K保存一次，方便保留进度
    save_csv(
        "perplexity_runs.csv",
        [
            "n_topics", "seed", "train_perplexity",
            "validation_perplexity", "iterations",
        ],
        run_results,
    )

    save_csv(
        "model_comparison.csv",
        [
            "n_topics", "train_perplexity_mean",
            "validation_perplexity_mean",
            "validation_perplexity_std",
            "manual_interpretability",
            "manual_topic_overlap", "manual_decision",
        ],
        summary_results,
    )


# ==================== 保存设置与词表 ====================

save_csv(
    "vocabulary.csv",
    ["word"],
    [{"word": word} for word in words],
)

settings = {
    "input_file": str(INPUT),
    "input_rows": len(raw_rows),
    "nonempty_rows": len(records),
    "train_rows_before_filter": len(train_idx),
    "validation_rows_before_filter": len(val_idx),
    "train_rows_used": int(X_train.shape[0]),
    "validation_rows_used": int(X_val.shape[0]),
    "vocabulary_size": len(words),
    "validation_size": VALIDATION_SIZE,
    "split_seed": SPLIT_SEED,
    "topic_counts": TOPIC_COUNTS,
    "model_seeds": MODEL_SEEDS,
    "max_iter": MAX_ITER,
    "doc_topic_prior": 0.1,
    "topic_word_prior": 0.01,
    "vectorizer": VECTORIZER_SETTINGS,
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "numpy_version": np.__version__,
    "notes": [
        "词表仅在训练集拟合，验证集使用同一词表转换。",
        "验证集仅用于模型选择，不是独立最终测试集。",
        "标准差仅反映三个随机种子间的变化。",
        "不同运行的主题编号没有一一对应关系。",
        "分析单位是文本记录，可能包含多条评论拼接。",
        "去除完全重复文本不能排除近重复或共享片段。",
        "代表文本按主主题内的主题概率排序，需人工核查。",
        "不单凭最低困惑度或曲线拐点决定最终主题数。",
    ],
}

with (OUT / "settings.json").open("w", encoding="utf-8") as f:
    json.dump(settings, f, ensure_ascii=False, indent=2)


# ==================== 绘制曲线 ====================

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\n未安装matplotlib，数据结果已保存，但未生成曲线。")
    print("可以用model_comparison.csv在Excel中绘图。")
else:
    ks = [r["n_topics"] for r in summary_results]
    train_means = [
        r["train_perplexity_mean"] for r in summary_results
    ]
    val_means = [
        r["validation_perplexity_mean"] for r in summary_results
    ]
    val_stds = [
        r["validation_perplexity_std"] for r in summary_results
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        ks, train_means, "o--",
        label="Training mean", color="#888888",
    )
    axes[0].plot(
        ks, val_means, "o-",
        label="Validation mean", color="#2878B5",
    )
    axes[0].set_title("Training and validation perplexity")
    axes[0].legend()

    axes[1].errorbar(
        ks, val_means, yerr=val_stds,
        fmt="o-", capsize=4, color="#2878B5",
    )
    axes[1].set_title("Validation mean +/- SD (3 seeds)")

    for ax in axes:
        ax.set_xlabel("Number of topics (K)")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_xticks(ks)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "perplexity_curve.png", dpi=200)
    plt.close(fig)


print("\n全部完成！结果文件夹：")
print(OUT)
print("\n下一步：查看perplexity_curve.png和model_comparison.csv。")
print("不要仅凭困惑度最低值，直接确定最佳主题数。")