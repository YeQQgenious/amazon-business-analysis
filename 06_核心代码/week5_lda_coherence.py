from pathlib import Path
from datetime import datetime
import csv
import json
import platform

import numpy as np
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 固定使用之前完整运行的结果，不自动寻找其他文件夹
BASE = Path("/Users/ethen/CS/Amazon/mydata")
SOURCE = BASE / "outputs/week5_lda_selection_20261124_005414_842014"

TOP_N = 12
WINDOW_SIZE = 110


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # 读取原实验设置
    with (SOURCE / "settings.json").open(
        "r", encoding="utf-8"
    ) as f:
        original_settings = json.load(f)

    topic_counts = original_settings["topic_counts"]
    seeds = original_settings["model_seeds"]

    raw_rows = read_csv(Path(original_settings["input_file"]))
    text_by_id = {}

    for row in raw_rows:
        text_id = row["clean_text_id"].strip()
        if text_id in text_by_id:
            raise ValueError(f"重复文本编号：{text_id}")
        text_by_id[text_id] = row["review_content_clean"].strip()

    # 使用原来保存的划分，不重新随机分组
    split_rows = read_csv(SOURCE / "data_split.csv")
    train_ids = [
        row["clean_text_id"]
        for row in split_rows
        if row["split"] == "train"
        and row["included"].strip().lower() == "true"
    ]

    if len(train_ids) != original_settings["train_rows_used"]:
        raise ValueError("训练集记录数与原设置不一致，请检查文件。")

    if len(train_ids) != len(set(train_ids)):
        raise ValueError("训练集划分中存在重复编号。")

    vocabulary = {
        row["word"] for row in read_csv(SOURCE / "vocabulary.csv")
    }

    # 沿用原来的分词和停用词规则，并限制在原词表内
    analyzer = CountVectorizer(
        **original_settings["vectorizer"]
    ).build_analyzer()

    tokenized_texts = [
        [
            word
            for word in analyzer(text_by_id[text_id])
            if word in vocabulary
        ]
        for text_id in train_ids
    ]

    if any(not tokens for tokens in tokenized_texts):
        raise ValueError("参考文本出现空记录，请检查源数据是否变化。")

    dictionary = Dictionary(tokenized_texts)

    # 先检查所有关键词文件，避免漏算某个模型
    topic_sets = {}

    for k in topic_counts:
        for seed in seeds:
            path = SOURCE / f"k{k}_seed{seed}_keywords.csv"
            rows = sorted(
                read_csv(path),
                key=lambda row: int(row["topic_id"]),
            )

            if [int(row["topic_id"]) for row in rows] != list(
                range(1, k + 1)
            ):
                raise ValueError(f"{path.name} 的主题编号不完整。")

            topics = []

            for row in rows:
                words = [
                    word.strip()
                    for word in row["keywords"].split(",")
                    if word.strip()
                ]

                if len(words) != TOP_N or len(set(words)) != TOP_N:
                    raise ValueError(
                        f"{path.name} 某主题不是12个不同关键词。"
                    )

                missing = [
                    word for word in words
                    if word not in dictionary.token2id
                ]
                if missing:
                    raise ValueError(
                        f"{path.name} 关键词不在参考词表：{missing}"
                    )

                topics.append(words)

            topic_sets[(k, seed)] = topics

    # 新建结果文件夹，不覆盖旧结果
    out = BASE / "outputs" / (
        "week5_lda_coherence_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    out.mkdir(parents=True, exist_ok=False)

    run_rows = []
    topic_rows = []
    summary_rows = []

    print(f"参考训练文本数：{len(tokenized_texts)}")
    print(f"每个主题关键词数：{TOP_N}")
    print(f"C_v窗口大小：{WINDOW_SIZE}")
    print("开始计算一致性，不重新训练LDA。", flush=True)

    for k in topic_counts:
        values = []

        for seed in seeds:
            topics = topic_sets[(k, seed)]

            evaluator = CoherenceModel(
                topics=topics,
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence="c_v",
                topn=TOP_N,
                window_size=WINDOW_SIZE,
                processes=1,
            )

            per_topic = np.asarray(
                evaluator.get_coherence_per_topic(),
                dtype=float,
            )

            if not np.isfinite(per_topic).all():
                raise ValueError(
                    f"K={k}, seed={seed}产生无效得分，请停止核查。"
                )

            mean_score = float(per_topic.mean())
            values.append(mean_score)

            run_rows.append({
                "n_topics": k,
                "seed": seed,
                "cv_mean_over_topics": mean_score,
            })

            for topic_id, score in enumerate(per_topic, start=1):
                topic_rows.append({
                    "n_topics": k,
                    "seed": seed,
                    "topic_id": topic_id,
                    "cv_score": float(score),
                    "keywords": ", ".join(topics[topic_id - 1]),
                })

            print(
                f"K={k}, seed={seed}, C_v={mean_score:.4f}",
                flush=True,
            )

        summary_rows.append({
            "n_topics": k,
            "cv_mean": float(np.mean(values)),
            "cv_std": float(np.std(values, ddof=1)),
        })

        # 每完成一个主题数就保存进度
        save_csv(
            out / "coherence_runs.csv",
            ["n_topics", "seed", "cv_mean_over_topics"],
            run_rows,
        )
        save_csv(
            out / "coherence_per_topic.csv",
            ["n_topics", "seed", "topic_id", "cv_score", "keywords"],
            topic_rows,
        )
        save_csv(
            out / "coherence_summary.csv",
            ["n_topics", "cv_mean", "cv_std"],
            summary_rows,
        )

    # 将一致性加入原比较表的新副本，保留人工备注
    original_comparison = read_csv(SOURCE / "model_comparison.csv")
    summary_by_k = {
        row["n_topics"]: row for row in summary_rows
    }

    if sorted(int(r["n_topics"]) for r in original_comparison) != sorted(
        topic_counts
    ):
        raise ValueError("原比较表的主题数范围不完整。")

    for row in original_comparison:
        summary = summary_by_k[int(row["n_topics"])]
        row["cv_mean"] = summary["cv_mean"]
        row["cv_std"] = summary["cv_std"]

    save_csv(
        out / "model_comparison_with_coherence.csv",
        list(original_comparison[0]),
        original_comparison,
    )

    # 绘制一致性曲线
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        [r["n_topics"] for r in summary_rows],
        [r["cv_mean"] for r in summary_rows],
        yerr=[r["cv_std"] for r in summary_rows],
        fmt="o-",
        capsize=4,
        color="#2878B5",
    )

    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("C_v coherence (higher is better)")
    ax.set_title("Topic coherence: mean +/- SD across 3 seeds")
    ax.set_xticks(topic_counts)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "coherence_curve.png", dpi=200)
    plt.close(fig)

    settings = {
        "source_experiment": str(SOURCE),
        "metric": "c_v",
        "reference_corpus": "original training split only",
        "reference_text_count": len(tokenized_texts),
        "top_n": TOP_N,
        "window_size": WINDOW_SIZE,
        "topic_counts": topic_counts,
        "model_seeds": seeds,
        "preprocessing": original_settings["vectorizer"],
        "gensim_version": gensim.__version__,
        "python_version": platform.python_version(),
        "notes": [
            "读取已保存关键词，不重新训练模型。",
            "每个模型的得分为各主题得分的算术平均。",
            "每个K再对三个模型得分计算均值和样本标准差。",
            "标准差反映不同随机种子间的得分变化，不是置信区间。",
            "参考文本仅使用原训练集，不是独立测试集评价。",
            "C_v不是准确率，也不直接衡量不同主题之间的重复。",
            "文本拼接和通用词可能影响一致性评价。",
        ],
    }

    with (out / "coherence_settings.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

    print("\n计算完成，结果文件夹：")
    print(out)
    print("请结合关键词和代表文本解释，不要只按最高分选择K。")


if __name__ == "__main__":
    main()