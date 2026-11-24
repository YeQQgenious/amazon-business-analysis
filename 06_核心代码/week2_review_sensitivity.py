from pathlib import Path
import pandas as pd


output_dir = Path("/Users/ethen/CS/Amazon/mydata/outputs")

# 1. 读取清洗后的唯一文本及其追溯关系
clean_texts = pd.read_csv(
    output_dir / "review_clean_unique_texts.csv",
    dtype="string",
    keep_default_na=False
)

text_map = pd.read_csv(
    output_dir / "review_text_id_map.csv",
    dtype="string",
    keep_default_na=False
)

product_map = pd.read_csv(
    output_dir / "product_clean_text_map.csv",
    dtype="string",
    keep_default_na=False
)

# 2. 检查用于连接的编号
if clean_texts["clean_text_id"].duplicated().any():
    raise ValueError("清洗文本表中的 clean_text_id 不唯一。")

if text_map["text_id"].duplicated().any():
    raise ValueError("原始文本映射表中的 text_id 不唯一。")

if clean_texts["review_content_clean"].str.strip().eq("").any():
    raise ValueError("存在空白的清洗正文。")

# 检查两张映射表中的清洗文本编号能否找到
known_ids = set(clean_texts["clean_text_id"])

if not set(text_map["clean_text_id"]).issubset(known_ids):
    raise ValueError("原始文本映射中存在找不到的清洗文本编号。")

if not set(product_map["clean_text_id"]).issubset(known_ids):
    raise ValueError("商品映射中存在找不到的清洗文本编号。")


# 3. A 组：按原始记录数量计算权重
record_counts = pd.to_numeric(
    product_map["source_record_count"],
    errors="raise"
)

if (
    record_counts.isna().any()
    or (record_counts <= 0).any()
    or (record_counts % 1 != 0).any()
):
    raise ValueError("原始记录数量应为正整数。")

product_map["source_record_count"] = record_counts.astype("int64")

weight_a = (
    product_map.groupby("clean_text_id")["source_record_count"]
    .sum()
)


# 4. B 组：每个原始文本编号计一次
# 不同原始文本可能在清洗后变成同一正文
weight_b = text_map.groupby("clean_text_id").size()


# 5. C 组：每个清洗后的唯一正文计一次
comparison = clean_texts[
    ["clean_text_id", "review_content_clean"]
].copy()

comparison["weight_A"] = (
    comparison["clean_text_id"].map(weight_a).fillna(0).astype("int64")
)

comparison["weight_B"] = (
    comparison["clean_text_id"].map(weight_b).fillna(0).astype("int64")
)

comparison["weight_C"] = 1

if (comparison[["weight_A", "weight_B"]] <= 0).any().any():
    raise ValueError("存在无法追溯到原始数据的清洗文本。")


# 6. 核对三组数量
summary = pd.DataFrame({
    "group": ["A_original_records", "B_unique_raw_texts", "C_unique_clean_texts"],
    "text_count": [
        comparison["weight_A"].sum(),
        comparison["weight_B"].sum(),
        comparison["weight_C"].sum()
    ]
})

print("三组对照数据规模：")
print(summary.to_string(index=False))

expected = [1465, 1212, 1194]

if summary["text_count"].tolist() != expected:
    raise ValueError("三组数量与此前结果不一致，请先检查映射文件。")


# 7. 保存供后续词频和情感分析使用
comparison.to_csv(
    output_dir / "review_sensitivity_weights.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    output_dir / "review_sensitivity_group_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n数量核对通过，已保存三组分析权重。")



import re
from collections import Counter
from itertools import combinations


# 8. 设置三组共用的分词规则和停用词
# 停用词：本次词频分析中排除的常见功能词
# 这是一个固定的基础词表，不代表完整的英文停用词表
stop_words = set("""
a an the and or but if while
of at by for with about into through during
to from in out on off over under
is am are was were be been being
have has had having do does did doing
i me my myself we our ours ourselves
you your yours yourself yourselves
he him his himself she her hers herself
it its itself they them their theirs themselves
this that these those
as until than then once here there
when where why how
all any both each few more most other some such
only own same so too very
can will just should now
""".split())


def tokenize(text):
    # 保留英文单词及单词内部的英文撇号
    text = str(text).lower().replace("’", "'")
    words = re.findall(r"[a-z]+(?:'[a-z]+)*", text)

    return [
        word for word in words
        if len(word) >= 2 and word not in stop_words
    ]


# 每段清洗文本只分词一次，三组共用同一结果
token_counts = [
    Counter(tokenize(text))
    for text in comparison["review_content_clean"]
]

group_columns = {
    "A": "weight_A",
    "B": "weight_B",
    "C": "weight_C"
}

frequency_tables = {}
top20_lists = {}
token_totals = {}


# 9. 根据三组权重分别统计词频
for group, weight_column in group_columns.items():
    word_counts = Counter()

    for text_counts, weight in zip(
        token_counts,
        comparison[weight_column]
    ):
        for word, count in text_counts.items():
            word_counts[word] += count * int(weight)

    total_tokens = sum(word_counts.values())

    if total_tokens == 0:
        raise ValueError(f"{group} 组没有可统计的英文词。")

    token_totals[group] = total_tokens

    # 次数相同时按字母排序，保证结果可复现
    ordered_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    table = pd.DataFrame(
        ordered_words,
        columns=["word", "count"]
    )

    table["rank"] = range(1, len(table) + 1)

    # 分母是本组分词、去停用词后保留的全部词语出现次数
    table["relative_freq_pct"] = (
        table["count"] / total_tokens * 100
    )

    frequency_tables[group] = table
    top20_lists[group] = table.head(20)["word"].tolist()

    table.to_csv(
        output_dir / f"review_word_frequency_{group}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n{group} 组前 20 个高频词：")
    print(
        table.head(20)[
            ["rank", "word", "count", "relative_freq_pct"]
        ].round(4).to_string(index=False)
    )


# 10. 比较三组前 20 个高频词的重合程度
overlap_rows = []

for group_1, group_2 in combinations(group_columns, 2):
    words_1 = set(top20_lists[group_1])
    words_2 = set(top20_lists[group_2])

    common_count = len(words_1 & words_2)
    denominator = min(len(words_1), len(words_2))

    overlap_rows.append({
        "comparison": f"{group_1}_vs_{group_2}",
        "common_word_count": common_count,
        "top_word_overlap_pct": (
            common_count / denominator * 100
        )
    })

overlap_summary = pd.DataFrame(overlap_rows)

print("\n前 20 个高频词的重合情况：")
print(overlap_summary.round(2).to_string(index=False))

overlap_summary.to_csv(
    output_dir / "review_top20_overlap.csv",
    index=False,
    encoding="utf-8-sig"
)


# 11. 汇总三组前 20 个词的并集，比较排名和相对词频
selected_words = sorted(
    set().union(*(set(words) for words in top20_lists.values()))
)

word_comparison = pd.DataFrame({"word": selected_words})

for group, table in frequency_tables.items():
    selected_columns = table[
        ["word", "rank", "relative_freq_pct"]
    ].rename(columns={
        "rank": f"rank_{group}",
        "relative_freq_pct": f"freq_pct_{group}"
    })

    word_comparison = word_comparison.merge(
        selected_columns,
        on="word",
        how="left",
        validate="one_to_one"
    )

# 相对词频之差，单位是“百分点”
word_comparison["B_minus_A_pp"] = (
    word_comparison["freq_pct_B"] - word_comparison["freq_pct_A"]
)

word_comparison["C_minus_B_pp"] = (
    word_comparison["freq_pct_C"] - word_comparison["freq_pct_B"]
)

word_comparison["C_minus_A_pp"] = (
    word_comparison["freq_pct_C"] - word_comparison["freq_pct_A"]
)

# 排名数字越小越靠前；正值表示 C 组排名比 A 组下降
word_comparison["rank_change_C_minus_A"] = (
    word_comparison["rank_C"] - word_comparison["rank_A"]
)

word_comparison = word_comparison.sort_values("rank_A")

word_comparison.to_csv(
    output_dir / "review_top_words_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

largest_changes = (
    word_comparison.assign(
        absolute_change=word_comparison["C_minus_A_pp"].abs()
    )
    .sort_values("absolute_change", ascending=False)
    .head(10)
)

print("\n候选高频词中，A 与 C 相对词频变化最大的 10 个词：")
print(
    largest_changes[
        [
            "word",
            "rank_A",
            "rank_C",
            "freq_pct_A",
            "freq_pct_C",
            "C_minus_A_pp"
        ]
    ].round(4).to_string(index=False)
)

print("\n三组保留词语的总出现次数：")
for group, total in token_totals.items():
    print(f"{group} 组：{total:,}")

print("\n词频敏感性分析结果已保存。")