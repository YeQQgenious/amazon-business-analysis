from pathlib import Path
import pandas as pd

file_path = Path("/Users/ethen/CS/Amazon/mydata/amazon.csv")

df = pd.read_csv(
    file_path,
    dtype="string",
    encoding="utf-8-sig"
)

# 先检查第一条记录
row_number = 0
row = df.iloc[row_number]

# 需要一起对照的字段
fields = [
    "product_id",
    "product_name",
    "user_id",
    "user_name",
    "review_id",
    "review_title",
    "review_content"
]

output_dir = file_path.parent / "outputs"
output_dir.mkdir(exist_ok=True)

report_path = output_dir / "review_structure_example.txt"

# 保存完整内容，避免终端省略长文本
with report_path.open("w", encoding="utf-8") as report:
    report.write(f"检查记录的位置：{row_number}\n")

    for field in fields:
        value = row[field]

        report.write(f"\n===== {field} =====\n")

        if pd.isna(value):
            report.write("【缺失】\n")
        else:
            report.write(str(value) + "\n")

print("商品编号：", row["product_id"])
print("检查结果已保存至：", report_path)

# 固定随机抽取 5 条，方便以后复现
sample_rows = df.sample(n=5, random_state=42)

sample_path = output_dir / "review_structure_samples.txt"

with sample_path.open("w", encoding="utf-8") as report:
    for row_index, sample_row in sample_rows.iterrows():
        report.write(f"\n{'=' * 60}\n")
        report.write(f"原始行索引：{row_index}\n")

        for field in fields:
            value = sample_row[field]

            report.write(f"\n----- {field} -----\n")

            if pd.isna(value):
                report.write("【缺失】\n")
            else:
                report.write(str(value) + "\n")

print("5 条抽查记录已保存至：", sample_path)


# 缺失值临时替换为空字符串，仅用于检查
review_text = df["review_content"].fillna("")

# 检查空白正文
blank_text = review_text.str.strip().eq("")

# 检查是否包含链接
has_link = review_text.str.contains(
    r"https?://|www\.",
    case=False,
    regex=True
)

# 检查完全相同的非空正文，暂不做清洗或删除
nonblank_text = review_text[~blank_text]

duplicate_text = nonblank_text.duplicated()

print("\n检查记录总数：", len(df))
print("缺失或纯空白的正文记录数：", blank_text.sum())
print("包含链接的正文记录数：", has_link.sum())
print("完全相同正文的额外出现次数：", duplicate_text.sum())

# 保存有链接的记录，方便后续核查
link_examples = df.loc[
    has_link,
    ["product_id", "review_content"]
].copy()

link_examples.insert(
    0,
    "source_row",
    link_examples.index
)

link_examples.to_csv(
    output_dir / "reviews_with_links.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n含链接记录已保存为 reviews_with_links.csv")


# 只检查非空正文，保留原始文本，不进行链接清理
text_records = df.loc[
    ~blank_text,
    ["product_id", "review_content"]
].copy()

# 保留原始行索引，方便追溯
text_records.insert(0, "source_row", text_records.index)

# 1. 同一个商品下，完全相同正文的额外次数
same_product_duplicates = text_records.duplicated(
    subset=["product_id", "review_content"]
)

print("\n同商品、同正文的额外重复次数：")
print(same_product_duplicates.sum())

# 2. 每种正文对应多少个不同商品
products_per_text = text_records.groupby(
    "review_content"
)["product_id"].nunique()

# 找出出现在多个商品编号下的正文
cross_product_texts = products_per_text[
    products_per_text > 1
]

print("\n出现在多个商品下的相同正文种数：")
print(len(cross_product_texts))

# 3. 保存相关记录，供人工核查
cross_product_rows = text_records[
    text_records["review_content"].isin(cross_product_texts.index)
].copy()

print("\n涉及跨商品相同正文的记录数：")
print(len(cross_product_rows))

cross_product_rows.to_csv(
    output_dir / "cross_product_duplicate_texts.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n核查文件已保存：cross_product_duplicate_texts.csv")


# 1. 每种不同正文分配一个编号
unique_texts = (
    text_records[["review_content"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

unique_texts.insert(
    0,
    "text_id",
    [f"T{i:04d}" for i in range(1, len(unique_texts) + 1)]
)

# 2. 将文本编号对应回每条原始记录
mapped_records = text_records.merge(
    unique_texts,
    on="review_content",
    how="left",
    validate="many_to_one"
)

# 3. 同商品、同正文合并来源信息，不丢原始行索引
product_text_map = (
    mapped_records
    .groupby(["product_id", "text_id"], as_index=False)
    .agg(
        source_rows=(
            "source_row",
            lambda values: ",".join(str(v) for v in values)
        ),
        source_record_count=("source_row", "size")
    )
)

# 4. 核对数量
print("\n唯一正文数量：", len(unique_texts))
print("不同商品—正文对应关系数：", len(product_text_map))
print(
    "映射表追溯的原始记录数：",
    product_text_map["source_record_count"].sum()
)

# 5. 保存为新文件
unique_texts.to_csv(
    output_dir / "review_unique_texts.csv",
    index=False,
    encoding="utf-8-sig"
)

product_text_map.to_csv(
    output_dir / "product_review_text_map.csv",
    index=False,
    encoding="utf-8-sig"
)

from pathlib import Path
import pandas as pd

output_dir = Path("/Users/ethen/CS/Amazon/mydata/outputs")

texts = pd.read_csv(
    output_dir / "review_unique_texts.csv",
    dtype="string",
    encoding="utf-8-sig"
)

# 原文保留不变
original = texts["review_content"].fillna("")

# URL 候选规则：
# 遇到空白、逗号、引号或尖括号就停止，避免吞掉常见分隔符后的评论
url_pattern = r"""(?:https?://|www\.)[^\s,<>"']+"""

texts["had_link"] = original.str.contains(
    url_pattern,
    case=False,
    regex=True
)

# 新增清洗列：先去链接，再合并多余空白
texts["review_content_clean"] = (
    original
    .str.replace(url_pattern, " ", regex=True, case=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 标记清洗后空白和仍含链接前缀的文本
texts["is_blank_after_clean"] = (
    texts["review_content_clean"].eq("")
)

remaining_link = texts["review_content_clean"].str.contains(
    r"https?://|www\.",
    case=False,
    regex=True
)

# 清洗可能让原本不同的文本变得相同，先统计，不再次删除
nonblank = texts.loc[
    ~texts["is_blank_after_clean"],
    "review_content_clean"
]

print("处理的唯一正文数量：", len(texts))
print("含链接的唯一正文数量：", texts["had_link"].sum())
print("清洗后空白正文数量：", texts["is_blank_after_clean"].sum())
print("仍含链接前缀的正文数量：", remaining_link.sum())
print("清洗后相同正文的额外次数：", nonblank.duplicated().sum())

# 保存完整结果，text_id 不变，仍可关联商品映射表
texts.to_csv(
    output_dir / "review_texts_clean.csv",
    index=False,
    encoding="utf-8-sig"
)

# 保存前 5 段含链接正文的清洗前后对照
preview_path = output_dir / "review_cleaning_preview.txt"

with preview_path.open("w", encoding="utf-8") as report:
    for _, row in texts.loc[texts["had_link"]].head(5).iterrows():
        report.write(f"\n===== {row['text_id']} =====\n")
        report.write("\n【原文】\n" + row["review_content"] + "\n")
        report.write("\n【清洗后】\n" + row["review_content_clean"] + "\n")

#print("\n已保存清洗表和前后对照文件。")


# 找出清洗后正文重复的所有记录，包括每组第一次出现
duplicate_mask = (
    ~texts["is_blank_after_clean"]
    & texts["review_content_clean"].duplicated(keep=False)
)

duplicate_rows = texts.loc[
    duplicate_mask,
    ["text_id", "review_content", "review_content_clean"]
].copy()

# 将清洗后相同的正文放在一起，方便对比
duplicate_rows = duplicate_rows.sort_values(
    ["review_content_clean", "text_id"]
)

print("\n清洗后重复正文组数：", duplicate_rows["review_content_clean"].nunique())

print("\n涉及的原始文本编号数量：", len(duplicate_rows))


duplicate_rows.to_csv(
    output_dir / "review_duplicates_after_cleaning.csv",
    index=False,
    encoding="utf-8-sig"
)

#print("\n已保存清洗后重复正文核查表。")

# 1. 每种清洗后正文只保留一份
final_texts = (
    texts[["review_content_clean"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# 分配清洗后的文本编号
final_texts.insert(
    0,
    "clean_text_id",
    [f"C{i:04d}" for i in range(1, len(final_texts) + 1)]
)

# 2. 保留每个旧 text_id 对应哪个 clean_text_id
text_id_map = texts[["text_id", "review_content_clean"]].merge(
    final_texts,
    on="review_content_clean",
    how="left",
    validate="many_to_one"
)[["text_id", "clean_text_id"]]

# 3. 读取之前保存的商品—正文来源表
product_map = pd.read_csv(
    output_dir / "product_review_text_map.csv",
    dtype={
        "product_id": "string",
        "text_id": "string",
        "source_rows": "string",
        "source_record_count": "int64"
    },
    encoding="utf-8-sig"
)

# 添加新编号，保留旧编号及全部来源信息
product_clean_map = product_map.merge(
    text_id_map,
    on="text_id",
    how="left",
    validate="many_to_one"
)

# 4. 检查是否有映射遗漏
assert text_id_map["clean_text_id"].notna().all()
assert product_clean_map["clean_text_id"].notna().all()
assert len(product_clean_map) == len(product_map)

print("\n最终不同清洗文本数量：", len(final_texts))
print("原始文本编号映射数量：", len(text_id_map))
print("保留的商品—原始文本关系数量：", len(product_clean_map))
print(
    "可追溯的原始数据行数：",
    product_clean_map["source_record_count"].sum()
)

# 5. 保存新文件，不覆盖原文表
final_texts.to_csv(
    output_dir / "review_clean_unique_texts.csv",
    index=False,
    encoding="utf-8-sig"
)

text_id_map.to_csv(
    output_dir / "review_text_id_map.csv",
    index=False,
    encoding="utf-8-sig"
)

product_clean_map.to_csv(
    output_dir / "product_clean_text_map.csv",
    index=False,
    encoding="utf-8-sig"
)