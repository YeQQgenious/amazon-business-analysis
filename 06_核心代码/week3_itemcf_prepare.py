from pathlib import Path
from datetime import datetime, timezone
from array import array
import csv
import json
import sqlite3

import numpy as np
import scipy
from scipy.sparse import csr_matrix, save_npz


# ---------- 1. 固定路径与训练边界 ----------
base_dir = Path("/Users/ethen/CS/Amazon/mydata")
db_path = base_dir / "ratings_check.sqlite"
model_dir = base_dir / "outputs" / "itemcf"
model_dir.mkdir(parents=True, exist_ok=True)

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
    # ---------- 2. 确定全部训练期商品 ----------
    # 包括没有高星评分的训练期商品，保持候选范围一致
    print("正在读取训练期商品编号……")

    product_ids = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT product_id
            FROM ratings
            WHERE CAST(timestamp AS INTEGER) < ?
            ORDER BY product_id
            """,
            (cutoff,)
        )
    ]

    if len(product_ids) != 406064:
        raise ValueError("训练期商品数量与此前不一致。")

    product_to_index = {
        product_id: index
        for index, product_id in enumerate(product_ids)
    }

    # ---------- 3. 核对训练期正反馈数量 ----------
    expected_positive_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM ratings
        WHERE CAST(timestamp AS INTEGER) < ?
          AND CAST(rating AS REAL) BETWEEN 4 AND 5
        """,
        (cutoff,)
    ).fetchone()[0]

    print(f"训练期正反馈数量：{expected_positive_count:,}")

    # ---------- 4. 按用户逐行构建稀疏矩阵 ----------
    # indices：每条正反馈对应的商品列号
    # indptr：每个用户在 indices 中的起止位置
    # 使用紧凑整数数组，避免保存数百万个 Python 整数对象
    indices = array("q")
    indptr = array("q", [0])

    current_user = None
    user_count = 0
    processed = 0

    print("正在构建用户—商品正反馈矩阵……")

    cursor = conn.execute(
        """
        SELECT user_id, product_id
        FROM ratings
        WHERE CAST(timestamp AS INTEGER) < ?
          AND CAST(rating AS REAL) BETWEEN 4 AND 5
        ORDER BY user_id, product_id
        """,
        (cutoff,)
    )

    with (model_dir / "user_index.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["user_index", "user_id"])

        for user_id, product_id in cursor:
            if user_id != current_user:
                if current_user is not None:
                    indptr.append(len(indices))

                writer.writerow([user_count, user_id])
                user_count += 1
                current_user = user_id

            indices.append(product_to_index[product_id])
            processed += 1

            if processed % 1000000 == 0:
                print(f"已处理 {processed:,} 条训练正反馈……")

    if current_user is not None:
        indptr.append(len(indices))

    if processed != expected_positive_count or processed == 0:
        raise ValueError("训练正反馈数量核对失败。")

finally:
    conn.close()


# ---------- 5. 构建 CSR 稀疏矩阵 ----------
index_values = np.asarray(indices, dtype=np.int64)
pointer_values = np.asarray(indptr, dtype=np.int64)

# 使用浮点数，便于后续相似度运算
values = np.ones(processed, dtype=np.float32)

matrix = csr_matrix(
    (values, index_values, pointer_values),
    shape=(user_count, len(product_ids)),
    dtype=np.float32
)

# 检查是否存在重复的用户—商品正反馈
matrix.sum_duplicates()
matrix.sort_indices()

if matrix.nnz != processed or not np.all(matrix.data == 1):
    raise ValueError("发现重复用户—商品正反馈，请检查数据。")


# ---------- 6. 保存矩阵与商品映射 ----------
save_npz(
    model_dir / "train_positive_user_item.npz",
    matrix
)

positive_counts = np.asarray(
    matrix.getnnz(axis=0)
).ravel()

with (model_dir / "product_index.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "product_index",
        "product_id",
        "train_positive_count"
    ])

    for index, product_id in enumerate(product_ids):
        writer.writerow([
            index,
            product_id,
            int(positive_counts[index])
        ])


# ---------- 7. 输出模型输入规模 ----------
positive_products = int((positive_counts > 0).sum())
zero_positive_products = int((positive_counts == 0).sum())

matrix_memory_mb = (
    matrix.data.nbytes
    + matrix.indices.nbytes
    + matrix.indptr.nbytes
) / (1024 ** 2)

metadata = {
    "cutoff_date_utc": "2014-01-01",
    "positive_rating_min": 4,
    "positive_rating_max": 5,
    "positive_user_count": user_count,
    "candidate_product_count": len(product_ids),
    "positive_product_count": positive_products,
    "zero_positive_product_count": zero_positive_products,
    "train_positive_count": processed,
    "matrix_memory_mb": matrix_memory_mb,
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__
}

with (model_dir / "prepare_summary.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)

print("\n协同过滤输入准备完成：")
print(f"训练期有正反馈的用户数：{user_count:,}")
print(f"训练期候选商品数：{len(product_ids):,}")
print(f"训练期有正反馈的商品数：{positive_products:,}")
print(f"训练期没有正反馈的商品数：{zero_positive_products:,}")
print(f"训练期正反馈数量：{matrix.nnz:,}")
print(f"矩阵形状：{matrix.shape}")
print(f"稀疏矩阵数组占用：{matrix_memory_mb:.2f} MB")
print("注意：上述内存不包括编号映射和程序其他对象。")
print("\n文件保存至：", model_dir)
print("未读取测试期评分用于建模，原始数据库未修改。")