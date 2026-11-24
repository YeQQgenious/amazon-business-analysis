from pathlib import Path
import json
import time

import numpy as np
from scipy.sparse import load_npz


# ---------- 1. 固定配置 ----------
model_dir = Path(
    "/Users/ethen/CS/Amazon/mydata/outputs/itemcf"
)

matrix_path = model_dir / "train_positive_user_item.npz"

TOP_K = 50

# 仅控制计算内存，不改变相似度定义和最终邻居规则
BLOCK_SIZE = 32

if not matrix_path.exists():
    raise FileNotFoundError(f"找不到训练矩阵：{matrix_path}")

start_time = time.perf_counter()

print("正在读取训练期正反馈矩阵……")

matrix = load_npz(matrix_path).tocsr()
matrix.sort_indices()

if matrix.shape != (2624827, 406064):
    raise ValueError("矩阵形状与上一阶段结果不一致。")

if matrix.nnz != 4545529:
    raise ValueError("正反馈记录数量与上一阶段结果不一致。")

if not np.all(matrix.data == 1):
    raise ValueError("输入应为二值正反馈矩阵。")

n_users, n_products = matrix.shape


# ---------- 2. 保存完整的商品正反馈数量 ----------
# 相似度分母必须使用全部训练期正反馈，
# 不能在后面的计算优化之后重新计算这个分母
positive_counts = np.asarray(
    matrix.getnnz(axis=0)
).ravel().astype(np.int64)

norms = np.sqrt(positive_counts.astype(np.float64))

active_products = np.flatnonzero(positive_counts > 0)

print(f"训练期候选商品数：{n_products:,}")
print(f"有正反馈的商品数：{len(active_products):,}")


# ---------- 3. 优化共同用户计数 ----------
# 只有一个正反馈商品的用户，
# 不可能贡献两个不同商品之间的共同正反馈。
# 因此只在共同用户计数时跳过他们。
# 注意：上述分母仍然保留了这些用户的贡献。
user_positive_counts = np.diff(matrix.indptr)

cooccurrence_matrix = matrix[
    user_positive_counts >= 2
].tocsr()

# 使用 float64 计算共同用户数量及相似度
cooccurrence_matrix = cooccurrence_matrix.astype(np.float64)

del matrix

item_user_matrix = cooccurrence_matrix.T.tocsr()

print(
    "可贡献商品间共同反馈的用户数："
    f"{cooccurrence_matrix.shape[0]:,}"
)


# ---------- 4. 预分配邻居结果 ----------
# 每一行对应 product_index.csv 中的一个商品
# -1 表示该位置没有邻居
neighbor_indices = np.full(
    (n_products, TOP_K),
    -1,
    dtype=np.int32
)

neighbor_scores = np.zeros(
    (n_products, TOP_K),
    dtype=np.float32
)

neighbor_counts = np.zeros(n_products, dtype=np.int32)


def select_top_neighbors(ids, scores, k):
    """保留最高相似度；并列时选编号索引较小的商品。"""

    if len(ids) > k:
        boundary = np.partition(
            scores, len(scores) - k
        )[len(scores) - k]

        higher = np.flatnonzero(scores > boundary)
        tied = np.flatnonzero(scores == boundary)

        remaining = k - len(higher)

        # product_index 是按商品编号排序建立的
        tied = tied[
            np.argsort(ids[tied], kind="stable")[:remaining]
        ]

        selected = np.concatenate([higher, tied])
        ids = ids[selected]
        scores = scores[selected]

    order = np.lexsort((ids, -scores))

    return ids[order], scores[order]


# ---------- 5. 分块计算全部商品邻居 ----------
print("\n正在计算商品相似邻居……")

next_progress = 10000
active_count = len(active_products)

for start in range(0, active_count, BLOCK_SIZE):
    block_ids = active_products[start:start + BLOCK_SIZE]

    # 仅生成当前块与全部商品的共同用户计数
    # 不生成完整的 商品数 × 商品数 矩阵
    common = (
        item_user_matrix[block_ids]
        @ cooccurrence_matrix
    ).tocsr()

    common.sum_duplicates()
    common.eliminate_zeros()

    for local_row, product_index in enumerate(block_ids):
        begin = common.indptr[local_row]
        end = common.indptr[local_row + 1]

        ids = common.indices[begin:end]
        shared_counts = common.data[begin:end]

        # 去掉商品自身，以及无有效共同用户的位置
        valid = (ids != product_index) & (shared_counts > 0)

        ids = ids[valid]
        shared_counts = shared_counts[valid]

        if len(ids) == 0:
            continue

        scores = shared_counts / (
            norms[product_index] * norms[ids]
        )

        if (
            not np.isfinite(scores).all()
            or (scores <= 0).any()
            or (scores > 1 + 1e-10).any()
        ):
            raise ValueError("相似度出现异常数值。")

        # 处理浮点运算导致的极小边界误差
        scores = np.minimum(scores, 1.0)

        ids, scores = select_top_neighbors(
            ids, scores, TOP_K
        )

        count = len(ids)

        neighbor_indices[product_index, :count] = ids
        neighbor_scores[product_index, :count] = scores
        neighbor_counts[product_index] = count

    completed = min(start + BLOCK_SIZE, active_count)

    if completed >= next_progress or completed == active_count:
        elapsed_minutes = (
            time.perf_counter() - start_time
        ) / 60

        print(
            f"已处理 {completed:,}/{active_count:,} 个有正反馈商品，"
            f"累计耗时 {elapsed_minutes:.1f} 分钟"
        )

        while next_progress <= completed:
            next_progress += 10000


# ---------- 6. 核对结果 ----------
products_with_neighbors = int(
    (neighbor_counts > 0).sum()
)

products_without_neighbors = int(
    (neighbor_counts == 0).sum()
)

positive_products_without_neighbors = int(
    (
        (positive_counts > 0)
        & (neighbor_counts == 0)
    ).sum()
)

products_with_50_neighbors = int(
    (neighbor_counts == TOP_K).sum()
)

total_neighbor_links = int(neighbor_counts.sum())

if products_with_neighbors + products_without_neighbors != n_products:
    raise ValueError("商品数量核对失败。")

if (neighbor_counts > TOP_K).any():
    raise ValueError("邻居数量超过设定上限。")


# ---------- 7. 保存模型 ----------
np.savez_compressed(
    model_dir / "itemcf_neighbors_k50.npz",
    neighbor_indices=neighbor_indices,
    neighbor_scores=neighbor_scores,
    neighbor_counts=neighbor_counts
)

summary = {
    "method": "binary_positive_cosine",
    "cutoff_date_utc": "2014-01-01",
    "positive_rating_min": 4,
    "positive_rating_max": 5,
    "top_k_neighbors": TOP_K,
    "candidate_product_count": n_products,
    "positive_product_count": int(len(active_products)),
    "products_with_neighbors": products_with_neighbors,
    "products_without_neighbors": products_without_neighbors,
    "positive_products_without_neighbors":
        positive_products_without_neighbors,
    "products_with_50_neighbors": products_with_50_neighbors,
    "total_directed_neighbor_links": total_neighbor_links,
    "mean_neighbors_all_products":
        float(neighbor_counts.mean()),
    "mean_neighbors_connected_products": (
        float(neighbor_counts[neighbor_counts > 0].mean())
        if products_with_neighbors else 0.0
    ),
    "elapsed_minutes":
        (time.perf_counter() - start_time) / 60
}

with (model_dir / "itemcf_neighbors_summary.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(
        summary, file,
        ensure_ascii=False, indent=2
    )

print("\n商品相似邻居计算完成：")
print(f"候选商品数：{n_products:,}")
print(f"至少有一个邻居的商品数：{products_with_neighbors:,}")
print(f"没有邻居的商品数：{products_without_neighbors:,}")
print(
    "有正反馈但没有邻居的商品数："
    f"{positive_products_without_neighbors:,}"
)
print(f"拥有 50 个邻居的商品数：{products_with_50_neighbors:,}")
print(f"保留的有向邻居关系数：{total_neighbor_links:,}")
print(f"全部商品平均邻居数：{neighbor_counts.mean():.2f}")
print(
    "有邻居商品的平均邻居数："
    f"{summary['mean_neighbors_connected_products']:.2f}"
)
print(f"累计耗时：{summary['elapsed_minutes']:.1f} 分钟")
print("\n模型及摘要已保存至：", model_dir)
print("未使用测试期评分计算相似度，原始数据库未修改。")