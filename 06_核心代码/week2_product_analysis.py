from pathlib import Path
import sqlite3
import pandas as pd
import math

db_path = Path(
    "/Users/ethen/CS/Amazon/mydata/ratings_check.sqlite"
)

connection = sqlite3.connect(
    db_path.as_uri() + "?mode=ro",
    uri=True
)

try:
    print("正在统计每个商品的评分数量……", flush=True)

    sql = """
    SELECT
        product_id,
        COUNT(*) AS rating_count
    FROM ratings
    GROUP BY product_id
    ORDER BY rating_count DESC, product_id ASC
    """

    product_stats = pd.read_sql_query(sql, connection)

finally:
    connection.close()

# 一、核对商品总数和评分总数
total_products = len(product_stats)
total_ratings = int(product_stats["rating_count"].sum())

print("\n商品总数：", total_products)
print("核对评分总数：", total_ratings)

# 二、只有一条评分的商品
single_rating_products = int(
    (product_stats["rating_count"] == 1).sum()
)

single_rating_pct = (
    single_rating_products / total_products * 100
)

print("\n只有一条评分的商品数：", single_rating_products)
print(f"只有一条评分的商品占比：{single_rating_pct:.2f}%")

print(f"\n平均每个商品的评分数：{total_ratings / total_products:.2f}")
print("商品评分数量中位数：", product_stats["rating_count"].median())

# 三、按评分条数排名，检查头部商品的评分占比
print("\n评分集中程度：")

for proportion in [0.01, 0.05, 0.10]:
    # 商品数向上取整，例如 476002 × 1% → 4761 个
    top_n = math.ceil(total_products * proportion)

    top_ratings = int(
        product_stats.head(top_n)["rating_count"].sum()
    )

    share = top_ratings / total_ratings * 100

    print(
        f"评分条数排名前 {proportion:.0%} 的商品"
        f"（{top_n:,} 个），"
        f"获得全部评分的 {share:.2f}%"
    )

# 四、查看评分条数最多的 10 个商品
print("\n评分条数最多的 10 个商品：")
print(product_stats.head(10).to_string(index=False))

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

# 保存每个商品的评分条数
product_stats.to_csv(
    output_dir / "product_rating_counts.csv",
    index=False,
    encoding="utf-8-sig"
)

# 保存关键指标
product_summary = pd.DataFrame({
    "metric": [
        "total_products",
        "total_ratings",
        "single_rating_products",
        "single_rating_product_pct",
        "mean_ratings_per_product",
        "median_ratings_per_product"
    ],
    "value": [
        total_products,
        total_ratings,
        single_rating_products,
        single_rating_pct,
        total_ratings / total_products,
        product_stats["rating_count"].median()
    ]
})

product_summary.to_csv(
    output_dir / "product_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# 保存前 1%、5%、10% 商品的评分集中程度
concentration_rows = []

for proportion in [0.01, 0.05, 0.10]:
    top_n = math.ceil(total_products * proportion)

    top_ratings = int(
        product_stats.head(top_n)["rating_count"].sum()
    )

    concentration_rows.append({
        "top_product_pct": proportion * 100,
        "product_count": top_n,
        "rating_count": top_ratings,
        "rating_share_pct": top_ratings / total_ratings * 100
    })

pd.DataFrame(concentration_rows).to_csv(
    output_dir / "product_rating_concentration.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n商品分析结果已保存至：", output_dir)