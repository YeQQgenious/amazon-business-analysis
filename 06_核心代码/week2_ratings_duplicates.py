import pandas as pd
import sqlite3
from pathlib import Path

file_path = "/Users/ethen/CS/Amazon/mydata/ratings_Electronics.csv"

# 新建本地数据库，不修改原始 CSV
db_path = Path("/Users/ethen/CS/Amazon/mydata/ratings_check.sqlite")

# 防止重复运行时，把同一份数据再次追加进去
if db_path.exists():
    raise FileExistsError(
        "数据库文件已经存在，先不要重复导入。请把提示发给我。"
    )

columns = ["user_id", "product_id", "rating", "timestamp"]

reader = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    dtype="string",
    chunksize=100000
)

connection = sqlite3.connect(db_path)
total_rows = 0

try:
    for batch_number, chunk in enumerate(reader, start=1):
        # 与第一周检查保持一致：去除首尾空格
        for column in columns:
            chunk[column] = chunk[column].str.strip()

        # 保留全部记录，不去重、不覆盖已有批次
        chunk.to_sql(
            "ratings",
            connection,
            if_exists="append",
            index=False
        )

        total_rows += len(chunk)

        if batch_number % 10 == 0:
            print(f"已导入 {total_rows:,} 条记录……", flush=True)

    # 检查数据库中实际保存的数量
    database_rows = connection.execute(
        "SELECT COUNT(*) FROM ratings"
    ).fetchone()[0]

    print("\n导入完成")
    print("CSV 本次读取行数：", total_rows)
    print("数据库实际行数：", database_rows)
    print("两者是否一致：", total_rows == database_rows)

finally:
    connection.close()