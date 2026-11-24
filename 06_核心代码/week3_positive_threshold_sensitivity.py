"""评分阈值敏感性实验：只读数据库，独立保存，不修改旧结果。

先运行 --threshold 4，核对原实验；再运行 --threshold 3 5。
"""
from argparse import ArgumentParser
from array import array
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
import csv
import json
import sqlite3

import numpy as np
from scipy.sparse import csr_matrix

CUTOFF = int(datetime(2014, 1, 1, tzinfo=timezone.utc).timestamp())
NEIGHBORS = 50
K = 10
DISCOUNTS = 1 / np.log2(np.arange(1, K + 1) + 1)
BASE = Path('/Users/ethen/CS/Amazon/mydata')


def log(message):
    print(message, flush=True)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'没有可保存的结果：{path}')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_neighbors(conn, threshold, product_map):
    log(f'[{threshold}星] 构建二值正反馈矩阵……')
    indices, pointers = array('q'), array('q', [0])
    previous = None
    user_count = 0
    for user, product in conn.execute('''
        SELECT user_id, product_id FROM ratings
        WHERE CAST(timestamp AS INTEGER) < ?
          AND CAST(rating AS REAL) BETWEEN ? AND 5
        ORDER BY user_id, product_id
    ''', (CUTOFF, threshold)):
        if user != previous:
            if previous is not None:
                pointers.append(len(indices))
            previous = user
            user_count += 1
        indices.append(product_map[product])
    if previous is not None:
        pointers.append(len(indices))
    if not indices:
        raise ValueError('训练期没有正反馈，无法继续。')
    count = len(indices)
    matrix = csr_matrix((np.ones(count, dtype=np.float32),
                         np.asarray(indices, dtype=np.int64),
                         np.asarray(pointers, dtype=np.int64)),
                        shape=(user_count, len(product_map)))
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.nnz != count or not np.all(matrix.data == 1):
        raise ValueError('存在重复用户—商品正反馈，停止运行。')
    del indices, pointers
    # 分母必须在排除单正反馈用户之前计算。
    positive_counts = np.asarray(matrix.getnnz(axis=0)).ravel()
    norms = np.sqrt(positive_counts.astype(np.float64))
    co = matrix[np.diff(matrix.indptr) >= 2].astype(np.float64).tocsr()
    del matrix
    transposed = co.T.tocsr()
    active = np.flatnonzero(positive_counts > 0)
    ids_out = np.full((len(product_map), NEIGHBORS), -1, dtype=np.int32)
    scores_out = np.zeros(ids_out.shape, dtype=np.float32)
    counts_out = np.zeros(len(product_map), dtype=np.int32)
    log(f'[{threshold}星] 训练正反馈 {count:,} 条；计算商品邻居……')
    for start in range(0, len(active), 32):
        block = active[start:start + 32]
        common = (transposed[block] @ co).tocsr()
        common.sum_duplicates()
        common.eliminate_zeros()
        for row, source in enumerate(block):
            begin, end = common.indptr[row:row + 2]
            ids = common.indices[begin:end]
            shared = common.data[begin:end]
            keep = (ids != source) & (shared > 0)
            ids, shared = ids[keep], shared[keep]
            if not len(ids):
                continue
            scores = shared / (norms[source] * norms[ids])
            if not np.isfinite(scores).all() or (scores > 1 + 1e-10).any():
                raise ValueError('相似度异常。')
            scores = np.minimum(scores, 1.0)
            # 与原脚本一致：相似度降序，编号升序处理并列。
            if len(ids) > NEIGHBORS:
                boundary = np.partition(scores, len(scores) - NEIGHBORS)[-NEIGHBORS]
                higher = np.flatnonzero(scores > boundary)
                tied = np.flatnonzero(scores == boundary)
                tied = tied[np.argsort(ids[tied], kind='stable')[:NEIGHBORS-len(higher)]]
                chosen = np.concatenate([higher, tied])
                ids, scores = ids[chosen], scores[chosen]
            order = np.lexsort((ids, -scores))
            n = len(order)
            ids_out[source, :n] = ids[order]
            scores_out[source, :n] = scores[order]
            counts_out[source] = n
        if start % 32000 == 0 or start + 32 >= len(active):
            log(f'  邻居进度：{min(start+32, len(active)):,}/{len(active):,}')
    return ids_out, scores_out, counts_out, positive_counts, count, user_count


def verify_reference(rows, reference):
    """从原始完整精度 CSV 校验阈值4，而非截图中的舍入值。"""
    checks = []
    for method, filename in [('popularity', 'week3_popularity_summary.csv'),
                             ('itemcf', 'week3_itemcf_summary.csv')]:
        path = reference / filename
        if not path.exists():
            raise FileNotFoundError(f'复现核对缺少原结果：{path}')
        with path.open(encoding='utf-8-sig', newline='') as f:
            original = {r['group']: r for r in csv.DictReader(f)}
        for row in rows:
            if row['group'] == 'common_5star_users':
                continue
            old = original[row['group']]
            checks.append({'method': method, 'group': row['group'], 'metric': 'user_count',
                           'old': old['user_count'], 'new': row['user_count'],
                           'passed': int(old['user_count']) == row['user_count']})
            for metric in ['recall_at_10', 'ndcg_at_10']:
                new = row[f'{method}_{metric}']
                value = float(old[metric])
                checks.append({'method': method, 'group': row['group'], 'metric': metric,
                               'old': value, 'new': new,
                               'passed': bool(np.isclose(new, value, rtol=1e-9, atol=1e-12))})
    return checks


def run_threshold(conn, threshold, root, reference):
    destination = root / f'threshold_{threshold}' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True, exist_ok=False)
    products = [r[0] for r in conn.execute('''SELECT DISTINCT product_id FROM ratings
        WHERE CAST(timestamp AS INTEGER) < ? ORDER BY product_id''', (CUTOFF,))]
    product_map = {p: i for i, p in enumerate(products)}
    if len(products) < K:
        raise ValueError('候选商品不足10件。')
    ids, scores, neighbor_counts, positive_counts, train_count, train_users = make_neighbors(
        conn, threshold, product_map)
    popularity = np.lexsort((np.arange(len(products)), -positive_counts)).tolist()
    write_csv(destination / 'popularity_ranking.csv', (
        {'product_id': products[i], 'train_positive_count': int(positive_counts[i])}
        for i in popularity))
    # 五星目标用户是三组评估用户的交集；仍使用各组自己的目标定义。
    common_users = {r[0] for r in conn.execute('''SELECT DISTINCT user_id FROM ratings
        WHERE CAST(timestamp AS INTEGER) >= ? AND CAST(rating AS REAL) = 5''', (CUTOFF,))}
    eval_users = {r[0] for r in conn.execute('''SELECT DISTINCT user_id FROM ratings
        WHERE CAST(timestamp AS INTEGER) >= ? AND CAST(rating AS REAL) BETWEEN ? AND 5''',
        (CUTOFF, threshold))}
    if not eval_users:
        raise ValueError('该阈值下没有测试正反馈用户。')
    histories, liked = {}, {}
    log(f'[{threshold}星] 读取 {len(eval_users):,} 名评估用户的历史……')
    for user, product, rating in conn.execute('''SELECT user_id, product_id, CAST(rating AS REAL)
        FROM ratings WHERE CAST(timestamp AS INTEGER) < ?''', (CUTOFF,)):
        if user in eval_users:
            i = product_map[product]
            histories.setdefault(user, set()).add(i)
            if threshold <= rating <= 5:
                liked.setdefault(user, set()).add(i)
    expected_users = len(eval_users)
    del eval_users
    # 用户数、目标数、两种方法的Recall/NDCG之和、个性化用户数。
    aggregates = defaultdict(lambda: [0, 0, 0., 0., 0., 0., 0])
    coverage = {}
    target_total = 0
    processed = 0
    fields = ['user_id', 'train_history_count', 'target_count', 'common_5star_user',
              'personalized_count', 'popularity_hits', 'itemcf_hits',
              'popularity_recall_at_10', 'itemcf_recall_at_10',
              'popularity_ndcg_at_10', 'itemcf_ndcg_at_10']
    with (destination / 'user_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        cursor = conn.execute('''SELECT user_id, product_id FROM ratings
            WHERE CAST(timestamp AS INTEGER) >= ? AND CAST(rating AS REAL) BETWEEN ? AND 5
            ORDER BY user_id, product_id''', (CUTOFF, threshold))
        for user, records in groupby(cursor, key=lambda r: r[0]):
            raw = [r[1] for r in records]
            targets = set(raw)
            seen = histories.get(user, set())
            if len(raw) != len(targets) or any(product_map.get(p, -1) in seen for p in targets):
                raise ValueError('发现重复测试目标或训练/测试用户商品重叠。')
            base = []
            for i in popularity:
                if i not in seen:
                    base.append(i)
                    if len(base) == K:
                        break
            totals = defaultdict(float)
            for source in sorted(liked.get(user, set())):
                for position in range(int(neighbor_counts[source])):
                    candidate = int(ids[source, position])
                    if candidate not in seen:
                        totals[candidate] += float(scores[source, position])
            recs = sorted(totals, key=lambda i: (-totals[i], i))[:K]
            personal = len(recs)
            chosen = set(recs)
            # 基线的前10个未评分商品足以为长度不超过10的列表补齐。
            for i in base:
                if len(recs) == K:
                    break
                if i not in chosen:
                    recs.append(i)
                    chosen.add(i)
            if len(base) != K or len(recs) != K or chosen & seen:
                raise ValueError('推荐列表不完整或含历史商品。')
            hit_counts, recalls, ndcgs = [], [], []
            for rec in (base, recs):
                relevance = [int(products[i] in targets) for i in rec]
                hits = sum(relevance)
                hit_counts.append(hits)
                recalls.append(hits / len(targets))
                ndcgs.append(sum(v * float(DISCOUNTS[p]) for p, v in enumerate(relevance)) /
                             sum(float(v) for v in DISCOUNTS[:min(K, len(targets))]))
            groups = ['all_positive_users', 'known_users' if seen else 'new_users']
            if user in common_users:
                groups.append('common_5star_users')
            for group in groups:
                a = aggregates[group]
                a[0] += 1
                a[1] += len(targets)
                a[2] += recalls[0]
                a[3] += ndcgs[0]
                a[4] += recalls[1]
                a[5] += ndcgs[1]
                a[6] += int(personal > 0)
                if group not in coverage:
                    coverage[group] = (np.zeros(len(products), dtype=bool),
                                       np.zeros(len(products), dtype=bool))
                coverage[group][0][base] = True
                coverage[group][1][recs] = True
            writer.writerow([user, len(seen), len(targets), int(user in common_users), personal,
                             *hit_counts, *recalls, *ndcgs])
            processed += 1
            target_total += len(targets)
            if processed % 100000 == 0:
                log(f'  已评估 {processed:,}/{expected_users:,} 名用户')
    if processed != expected_users:
        raise ValueError('评估人数核对失败。')
    summary = []
    for group, a in aggregates.items():
        n = a[0]
        pop_recall, pop_ndcg, cf_recall, cf_ndcg = [v/n for v in a[2:6]]
        summary.append({
            'threshold': threshold, 'group': group, 'user_count': n,
            'train_positive_count': train_count, 'train_positive_user_count': train_users,
            'test_positive_target_count': a[1], 'candidate_product_count': len(products),
            'popularity_recall_at_10': pop_recall, 'itemcf_recall_at_10': cf_recall,
            'recall_relative_change_pct': (cf_recall/pop_recall-1)*100 if pop_recall else '',
            'popularity_ndcg_at_10': pop_ndcg, 'itemcf_ndcg_at_10': cf_ndcg,
            'ndcg_relative_change_pct': (cf_ndcg/pop_ndcg-1)*100 if pop_ndcg else '',
            'popularity_coverage_pct': float(coverage[group][0].mean()*100),
            'itemcf_coverage_pct': float(coverage[group][1].mean()*100),
            'personalized_user_pct': a[6]/n*100,
        })
    write_csv(destination / 'summary.csv', summary)
    if threshold == 4 and reference is not None:
        checks = verify_reference(summary, reference)
        write_csv(destination / 'reference_check.csv', checks)
        if not all(c['passed'] for c in checks):
            raise ValueError(f'阈值4未复现原结果，先不要运行其他阈值。查看 {destination}/reference_check.csv')
        log('≥4星组与原实验的用户数、Recall、NDCG核对通过。')
    settings = {'threshold': threshold, 'cutoff_utc': '2014-01-01', 'top_neighbors': NEIGHBORS,
                'recommendation_k': K, 'database': str(conn.execute('PRAGMA database_list').fetchone()[2]),
                'train_positive_count': train_count, 'test_positive_target_count': target_total,
                'evaluation_user_count': processed, 'numpy_version': np.__version__,
                'common_cohort_note': '测试期至少有一次5星评分的用户；各组测试目标仍随阈值变化。',
                'reference_verified': threshold == 4 and reference is not None,
                'completed': True}
    (destination / 'completed.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'[{threshold}星] 完成，结果保存至：{destination}')
    overall = next(r for r in summary if r['group'] == 'all_positive_users')
    log(f"  Recall@10：{overall['popularity_recall_at_10']:.6f} → {overall['itemcf_recall_at_10']:.6f}")
    return destination


def collect_summaries(root):
    rows = []
    for threshold in (3, 4, 5):
        completed = sorted((root / f'threshold_{threshold}').glob('*/completed.json'))
        if not completed:
            continue
        directory = completed[-1].parent
        with (directory / 'summary.csv').open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                row['source_run'] = str(directory)
                rows.append(row)
    if rows:
        path = root / 'comparison_summary.csv'
        write_csv(path, rows)
        log(f'已更新汇总表（仅包含已完成的阈值组）：{path}')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--threshold', nargs='+', type=int, choices=[3, 4, 5], default=[4])
    parser.add_argument('--database', type=Path, default=BASE / 'ratings_check.sqlite')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'outputs' / 'positive_threshold_sensitivity')
    parser.add_argument('--reference-dir', type=Path, default=BASE / 'outputs')
    args = parser.parse_args()
    if not args.database.is_file():
        raise FileNotFoundError(args.database)
    # 只读连接，原数据库不做任何修改。
    conn = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        for threshold in dict.fromkeys(args.threshold):
            run_threshold(conn, threshold, args.output_dir, args.reference_dir)
            collect_summaries(args.output_dir)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
