"""
Lab 3 — Empirical Analysis of DFS and BFS
==========================================
Implements Depth-First Search (iterative + recursive) and Breadth-First Search,
benchmarks them across multiple graph types and sizes, exports results to CSV,
and generates comparison charts.
"""

import time
import random
import csv
import sys
import math
import os
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
#  GRAPH REPRESENTATION
# ============================================================

def make_graph(n, edges):
    """Build adjacency list (dict[int -> list[int]]) from an edge list."""
    g = {i: [] for i in range(n)}
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    return g


# ============================================================
#  GRAPH GENERATORS
# ============================================================

def random_graph(n, p):
    """Erdős–Rényi G(n, p): each pair of nodes connected independently with prob p."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                edges.append((u, v))
    return make_graph(n, edges)


def sparse_random_graph(n):
    """Sparse random graph: average degree ~4 (p ≈ 4/n)."""
    p = min(4.0 / max(n - 1, 1), 1.0)
    return random_graph(n, p)


def dense_random_graph(n):
    """Dense random graph: p = 0.15 (roughly 15% of all possible edges)."""
    return random_graph(n, p=0.15)


def tree_graph(n):
    """Random tree: each node i > 0 is connected to a random parent in [0, i-1]."""
    if n == 1:
        return {0: []}
    edges = [(i, random.randint(0, i - 1)) for i in range(1, n)]
    return make_graph(n, edges)


def path_graph(n):
    """Simple path: 0-1-2-...(n-1)."""
    return make_graph(n, [(i, i + 1) for i in range(n - 1)])


def cycle_graph(n):
    """Cycle: 0-1-2-...-(n-1)-0."""
    return make_graph(n, [(i, (i + 1) % n) for i in range(n)])


def grid_graph(n):
    """
    Approximately n-node 2D grid.
    side = floor(sqrt(n)); actual node count = side².
    """
    side = max(1, int(math.isqrt(n)))
    actual_n = side * side
    edges = []
    for r in range(side):
        for c in range(side):
            node = r * side + c
            if c + 1 < side:
                edges.append((node, node + 1))
            if r + 1 < side:
                edges.append((node, node + side))
    return make_graph(actual_n, edges)


# ============================================================
#  BFS  (iterative, queue-based)
# ============================================================

def bfs(graph, start=0):
    """
    Breadth-First Search from `start`.
    Handles disconnected graphs by restarting from every unvisited node.
    Returns (traversal_order: list[int], nodes_visited: int).
    """
    visited = set()
    order = []

    def _bfs_component(src):
        queue = collections.deque([src])
        visited.add(src)
        while queue:
            node = queue.popleft()
            order.append(node)
            for nb in graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

    _bfs_component(start)
    for node in graph:
        if node not in visited:
            _bfs_component(node)

    return order, len(order)


# ============================================================
#  DFS  (iterative, explicit stack)
# ============================================================

def dfs_iterative(graph, start=0):
    """
    Iterative Depth-First Search using an explicit stack.
    Handles disconnected graphs.
    Returns (traversal_order: list[int], nodes_visited: int).
    """
    visited = set()
    order = []

    def _dfs_component(src):
        stack = [src]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            # Push neighbors in reverse so the first neighbour is processed first
            for nb in reversed(graph[node]):
                if nb not in visited:
                    stack.append(nb)

    _dfs_component(start)
    for node in graph:
        if node not in visited:
            _dfs_component(node)

    return order, len(order)


# ============================================================
#  DFS  (recursive)
# ============================================================

def dfs_recursive(graph, start=0):
    """
    Recursive Depth-First Search.
    Only used for small graphs (n ≤ MAX_RECURSIVE_N) to avoid stack overflow.
    Returns (traversal_order: list[int], nodes_visited: int).
    """
    visited = set()
    order = []

    def _dfs(node):
        visited.add(node)
        order.append(node)
        for nb in graph[node]:
            if nb not in visited:
                _dfs(nb)

    _dfs(start)
    for node in graph:
        if node not in visited:
            _dfs(node)

    return order, len(order)


# ============================================================
#  BENCHMARKING CONFIG
# ============================================================

RUNS = 5
SIZES = [100, 500, 1000, 2000, 5000, 10000]
MAX_RECURSIVE_N = 500   # avoid Python recursion-limit issues for deeper graphs

GRAPH_TYPES = {
    'SPARSE_RANDOM': sparse_random_graph,
    'DENSE_RANDOM':  dense_random_graph,
    'TREE':          tree_graph,
    'PATH':          path_graph,
    'CYCLE':         cycle_graph,
    'GRID':          grid_graph,
}

ALGORITHMS = {
    'BFS':           bfs,
    'DFS_ITERATIVE': dfs_iterative,
    'DFS_RECURSIVE': dfs_recursive,
}


# ============================================================
#  BENCHMARK RUNNER
# ============================================================

def benchmark_single(algo_func, graph):
    """
    Run algo_func on graph RUNS times and return (avg_ns, nodes_visited).
    avg_ns is the arithmetic mean of per-run durations in nanoseconds.
    """
    total_ns = 0
    nodes_visited = 0
    for _ in range(RUNS):
        start = time.perf_counter_ns()
        _, nv = algo_func(graph)
        end   = time.perf_counter_ns()
        total_ns    += (end - start)
        nodes_visited = nv
    return total_ns // RUNS, nodes_visited


def run_benchmarks():
    results = []
    for gt_name, gen_func in GRAPH_TYPES.items():
        print(f"\n[{gt_name}]")
        for n in SIZES:
            graph    = gen_func(n)
            actual_n = len(graph)
            row_prefix = f"  n={n:>5} (actual={actual_n})"

            for algo_name, algo_func in ALGORITHMS.items():
                # Skip recursive DFS on large or deep graphs
                if algo_name == 'DFS_RECURSIVE' and n > MAX_RECURSIVE_N:
                    continue

                avg_ns, nv = benchmark_single(algo_func, graph)
                avg_ms = avg_ns / 1_000_000

                results.append({
                    'Algorithm':     algo_name,
                    'GraphType':     gt_name,
                    'NodeCount':     actual_n,
                    'AverageTimeNs': avg_ns,
                    'AverageTimeMs': round(avg_ms, 4),
                    'NodesVisited':  nv,
                })
                print(f"  {algo_name:<16} n={n:<6} → {avg_ms:8.4f} ms  (nodes={nv})")

    return results


def save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ['Algorithm', 'GraphType', 'NodeCount',
                  'AverageTimeNs', 'AverageTimeMs', 'NodesVisited']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")


# ============================================================
#  VISUALIZATION
# ============================================================

ALGO_COLORS = {
    'BFS':           '#1f77b4',   # blue
    'DFS_ITERATIVE': '#d62728',   # red
    'DFS_RECURSIVE': '#2ca02c',   # green
}

GT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c',
    '#d62728', '#9467bd', '#8c564b',
]


def _group_data(results):
    """Returns data[algo][graph_type] = sorted list of (node_count, avg_ms)."""
    data = {}
    for row in results:
        algo = row['Algorithm']
        gt   = row['GraphType']
        data.setdefault(algo, {}).setdefault(gt, []).append(
            (row['NodeCount'], row['AverageTimeMs'])
        )
    for algo in data:
        for gt in data[algo]:
            data[algo][gt].sort()
    return data


def plot_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data       = _group_data(results)
    gt_list    = list(GRAPH_TYPES.keys())
    algo_list  = ['BFS', 'DFS_ITERATIVE']

    # ── 1. BFS vs DFS Iterative grid (one subplot per graph type) ──────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, gt in enumerate(gt_list):
        ax = axes[idx]
        for algo in algo_list:
            if algo in data and gt in data[algo]:
                pts = data[algo][gt]
                xs  = [p[0] for p in pts]
                ys  = [p[1] for p in pts]
                ax.plot(xs, ys, marker='o', label=algo,
                        color=ALGO_COLORS[algo], linewidth=2)
        ax.set_title(gt, fontweight='bold')
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Time (ms)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('BFS vs DFS (Iterative) — Time vs Graph Size',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(out_dir, 'bfs_vs_dfs_by_graphtype.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path1}")

    # ── 2. All three algorithms on SPARSE_RANDOM ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for algo in ['BFS', 'DFS_ITERATIVE', 'DFS_RECURSIVE']:
        if algo in data and 'SPARSE_RANDOM' in data[algo]:
            pts = data[algo]['SPARSE_RANDOM']
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            ax.plot(xs, ys, marker='o', label=algo,
                    color=ALGO_COLORS[algo], linewidth=2)
    ax.set_title('BFS vs DFS Variants — Sparse Random Graph', fontweight='bold')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(out_dir, 'all_algos_sparse_random.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path2}")

    # ── 3. BFS across all graph types ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, gt in enumerate(gt_list):
        if 'BFS' in data and gt in data['BFS']:
            pts = data['BFS'][gt]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            ax.plot(xs, ys, marker='o', label=gt,
                    color=GT_COLORS[i], linewidth=2)
    ax.set_title('BFS — Performance Across Graph Types', fontweight='bold')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Time (ms)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(out_dir, 'bfs_by_graphtype.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path3}")

    # ── 4. DFS iterative across all graph types ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, gt in enumerate(gt_list):
        if 'DFS_ITERATIVE' in data and gt in data['DFS_ITERATIVE']:
            pts = data['DFS_ITERATIVE'][gt]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            ax.plot(xs, ys, marker='o', label=gt,
                    color=GT_COLORS[i], linewidth=2)
    ax.set_title('DFS (Iterative) — Performance Across Graph Types', fontweight='bold')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Time (ms)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path4 = os.path.join(out_dir, 'dfs_by_graphtype.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path4}")

    # ── 5. Dense vs Sparse comparison (BFS) ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for gt in ['SPARSE_RANDOM', 'DENSE_RANDOM']:
        if 'BFS' in data and gt in data['BFS']:
            pts = data['BFS'][gt]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            ax.plot(xs, ys, marker='o', label=f'BFS — {gt}', linewidth=2)
        if 'DFS_ITERATIVE' in data and gt in data['DFS_ITERATIVE']:
            pts = data['DFS_ITERATIVE'][gt]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            ax.plot(xs, ys, marker='s', label=f'DFS — {gt}',
                    linewidth=2, linestyle='--')
    ax.set_title('Effect of Edge Density on BFS and DFS', fontweight='bold')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Time (ms)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path5 = os.path.join(out_dir, 'density_comparison.png')
    plt.savefig(path5, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path5}")


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == '__main__':
    sys.setrecursionlimit(10_000)
    random.seed(42)

    OUT_DIR = os.path.join(os.path.dirname(__file__))   # Lab3/

    print("=" * 60)
    print("  Lab 3 — DFS / BFS Empirical Analysis")
    print("=" * 60)
    print(f"Graph sizes : {SIZES}")
    print(f"Graph types : {list(GRAPH_TYPES.keys())}")
    print(f"Algorithms  : {list(ALGORITHMS.keys())}")
    print(f"Runs/test   : {RUNS}")
    print(f"Recursive DFS skipped for n > {MAX_RECURSIVE_N}")
    print()

    results = run_benchmarks()

    csv_path = os.path.join(OUT_DIR, 'dfs_bfs_results.csv')
    save_csv(results, csv_path)

    print("\nGenerating plots...")
    plot_results(results, OUT_DIR)

    print("\nAll done!")
