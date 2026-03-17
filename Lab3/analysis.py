#!/usr/bin/env python3
"""
Empirical Analysis of DFS and BFS Graph Traversal Algorithms
Laboratory Work
"""

import time
import random
import tracemalloc
import json
import os
from collections import deque

# ============================================================
# 1. Algorithm Implementations
# ============================================================

def bfs(graph, start, goal=None):
    """Breadth-First Search implementation."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []
    parent = {start: None}

    while queue:
        node = queue.popleft()
        order.append(node)

        if goal is not None and node == goal:
            # Reconstruct path
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return order, path[::-1]

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    return order, []


def dfs(graph, start, goal=None):
    """Depth-First Search implementation (iterative with stack)."""
    visited = set()
    stack = [start]
    order = []
    parent = {start: None}

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        if goal is not None and node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return order, path[::-1]

        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                parent[neighbor] = node
                stack.append(neighbor)

    return order, []


# ============================================================
# 2. Graph Generators
# ============================================================

def generate_sparse_graph(n, edge_factor=2):
    """Sparse graph: ~2*n edges (adjacency list)."""
    graph = {i: [] for i in range(n)}
    edges = set()
    # Ensure connectivity with a spanning tree
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(1, n):
        u, v = nodes[i - 1], nodes[i]
        graph[u].append(v)
        graph[v].append(u)
        edges.add((min(u, v), max(u, v)))

    target_edges = edge_factor * n
    attempts = 0
    while len(edges) < target_edges and attempts < target_edges * 10:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and (min(u, v), max(u, v)) not in edges:
            graph[u].append(v)
            graph[v].append(u)
            edges.add((min(u, v), max(u, v)))
        attempts += 1
    return graph, len(edges)


def generate_dense_graph(n, density=0.4):
    """Dense graph: ~density * n*(n-1)/2 edges."""
    graph = {i: [] for i in range(n)}
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < density:
                graph[i].append(j)
                graph[j].append(i)
                edge_count += 1
    return graph, edge_count


def generate_tree(n):
    """Random tree (connected, no cycles)."""
    graph = {i: [] for i in range(n)}
    for i in range(1, n):
        parent = random.randint(0, i - 1)
        graph[parent].append(i)
        graph[i].append(parent)
    return graph, n - 1


def generate_grid_graph(side):
    """Grid graph (side x side)."""
    n = side * side
    graph = {i: [] for i in range(n)}
    edge_count = 0
    for r in range(side):
        for c in range(side):
            node = r * side + c
            if c + 1 < side:
                neighbor = r * side + (c + 1)
                graph[node].append(neighbor)
                graph[neighbor].append(node)
                edge_count += 1
            if r + 1 < side:
                neighbor = (r + 1) * side + c
                graph[node].append(neighbor)
                graph[neighbor].append(node)
                edge_count += 1
    return graph, edge_count


# ============================================================
# 3. Measurement Functions
# ============================================================

def measure_algorithm(algo_func, graph, start, goal, runs=3):
    """Measure execution time, memory, nodes visited, path length."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        order, path = algo_func(graph, start, goal)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Memory measurement (single run)
    tracemalloc.start()
    order, path = algo_func(graph, start, goal)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "avg_time_ms": sum(times) / len(times) * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
        "peak_memory_kb": peak_mem / 1024,
        "nodes_visited": len(order),
        "path_length": len(path) if path else 0,
    }


# ============================================================
# 4. Run Experiments
# ============================================================

def run_experiments():
    random.seed(42)
    results = {}

    # --- Experiment 1: Scaling with graph size (Sparse) ---
    print("Experiment 1: Sparse graph scaling...")
    # reduced sizes for faster, reproducible runs
    sizes = [100, 500, 1000, 2000, 5000, 10000]
    exp1 = {"sizes": sizes, "bfs": [], "dfs": []}
    for n in sizes:
        graph, edges = generate_sparse_graph(n)
        goal = n - 1
        bfs_res = measure_algorithm(bfs, graph, 0, goal)
        dfs_res = measure_algorithm(dfs, graph, 0, goal)
        exp1["bfs"].append(bfs_res)
        exp1["dfs"].append(dfs_res)
        print(f"  n={n:>6}, edges={edges:>7} | BFS: {bfs_res['avg_time_ms']:.3f}ms | DFS: {dfs_res['avg_time_ms']:.3f}ms")
    results["sparse_scaling"] = exp1

    # --- Experiment 2: Scaling with graph size (Dense) ---
    print("\nExperiment 2: Dense graph scaling...")
    sizes_dense = [50, 100, 200, 400, 600, 800, 1000]
    exp2 = {"sizes": sizes_dense, "bfs": [], "dfs": []}
    for n in sizes_dense:
        graph, edges = generate_dense_graph(n, density=0.4)
        goal = n - 1
        bfs_res = measure_algorithm(bfs, graph, 0, goal)
        dfs_res = measure_algorithm(dfs, graph, 0, goal)
        exp2["bfs"].append(bfs_res)
        exp2["dfs"].append(dfs_res)
        print(f"  n={n:>5}, edges={edges:>7} | BFS: {bfs_res['avg_time_ms']:.3f}ms | DFS: {dfs_res['avg_time_ms']:.3f}ms")
    results["dense_scaling"] = exp2

    # --- Experiment 3: Tree graphs ---
    print("\nExperiment 3: Tree graph scaling...")
    sizes_tree = [100, 500, 1000, 5000, 10000]
    exp3 = {"sizes": sizes_tree, "bfs": [], "dfs": []}
    for n in sizes_tree:
        graph, edges = generate_tree(n)
        goal = n - 1
        bfs_res = measure_algorithm(bfs, graph, 0, goal)
        dfs_res = measure_algorithm(dfs, graph, 0, goal)
        exp3["bfs"].append(bfs_res)
        exp3["dfs"].append(dfs_res)
        print(f"  n={n:>6}, edges={edges:>7} | BFS: {bfs_res['avg_time_ms']:.3f}ms | DFS: {dfs_res['avg_time_ms']:.3f}ms")
    results["tree_scaling"] = exp3

    # --- Experiment 4: Grid graph (path quality) ---
    print("\nExperiment 4: Grid graph (path quality)...")
    sides = [5, 10, 15, 20, 30]
    exp4 = {"sides": sides, "sizes": [s * s for s in sides], "bfs": [], "dfs": []}
    for side in sides:
        n = side * side
        graph, edges = generate_grid_graph(side)
        goal = n - 1  # bottom-right corner
        bfs_res = measure_algorithm(bfs, graph, 0, goal)
        dfs_res = measure_algorithm(dfs, graph, 0, goal)
        exp4["bfs"].append(bfs_res)
        exp4["dfs"].append(dfs_res)
        print(f"  grid {side:>2}x{side:>2} (n={n:>4}) | BFS path={bfs_res['path_length']:>4} | DFS path={dfs_res['path_length']:>4}")
    results["grid_path"] = exp4

    # --- Experiment 5: Density impact ---
    print("\nExperiment 5: Density impact (n=1000)...")
    densities = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    exp5 = {"densities": densities, "bfs": [], "dfs": [], "edge_counts": []}
    for d in densities:
        graph, edges = generate_dense_graph(1000, density=d)
        bfs_res = measure_algorithm(bfs, graph, 0, 999)
        dfs_res = measure_algorithm(dfs, graph, 0, 999)
        exp5["bfs"].append(bfs_res)
        exp5["dfs"].append(dfs_res)
        exp5["edge_counts"].append(edges)
        print(f"  density={d:.2f}, edges={edges:>7} | BFS: {bfs_res['avg_time_ms']:.3f}ms | DFS: {dfs_res['avg_time_ms']:.3f}ms")
    results["density_impact"] = exp5

    return results


# ============================================================
# 5. Generate Charts
# ============================================================

def generate_charts(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    # Ensure output directory exists
    os.makedirs('charts', exist_ok=True)

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    colors = {'bfs': '#2196F3', 'dfs': '#FF5722'}

    chart_paths = []

    # --- Chart 1: Sparse Graph - Execution Time ---
    fig, ax = plt.subplots(figsize=(8, 5))
    exp = results["sparse_scaling"]
    ax.plot(exp["sizes"], [r["avg_time_ms"] for r in exp["bfs"]], 'o-',
            color=colors['bfs'], label='BFS', linewidth=2, markersize=6)
    ax.plot(exp["sizes"], [r["avg_time_ms"] for r in exp["dfs"]], 's-',
            color=colors['dfs'], label='DFS', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Vertices')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Execution Time vs Graph Size (Sparse Graph, ~2n edges)')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    path = 'charts/chart1_sparse_time.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_paths.append(path)

    # --- Chart 2: Dense Graph - Execution Time ---
    fig, ax = plt.subplots(figsize=(8, 5))
    exp = results["dense_scaling"]
    ax.plot(exp["sizes"], [r["avg_time_ms"] for r in exp["bfs"]], 'o-',
            color=colors['bfs'], label='BFS', linewidth=2, markersize=6)
    ax.plot(exp["sizes"], [r["avg_time_ms"] for r in exp["dfs"]], 's-',
            color=colors['dfs'], label='DFS', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Vertices')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Execution Time vs Graph Size (Dense Graph, density=0.4)')
    ax.legend()
    plt.tight_layout()
    path = 'charts/chart2_dense_time.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_paths.append(path)

    # --- Chart 3: Memory Usage Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (key, title) in enumerate([
        ("sparse_scaling", "Sparse Graph"),
        ("dense_scaling", "Dense Graph"),
        ("tree_scaling", "Tree")
    ]):
        exp = results[key]
        sizes = exp["sizes"]
        axes[idx].plot(sizes, [r["peak_memory_kb"] for r in exp["bfs"]], 'o-',
                       color=colors['bfs'], label='BFS', linewidth=2, markersize=5)
        axes[idx].plot(sizes, [r["peak_memory_kb"] for r in exp["dfs"]], 's-',
                       color=colors['dfs'], label='DFS', linewidth=2, markersize=5)
        axes[idx].set_xlabel('Number of Vertices')
        axes[idx].set_ylabel('Peak Memory (KB)')
        axes[idx].set_title(f'Memory: {title}')
        axes[idx].legend()
        axes[idx].set_xscale('log')
        axes[idx].set_yscale('log')
    plt.tight_layout()
    path = 'charts/chart3_memory.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_paths.append(path)

    # --- Chart 4: Grid Path Length Comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    exp = results["grid_path"]
    sizes = exp["sizes"]
    bfs_paths = [r["path_length"] for r in exp["bfs"]]
    dfs_paths = [r["path_length"] for r in exp["dfs"]]
    optimal = [int(2 * (s**0.5 - 1)) + 1 for s in sizes]  # Manhattan distance on grid

    x = np.arange(len(sizes))
    width = 0.3
    ax.bar(x - width/2, bfs_paths, width, label='BFS Path Length', color=colors['bfs'], alpha=0.85)
    ax.bar(x + width/2, dfs_paths, width, label='DFS Path Length', color=colors['dfs'], alpha=0.85)
    ax.plot(x, optimal, 'k--', label='Optimal (Manhattan)', linewidth=1.5, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.set_xlabel('Grid Size (total nodes)')
    ax.set_ylabel('Path Length (edges)')
    ax.set_title('Path Length: BFS vs DFS on Grid Graphs (0,0 → n-1,n-1)')
    ax.legend()
    plt.tight_layout()
    path = 'charts/chart4_path_quality.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_paths.append(path)

    # --- Chart 5: Density Impact ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    exp = results["density_impact"]
    densities = exp["densities"]

    ax1.plot(densities, [r["avg_time_ms"] for r in exp["bfs"]], 'o-',
             color=colors['bfs'], label='BFS', linewidth=2, markersize=6)
    ax1.plot(densities, [r["avg_time_ms"] for r in exp["dfs"]], 's-',
             color=colors['dfs'], label='DFS', linewidth=2, markersize=6)
    ax1.set_xlabel('Edge Density')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Time vs Edge Density (n=1000)')
    ax1.legend()

    ax2.plot(densities, [r["peak_memory_kb"] for r in exp["bfs"]], 'o-',
             color=colors['bfs'], label='BFS', linewidth=2, markersize=6)
    ax2.plot(densities, [r["peak_memory_kb"] for r in exp["dfs"]], 's-',
             color=colors['dfs'], label='DFS', linewidth=2, markersize=6)
    ax2.set_xlabel('Edge Density')
    ax2.set_ylabel('Peak Memory (KB)')
    ax2.set_title('Memory vs Edge Density (n=1000)')
    ax2.legend()

    plt.tight_layout()
    path = 'charts/chart5_density.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_paths.append(path)

    # --- Chart 6: Nodes Visited ---
    fig, ax = plt.subplots(figsize=(8, 5))
    exp = results["sparse_scaling"]
    ax.plot(exp["sizes"], [r["nodes_visited"] for r in exp["bfs"]], 'o-',
            color=colors['bfs'], label='BFS', linewidth=2, markersize=6)
    ax.plot(exp["sizes"], [r["nodes_visited"] for r in exp["dfs"]], 's-',
            color=colors['dfs'], label='DFS', linewidth=2, markersize=6)
    ax.plot(exp["sizes"], exp["sizes"], 'k--', label='Total Vertices', linewidth=1, alpha=0.5)
    ax.set_xlabel('Number of Vertices')
    ax.set_ylabel('Nodes Visited')
    ax.set_title('Nodes Visited Before Finding Goal (Sparse Graph)')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    path = 'charts/chart6_nodes_visited.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_paths.append(path)

    return chart_paths


if __name__ == "__main__":
    results = run_experiments()

    # Save raw results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    chart_paths = generate_charts(results)
    print(f"\nGenerated {len(chart_paths)} charts.")
    print("Done.")
