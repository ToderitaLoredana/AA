"""
Lab 3 — BFS & DFS Animated GIF Demo
=====================================
Creates two GIFs (bfs_demo.gif, dfs_demo.gif) that visualise the traversal
of a small, hand-crafted 20-node graph step by step.

Requirements: matplotlib, Pillow  (pip install matplotlib pillow)
"""

import math
import os
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

# ============================================================
#  DEMO GRAPH  (20 nodes, tree-like with a few cross-edges)
# ============================================================

DEMO_N = 20
DEMO_EDGES = [
    # root → level-1
    (0, 1), (0, 2), (0, 3),
    # level-1 → level-2
    (1, 4), (1, 5),
    (2, 6), (2, 7),
    (3, 8), (3, 9),
    # level-2 → level-3
    (4, 10), (4, 11),
    (5, 12), (5, 13),
    (6, 14),
    (7, 15), (7, 16),
    (8, 17),
    (9, 18), (9, 19),
    # cross-edges (makes it a proper graph, not just a tree)
    (10, 12),
    (14, 17),
]

def build_demo_graph():
    g = {i: [] for i in range(DEMO_N)}
    for u, v in DEMO_EDGES:
        g[u].append(v)
        g[v].append(u)
    return g


# ============================================================
#  HIERARCHICAL LAYOUT  (BFS-derived, top-down)
# ============================================================

def hierarchical_layout(graph, root=0):
    """
    Compute (x, y) positions with the root at the top.
    Nodes at the same BFS depth share the same y-coordinate.
    """
    levels = {}
    level_buckets = {}
    visited = {root}
    queue = collections.deque([(root, 0)])

    while queue:
        node, depth = queue.popleft()
        levels[node] = depth
        level_buckets.setdefault(depth, []).append(node)
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))

    # Any node unreachable from root gets its own row at the bottom
    for node in graph:
        if node not in levels:
            depth = max(level_buckets.keys()) + 1
            levels[node] = depth
            level_buckets.setdefault(depth, []).append(node)

    max_depth = max(level_buckets.keys())
    positions = {}
    for depth, nodes in level_buckets.items():
        y = 1.0 - 2.0 * depth / max(max_depth, 1)
        for i, node in enumerate(nodes):
            x = -1.0 + 2.0 * (i + 1) / (len(nodes) + 1)
            positions[node] = (x, y)
    return positions


# ============================================================
#  STEP GENERATORS  (yield state snapshots)
# ============================================================

# Each snapshot = (visited_set, current_node_or_None, frontier_set)
#   visited   — nodes already fully processed (green)
#   current   — node being expanded right now (orange-red)
#   frontier  — nodes in the queue / stack, waiting (yellow)

def bfs_steps(graph, start=0):
    visited  = set()
    in_queue = {start}
    queue    = collections.deque([start])

    yield frozenset(visited), None, frozenset(in_queue)   # initial

    while queue:
        node = queue.popleft()
        in_queue.discard(node)
        visited.add(node)

        for nb in graph[node]:
            if nb not in visited and nb not in in_queue:
                in_queue.add(nb)
                queue.append(nb)

        yield frozenset(visited), node, frozenset(in_queue)


def dfs_steps(graph, start=0):
    visited  = set()
    in_stack = {start}
    stack    = [start]

    yield frozenset(visited), None, frozenset(in_stack)   # initial

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        in_stack.discard(node)
        visited.add(node)

        for nb in reversed(graph[node]):
            if nb not in visited and nb not in in_stack:
                in_stack.add(nb)
                stack.append(nb)

        yield frozenset(visited), node, frozenset(in_stack)


# ============================================================
#  COLOUR PALETTE
# ============================================================

C_UNVISITED = '#D0D0D0'    # light grey
C_FRONTIER  = '#FFD700'    # gold  — in queue / stack
C_CURRENT   = '#FF4500'    # orange-red — being expanded
C_VISITED   = '#3CB371'    # medium sea green — done
C_EDGE      = '#A0A0A0'    # grey edges
NODE_R      = 0.07         # node radius in data coordinates


# ============================================================
#  GIF BUILDER
# ============================================================

def _node_color(node, visited, current, frontier):
    if node == current:
        return C_CURRENT
    if node in visited:
        return C_VISITED
    if node in frontier:
        return C_FRONTIER
    return C_UNVISITED


def make_gif(title, steps, graph, positions, out_path, fps=2):
    """Render steps as an animated GIF saved to out_path."""

    # Deduplicate edges for drawing
    drawn_edges = set()
    edge_list   = []
    for u in graph:
        for v in graph[u]:
            key = (min(u, v), max(u, v))
            if key not in drawn_edges:
                drawn_edges.add(key)
                edge_list.append((u, v))

    fig, ax = plt.subplots(figsize=(10, 8))

    legend_elements = [
        mpatches.Patch(facecolor=C_UNVISITED, edgecolor='black', label='Unvisited'),
        mpatches.Patch(facecolor=C_FRONTIER,  edgecolor='black', label='Queue / Stack'),
        mpatches.Patch(facecolor=C_CURRENT,   edgecolor='black', label='Current node'),
        mpatches.Patch(facecolor=C_VISITED,   edgecolor='black', label='Visited'),
    ]

    def draw(frame_idx):
        ax.clear()
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect('equal')
        ax.axis('off')

        visited, current, frontier = steps[frame_idx]

        # ── edges ──
        for u, v in edge_list:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], '-', color=C_EDGE,
                    linewidth=1.4, zorder=1)

        # ── nodes ──
        for node in graph:
            x, y  = positions[node]
            color = _node_color(node, visited, current, frontier)
            circle = plt.Circle((x, y), NODE_R, color=color,
                                 ec='#333333', linewidth=1.5, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, str(node), ha='center', va='center',
                    fontsize=8, fontweight='bold', zorder=3)

        # ── legend ──
        ax.legend(handles=legend_elements, loc='lower right',
                  fontsize=8, framealpha=0.9)

        # ── title + step counter ──
        step_info = (f"Step {frame_idx}/{len(steps) - 1}"
                     + (f"  —  expanding node {current}" if current is not None else "  —  initialising"))
        ax.set_title(f"{title}\n{step_info}",
                     fontsize=11, fontweight='bold', pad=8)

        # ── visited counter ──
        ax.text(-1.3, -1.28,
                f"Visited: {len(visited)} / {len(graph)} nodes",
                fontsize=9, va='bottom')

    ani = animation.FuncAnimation(
        fig, draw,
        frames=len(steps),
        interval=1000 // fps,
        repeat=True,
    )

    try:
        ani.save(out_path, writer='pillow', fps=fps)
        print(f"Saved → {out_path}")
    except Exception as exc:
        print(f"ERROR saving GIF: {exc}")
        print("Make sure Pillow is installed:  pip install pillow")
    finally:
        plt.close(fig)


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))   # Lab3/
    os.makedirs(out_dir, exist_ok=True)

    graph     = build_demo_graph()
    positions = hierarchical_layout(graph, root=0)

    # ── BFS GIF ──────────────────────────────────────────────
    print("Generating BFS demo GIF…")
    bfs_state = list(bfs_steps(graph, start=0))
    make_gif(
        title    = 'BFS — Breadth-First Search',
        steps    = bfs_state,
        graph    = graph,
        positions= positions,
        out_path = os.path.join(out_dir, 'bfs_demo.gif'),
        fps      = 2,
    )

    # ── DFS GIF ──────────────────────────────────────────────
    print("Generating DFS demo GIF…")
    dfs_state = list(dfs_steps(graph, start=0))
    make_gif(
        title    = 'DFS — Depth-First Search',
        steps    = dfs_state,
        graph    = graph,
        positions= positions,
        out_path = os.path.join(out_dir, 'dfs_demo.gif'),
        fps      = 2,
    )

    print("\nDone!  Files written to Lab3/")
