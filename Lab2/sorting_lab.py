import random
import time
import csv
import math
import sys
import os
from typing import List, Callable, Dict, Tuple
import matplotlib
matplotlib.use("Agg")  # non-interactive backend – saves PNGs and GIFs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# CONFIG
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SIZES = [100, 500, 1000, 2000, 5000, 10000]
EDGE_CASE_SIZES = [100, 500, 1000]  # smaller sizes for expensive edge cases
RUNS_PER_TEST = 5
CSV_FILE = "sorting_results.csv"
GRAPH_DIR = "graphs"
LIVE_VIS_SIZE = 50  # array size for live sorting visualization (kept small for animation speed)

os.makedirs(GRAPH_DIR, exist_ok=True)

# ============================================================
#  ORIGINAL SORTING ALGORITHMS
# ============================================================

#  Selection Sort (chosen algorithm)
def selection_sort(arr: List[int]) -> None:
    n = len(arr)
    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]


# Quick Sort
def quick_sort(arr: List[int]) -> None:
    _quick_sort(arr, 0, len(arr) - 1)


def _quick_sort(arr: List[int], low: int, high: int) -> None:
    if low < high:
        pi = partition(arr, low, high)
        _quick_sort(arr, low, pi - 1)
        _quick_sort(arr, pi + 1, high)


def partition(arr: List[int], low: int, high: int) -> int:
    pivot = arr[high]  # last element as pivot
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# Merge Sort
def merge_sort(arr: List[int]) -> None:
    if len(arr) < 2:
        return

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    merge_sort(left)
    merge_sort(right)

    merge(arr, left, right)


def merge(arr: List[int], left: List[int], right: List[int]) -> None:
    i = j = k = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1


# Heap Sort
def heap_sort(arr: List[int]) -> None:
    n = len(arr)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)


def heapify(arr: List[int], n: int, i: int) -> None:
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


# ============================================================
#  OPTIMIZED SORTING ALGORITHMS
# ============================================================

# Optimized Selection Sort – tracks whether a swap is needed & uses
# bidirectional (cocktail) selection to cut iterations roughly in half.
def selection_sort_optimized(arr: List[int]) -> None:
    n = len(arr)
    left = 0
    right = n - 1
    while left < right:
        min_idx = left
        max_idx = left
        for i in range(left, right + 1):
            if arr[i] < arr[min_idx]:
                min_idx = i
            if arr[i] > arr[max_idx]:
                max_idx = i
        arr[left], arr[min_idx] = arr[min_idx], arr[left]
        # If max was at the position we just swapped from, update index
        if max_idx == left:
            max_idx = min_idx
        arr[right], arr[max_idx] = arr[max_idx], arr[right]
        left += 1
        right -= 1


# Optimized Quick Sort – median-of-three pivot + insertion sort cutoff
def quick_sort_optimized(arr: List[int]) -> None:
    _quick_sort_opt(arr, 0, len(arr) - 1)


def _quick_sort_opt(arr: List[int], low: int, high: int) -> None:
    while low < high:
        # Insertion sort for small subarrays
        if high - low + 1 < 16:
            _insertion_sort_range(arr, low, high)
            return
        # Median-of-three pivot selection
        mid = (low + high) // 2
        if arr[low] > arr[mid]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[mid] > arr[high]:
            arr[mid], arr[high] = arr[high], arr[mid]
        arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
        pivot = arr[high - 1]
        i = low
        j = high - 1
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            j -= 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                break
            arr[i], arr[j] = arr[j], arr[i]
        arr[i], arr[high - 1] = arr[high - 1], arr[i]
        # Tail-call optimization: recurse on smaller partition, loop on larger
        if i - low < high - i:
            _quick_sort_opt(arr, low, i - 1)
            low = i + 1
        else:
            _quick_sort_opt(arr, i + 1, high)
            high = i - 1


def _insertion_sort_range(arr: List[int], low: int, high: int) -> None:
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# Optimized Merge Sort – uses insertion sort for small runs & avoids
# copy when already merged; in-place auxiliary buffer reuse.
def merge_sort_optimized(arr: List[int]) -> None:
    aux = arr.copy()
    _merge_sort_opt(arr, aux, 0, len(arr) - 1)


def _merge_sort_opt(arr: List[int], aux: List[int], low: int, high: int) -> None:
    if high - low < 16:
        _insertion_sort_range(arr, low, high)
        return
    mid = (low + high) // 2
    _merge_sort_opt(arr, aux, low, mid)
    _merge_sort_opt(arr, aux, mid + 1, high)
    # Skip merge if already sorted
    if arr[mid] <= arr[mid + 1]:
        return
    _merge_opt(arr, aux, low, mid, high)


def _merge_opt(arr: List[int], aux: List[int], low: int, mid: int, high: int) -> None:
    for k in range(low, high + 1):
        aux[k] = arr[k]
    i = low
    j = mid + 1
    for k in range(low, high + 1):
        if i > mid:
            arr[k] = aux[j]; j += 1
        elif j > high:
            arr[k] = aux[i]; i += 1
        elif aux[j] < aux[i]:
            arr[k] = aux[j]; j += 1
        else:
            arr[k] = aux[i]; i += 1


# Optimized Heap Sort – iterative sift-down (no recursion overhead)
def heap_sort_optimized(arr: List[int]) -> None:
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        _sift_down(arr, i, 0)


def _sift_down(arr: List[int], n: int, i: int) -> None:
    """Iterative sift-down avoids recursion overhead of heapify."""
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest


# ============================================================
#  GENERATOR-BASED SORTS FOR LIVE VISUALIZATION
#  Each yields the full array state after every meaningful step.
# ============================================================

def _gen_selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        yield arr.copy(), i, min_idx

def _gen_selection_sort_optimized(arr):
    n = len(arr)
    left, right = 0, n - 1
    while left < right:
        min_idx = left
        max_idx = left
        for i in range(left, right + 1):
            if arr[i] < arr[min_idx]:
                min_idx = i
            if arr[i] > arr[max_idx]:
                max_idx = i
        arr[left], arr[min_idx] = arr[min_idx], arr[left]
        if max_idx == left:
            max_idx = min_idx
        arr[right], arr[max_idx] = arr[max_idx], arr[right]
        yield arr.copy(), left, right
        left += 1
        right -= 1

def _gen_quick_sort(arr):
    yield from _gen_qs(arr, 0, len(arr) - 1)

def _gen_qs(arr, low, high):
    if low < high:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                yield arr.copy(), i, j
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        pi = i + 1
        yield arr.copy(), pi, high
        yield from _gen_qs(arr, low, pi - 1)
        yield from _gen_qs(arr, pi + 1, high)

def _gen_quick_sort_optimized(arr):
    """Visualizable optimized quicksort with median-of-three."""
    yield from _gen_qs_opt(arr, 0, len(arr) - 1)

def _gen_qs_opt(arr, low, high):
    if low >= high:
        return
    if high - low + 1 < 10:
        # insertion sort for small subarrays
        for i in range(low + 1, high + 1):
            key = arr[i]
            j = i - 1
            while j >= low and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
            yield arr.copy(), j + 1, i
        return
    mid = (low + high) // 2
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    arr[mid], arr[high] = arr[high], arr[mid]
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            yield arr.copy(), i, j
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    pi = i + 1
    yield arr.copy(), pi, high
    yield from _gen_qs_opt(arr, low, pi - 1)
    yield from _gen_qs_opt(arr, pi + 1, high)

def _gen_merge_sort(arr):
    yield from _gen_ms(arr, 0, len(arr) - 1)

def _gen_ms(arr, l, r):
    if l < r:
        m = (l + r) // 2
        yield from _gen_ms(arr, l, m)
        yield from _gen_ms(arr, m + 1, r)
        merged = []
        i, j = l, m + 1
        while i <= m and j <= r:
            if arr[i] <= arr[j]:
                merged.append(arr[i]); i += 1
            else:
                merged.append(arr[j]); j += 1
        while i <= m:
            merged.append(arr[i]); i += 1
        while j <= r:
            merged.append(arr[j]); j += 1
        for idx, val in enumerate(merged):
            arr[l + idx] = val
        yield arr.copy(), l, r

def _gen_merge_sort_optimized(arr):
    """Visualizable optimized merge sort."""
    yield from _gen_ms_opt(arr, 0, len(arr) - 1)

def _gen_ms_opt(arr, l, r):
    if r - l < 8:
        for i in range(l + 1, r + 1):
            key = arr[i]
            j = i - 1
            while j >= l and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        yield arr.copy(), l, r
        return
    m = (l + r) // 2
    yield from _gen_ms_opt(arr, l, m)
    yield from _gen_ms_opt(arr, m + 1, r)
    if arr[m] <= arr[m + 1]:
        return
    merged = []
    i, j = l, m + 1
    while i <= m and j <= r:
        if arr[i] <= arr[j]:
            merged.append(arr[i]); i += 1
        else:
            merged.append(arr[j]); j += 1
    while i <= m:
        merged.append(arr[i]); i += 1
    while j <= r:
        merged.append(arr[j]); j += 1
    for idx, val in enumerate(merged):
        arr[l + idx] = val
    yield arr.copy(), l, r

def _gen_heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        yield from _gen_heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        yield arr.copy(), 0, i
        yield from _gen_heapify(arr, i, 0)

def _gen_heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        yield arr.copy(), i, largest
        yield from _gen_heapify(arr, n, largest)

def _gen_heap_sort_optimized(arr):
    """Visualizable optimized heap sort with iterative sift-down."""
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        yield from _gen_sift_down(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        yield arr.copy(), 0, i
        yield from _gen_sift_down(arr, i, 0)

def _gen_sift_down(arr, n, i):
    while True:
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        yield arr.copy(), i, largest
        i = largest



# ============================================================
#  DATA GENERATION (Input Properties)
# ============================================================

def generate_array(n: int, data_type: str) -> List[int]:
    if data_type == "RANDOM":
        return [random.randint(0, 100000) for _ in range(n)]

    if data_type == "SORTED":
        return list(range(n))

    if data_type == "REVERSE_SORTED":
        return list(range(n, 0, -1))

    if data_type == "NEARLY_SORTED":
        arr = list(range(n))
        swaps = max(1, n // 20)  # ~5% random swaps
        for _ in range(swaps):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            arr[a], arr[b] = arr[b], arr[a]
        return arr

    if data_type == "MANY_DUPLICATES":
        return [random.randint(0, 10) for _ in range(n)]

    # --- EDGE CASES ---

    if data_type == "NEGATIVE_NUMBERS":
        return [random.randint(-100000, -1) for _ in range(n)]

    if data_type == "MIXED_NEGATIVE_POSITIVE":
        return [random.randint(-50000, 50000) for _ in range(n)]

    if data_type == "BIG_NUMBERS":
        return [random.randint(10**15, 10**18) for _ in range(n)]

    if data_type == "BIG_NEGATIVE_NUMBERS":
        return [random.randint(-(10**18), -(10**15)) for _ in range(n)]

    if data_type == "MIXED_BIG_NUMBERS":
        return [random.randint(-(10**18), 10**18) for _ in range(n)]

    if data_type == "SINGLE_ELEMENT":
        return [42]

    if data_type == "EMPTY_ARRAY":
        return []

    if data_type == "ALL_SAME":
        return [7] * n

    if data_type == "TWO_ELEMENTS":
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

    if data_type == "ALL_ZEROS":
        return [0] * n

    if data_type == "ALTERNATING":
        return [1 if i % 2 == 0 else -1 for i in range(n)]

    if data_type == "FLOAT_NUMBERS":
        return [random.uniform(-100000.0, 100000.0) for _ in range(n)]

    if data_type == "VERY_SMALL_FLOATS":
        return [random.uniform(-1e-10, 1e-10) for _ in range(n)]

    if data_type == "INF_VALUES":
        arr = [random.randint(-1000, 1000) for _ in range(n)]
        # Inject some inf / -inf values at random positions
        for _ in range(max(1, n // 10)):
            idx = random.randint(0, n - 1)
            arr[idx] = float('inf') if random.random() > 0.5 else float('-inf')
        return arr

    if data_type == "MAX_MIN_INT":
        arr = [random.randint(-1000, 1000) for _ in range(n)]
        arr[0] = sys.maxsize
        arr[-1] = -sys.maxsize - 1
        return arr

    raise ValueError(f"Unknown data type: {data_type}")


# ============================================================
#  UTILITIES
# ============================================================

def is_sorted(arr: List) -> bool:
    """Check if array is sorted in non-decreasing order."""
    if len(arr) <= 1:
        return True
    return all(arr[i - 1] <= arr[i] for i in range(1, len(arr)))


def benchmark(sort_func: Callable[[List[int]], None], base_array: List[int], runs: int) -> int:
    """Returns average execution time in nanoseconds."""
    total_ns = 0
    for _ in range(runs):
        arr = base_array.copy()
        start = time.perf_counter_ns()
        sort_func(arr)
        end = time.perf_counter_ns()
        if not is_sorted(arr):
            raise RuntimeError(f"{sort_func.__name__} failed: array is not sorted")
        total_ns += (end - start)
    return total_ns // runs


def print_result(algorithm: str, size: int, avg_ns: int) -> None:
    avg_ms = avg_ns / 1_000_000
    print(f"{algorithm:<25} {size:<10} {avg_ns:<18} {avg_ms:<15.6f}")


# ============================================================
#  LIVE VISUALIZATION  (animated bar-chart for each sort)
# ============================================================

def run_live_visualization():
    """Save an animated GIF for every sorting algorithm on RANDOM data."""

    vis_algorithms = {
        "Selection Sort":           _gen_selection_sort,
        "Selection Sort (Opt.)":    _gen_selection_sort_optimized,
        "Quick Sort":               _gen_quick_sort,
        "Quick Sort (Opt.)":        _gen_quick_sort_optimized,
        "Merge Sort":               _gen_merge_sort,
        "Merge Sort (Opt.)":        _gen_merge_sort_optimized,
        "Heap Sort":                _gen_heap_sort,
        "Heap Sort (Opt.)":         _gen_heap_sort_optimized,
    }

    random.seed(RANDOM_SEED)
    base = [random.randint(1, 100) for _ in range(LIVE_VIS_SIZE)]

    for name, gen_func in vis_algorithms.items():
        arr = base.copy()
        frames = list(gen_func(arr))  # collect all frames

        if not frames:
            continue

        # Limit frames to keep GIF size manageable
        max_frames = 200
        if len(frames) > max_frames:
            step = len(frames) // max_frames
            frames = frames[::step]
            # Always include the last frame (sorted result)
            frames.append((arr.copy(), -1, -1))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"Live Sorting — {name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        bar_rects = ax.bar(range(len(base)), base.copy(), color="steelblue",
                           edgecolor="black", linewidth=0.5)
        ax.set_xlim(-1, len(base))
        ax.set_ylim(0, max(base) + 10)
        iteration_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)

        def update(frame_data, rects=bar_rects, txt=iteration_text):
            state, idx1, idx2 = frame_data
            for rect, val in zip(rects, state):
                rect.set_height(val)
                rect.set_color("steelblue")
            if 0 <= idx1 < len(rects):
                rects[idx1].set_color("crimson")
            if 0 <= idx2 < len(rects):
                rects[idx2].set_color("limegreen")
            txt.set_text(f"Step {update.counter}")
            update.counter += 1
            return list(rects) + [txt]
        update.counter = 0

        anim = animation.FuncAnimation(
            fig, update, frames=frames, interval=50, blit=True, repeat=False
        )

        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        gif_path = os.path.join(GRAPH_DIR, f"anim_{safe_name}.gif")
        anim.save(gif_path, writer="pillow", fps=20)
        plt.close(fig)
        print(f"  ✓ Saved animation: {gif_path}")


# ============================================================
#  PNG GRAPH GENERATION
# ============================================================

def generate_graphs(results: List[dict]):
    """
    Create and save PNG comparison graphs from benchmark results.
    Generates:
      1. One graph per data-type  (all algorithms overlaid)
      2. One graph per algorithm  (all data-types overlaid)
      3. Original vs Optimized comparison per algorithm family
      4. Combined overview graph  (RANDOM data, all algorithms)
    """

    # ---------- helpers ----------
    def _plot_and_save(title, xlabel, ylabel, series: Dict[str, Tuple[list, list]], filepath):
        """series = { label: (xs, ys) }"""
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, (xs, ys) in series.items():
            ax.plot(xs, ys, marker="o", linewidth=2, label=label)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved {filepath}")

    # Index results by (algorithm, data_type) → [(size, ms)]
    index: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
    for r in results:
        key = (r["algorithm"], r["data_type"])
        index.setdefault(key, []).append((r["size"], r["avg_ms"]))
    for k in index:
        index[k].sort()

    all_algos = sorted({r["algorithm"] for r in results})
    all_data_types = sorted({r["data_type"] for r in results})

    # 1. One graph per data-type
    for dt in all_data_types:
        series = {}
        for algo in all_algos:
            if (algo, dt) in index:
                pts = index[(algo, dt)]
                series[algo] = ([p[0] for p in pts], [p[1] for p in pts])
        if series:
            safe = dt.lower().replace(" ", "_")
            _plot_and_save(
                f"All Algorithms — {dt}",
                "Array Size", "Avg Time (ms)",
                series,
                os.path.join(GRAPH_DIR, f"by_datatype_{safe}.png"),
            )

    # 2. One graph per algorithm
    for algo in all_algos:
        series = {}
        for dt in all_data_types:
            if (algo, dt) in index:
                pts = index[(algo, dt)]
                series[dt] = ([p[0] for p in pts], [p[1] for p in pts])
        if series:
            safe = algo.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
            _plot_and_save(
                f"{algo} — All Data Types",
                "Array Size", "Avg Time (ms)",
                series,
                os.path.join(GRAPH_DIR, f"by_algo_{safe}.png"),
            )

    # 3. Original vs Optimized comparison (on RANDOM data)
    families = {
        "Selection Sort": ("SelectionSort", "SelectionSort(Opt)"),
        "Quick Sort":     ("QuickSort",     "QuickSort(Opt)"),
        "Merge Sort":     ("MergeSort",     "MergeSort(Opt)"),
        "Heap Sort":      ("HeapSort",      "HeapSort(Opt)"),
    }
    for family_name, (orig, opt) in families.items():
        series = {}
        for dt in ["RANDOM", "SORTED", "REVERSE_SORTED"]:
            if (orig, dt) in index:
                pts = index[(orig, dt)]
                series[f"{orig} – {dt}"] = ([p[0] for p in pts], [p[1] for p in pts])
            if (opt, dt) in index:
                pts = index[(opt, dt)]
                series[f"{opt} – {dt}"] = ([p[0] for p in pts], [p[1] for p in pts])
        if series:
            safe = family_name.lower().replace(" ", "_")
            _plot_and_save(
                f"Original vs Optimized — {family_name}",
                "Array Size", "Avg Time (ms)",
                series,
                os.path.join(GRAPH_DIR, f"orig_vs_opt_{safe}.png"),
            )

    # 4. Combined overview on RANDOM
    series = {}
    for algo in all_algos:
        if (algo, "RANDOM") in index:
            pts = index[(algo, "RANDOM")]
            series[algo] = ([p[0] for p in pts], [p[1] for p in pts])
    if series:
        _plot_and_save(
            "All Algorithms on RANDOM Data",
            "Array Size", "Avg Time (ms)",
            series,
            os.path.join(GRAPH_DIR, "overview_random.png"),
        )

    # 5. Bar-chart comparison at largest common size on RANDOM
    largest = max(SIZES)
    bar_data = {}
    for algo in all_algos:
        if (algo, "RANDOM") in index:
            pts = dict(index[(algo, "RANDOM")])
            if largest in pts:
                bar_data[algo] = pts[largest]
    if bar_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        names = list(bar_data.keys())
        vals = list(bar_data.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Comparison at n = {largest} (RANDOM)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Avg Time (ms)", fontsize=12)
        ax.set_xlabel("Algorithm", fontsize=12)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(GRAPH_DIR, "bar_comparison_random.png"), dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved {os.path.join(GRAPH_DIR, 'bar_comparison_random.png')}")


# ============================================================
#  MAIN  (Empirical Analysis)
# ============================================================

def main():
    algorithms: Dict[str, Callable[[List[int]], None]] = {
        "QuickSort":          quick_sort,
        "MergeSort":          merge_sort,
        "HeapSort":           heap_sort,
        "SelectionSort":      selection_sort,        # chosen algorithm
        "QuickSort(Opt)":     quick_sort_optimized,
        "MergeSort(Opt)":     merge_sort_optimized,
        "HeapSort(Opt)":      heap_sort_optimized,
        "SelectionSort(Opt)": selection_sort_optimized,
    }

    data_types = [
        "RANDOM",
        "SORTED",
        "REVERSE_SORTED",
        "NEARLY_SORTED",
        "MANY_DUPLICATES",
    ]

    edge_case_types = [
        "NEGATIVE_NUMBERS",
        "MIXED_NEGATIVE_POSITIVE",
        "BIG_NUMBERS",
        "BIG_NEGATIVE_NUMBERS",
        "MIXED_BIG_NUMBERS",
        "SINGLE_ELEMENT",
        "EMPTY_ARRAY",
        "ALL_SAME",
        "TWO_ELEMENTS",
        "ALL_ZEROS",
        "ALTERNATING",
        "FLOAT_NUMBERS",
        "VERY_SMALL_FLOATS",
        "INF_VALUES",
        "MAX_MIN_INT",
    ]

    print("=" * 70)
    print("Laboratory Nr. 2 - Empirical Analysis of Sorting Algorithms")
    print("Algorithms: QuickSort, MergeSort, HeapSort, SelectionSort")
    print("  + Optimized variants of each")
    print("Chosen algorithm: SelectionSort")
    print("=" * 70)

    # ---------- PHASE 1 — Live Visualization ----------
    print("\n>>> PHASE 1: Generating sorting animation GIFs …")
    run_live_visualization()

    # ---------- PHASE 2 — Benchmarking ----------
    print("\n>>> PHASE 2: Benchmarking all algorithms …\n")

    sys.setrecursionlimit(50000)

    all_results: List[dict] = []

    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "DataType", "ArraySize", "AverageTimeNs", "AverageTimeMs"])

        # --- Standard data types with full SIZES ---
        for data_type in data_types:
            print(f"DATA TYPE: {data_type}")
            print(f"{'Algorithm':<25} {'Size':<10} {'Avg Time (ns)':<18} {'Avg Time (ms)':<15}")

            for n in SIZES:
                random.seed(RANDOM_SEED)
                base_array = generate_array(n, data_type)

                for name, func in algorithms.items():
                    # skip Selection Sort variants on very large inputs
                    if "SelectionSort" in name and n > 10000:
                        continue

                    avg_ns = benchmark(func, base_array, RUNS_PER_TEST)
                    avg_ms = avg_ns / 1_000_000
                    print_result(name, n, avg_ns)

                    writer.writerow([name, data_type, n, avg_ns, f"{avg_ms:.6f}"])
                    all_results.append({
                        "algorithm": name, "data_type": data_type,
                        "size": n, "avg_ns": avg_ns, "avg_ms": avg_ms,
                    })

                print()

        # --- Edge case data types ---
        print("=" * 70)
        print("EDGE CASES (negative, big, float, inf, empty, single, etc.)")
        print("=" * 70)

        for data_type in edge_case_types:
            print(f"\nDATA TYPE: {data_type}")
            print(f"{'Algorithm':<25} {'Size':<10} {'Avg Time (ns)':<18} {'Avg Time (ms)':<15}")

            if data_type == "SINGLE_ELEMENT":
                sizes_to_test = [1]
            elif data_type == "EMPTY_ARRAY":
                sizes_to_test = [0]
            elif data_type == "TWO_ELEMENTS":
                sizes_to_test = [2]
            else:
                sizes_to_test = EDGE_CASE_SIZES

            for n in sizes_to_test:
                random.seed(RANDOM_SEED)
                base_array = generate_array(n, data_type)

                for name, func in algorithms.items():
                    if "SelectionSort" in name and n > 10000:
                        continue

                    avg_ns = benchmark(func, base_array, RUNS_PER_TEST)
                    avg_ms = avg_ns / 1_000_000
                    print_result(name, n, avg_ns)

                    writer.writerow([name, data_type, n, avg_ns, f"{avg_ms:.6f}"])
                    all_results.append({
                        "algorithm": name, "data_type": data_type,
                        "size": n, "avg_ns": avg_ns, "avg_ms": avg_ms,
                    })

                print()

    print(f"\nCSV exported successfully: {CSV_FILE}")

    # ---------- PHASE 3 — Generate PNG graphs ----------
    print("\n>>> PHASE 3: Generating PNG graphs …")
    generate_graphs(all_results)

    print(f"\n✅ Done! All graphs saved in '{GRAPH_DIR}/' folder.")


if __name__ == "__main__":
    main()