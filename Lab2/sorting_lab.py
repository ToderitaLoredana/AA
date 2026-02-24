import random
import time
import csv
import math
import sys
from typing import List, Callable, Dict

# CONFIG
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SIZES = [100, 500, 1000, 2000, 5000, 10000]
EDGE_CASE_SIZES = [100, 500, 1000]  # smaller sizes for expensive edge cases
RUNS_PER_TEST = 5
CSV_FILE = "sorting_results.csv"

# SORTING ALGORITHMS

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


#Merge Sort 
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


#Heap Sort 
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



# DATA GENERATION (Input Properties)
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


# UTILITIES
def is_sorted(arr: List) -> bool:
    """Check if array is sorted in non-decreasing order. Handles int, float, inf."""
    if len(arr) <= 1:
        return True
    return all(arr[i - 1] <= arr[i] for i in range(1, len(arr)))


def benchmark(sort_func: Callable[[List[int]], None], base_array: List[int], runs: int) -> int:
    """
    Returns average execution time in nanoseconds.
    """
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
    print(f"{algorithm:<14} {size:<10} {avg_ns:<18} {avg_ms:<15.6f}")


# MAIN (Empirical Analysis)
def main():
    algorithms: Dict[str, Callable[[List[int]], None]] = {
        "QuickSort": quick_sort,
        "MergeSort": merge_sort,
        "HeapSort": heap_sort,
        "SelectionSort": selection_sort,  # chosen algorithm
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

    print("Laboratory Nr. 2 - Empirical Analysis of Sorting Algorithms")
    print("Algorithms: QuickSort, MergeSort, HeapSort, SelectionSort")
    print("Chosen algorithm: SelectionSort\n")

    # Increase recursion limit for QuickSort on large sorted inputs
    sys.setrecursionlimit(50000)

    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "DataType", "ArraySize", "AverageTimeNs", "AverageTimeMs"])

        # --- Standard data types with full SIZES ---
        for data_type in data_types:
            print(f"DATA TYPE: {data_type}")
            print(f"{'Algorithm':<14} {'Size':<10} {'Avg Time (ns)':<18} {'Avg Time (ms)':<15}")

            for n in SIZES:
                base_array = generate_array(n, data_type)

                for name, func in algorithms.items():
                    if name == "SelectionSort" and n > 10000:
                        continue

                    avg_ns = benchmark(func, base_array, RUNS_PER_TEST)
                    print_result(name, n, avg_ns)

                    writer.writerow([
                        name,
                        data_type,
                        n,
                        avg_ns,
                        f"{avg_ns / 1_000_000:.6f}"
                    ])

                print()

        # --- Edge case data types ---
        print("=" * 70)
        print("EDGE CASES (negative, big, float, inf, empty, single, etc.)")
        print("=" * 70)

        for data_type in edge_case_types:
            print(f"\nDATA TYPE: {data_type}")
            print(f"{'Algorithm':<14} {'Size':<10} {'Avg Time (ns)':<18} {'Avg Time (ms)':<15}")

            # For fixed-size edge cases, override sizes
            if data_type == "SINGLE_ELEMENT":
                sizes_to_test = [1]
            elif data_type == "EMPTY_ARRAY":
                sizes_to_test = [0]
            elif data_type == "TWO_ELEMENTS":
                sizes_to_test = [2]
            else:
                sizes_to_test = EDGE_CASE_SIZES

            for n in sizes_to_test:
                base_array = generate_array(n, data_type)

                for name, func in algorithms.items():
                    if name == "SelectionSort" and n > 10000:
                        continue

                    avg_ns = benchmark(func, base_array, RUNS_PER_TEST)
                    print_result(name, n, avg_ns)

                    writer.writerow([
                        name,
                        data_type,
                        n,
                        avg_ns,
                        f"{avg_ns / 1_000_000:.6f}"
                    ])

                print()

    print(f"CSV exported successfully: {CSV_FILE}")
    print("Use Excel / Google Sheets to create graphs.")


if __name__ == "__main__":
    main()