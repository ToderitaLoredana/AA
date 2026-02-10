import time
import matplotlib.pyplot as plt
import numpy as np

def matrix_multiply_naive(A, B):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_multiply_numpy(A, B):
    return np.dot(A, B)

def add_matrix(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]
def sub_matrix(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def generate_random_matrix(n):
    return [[np.random.randint(1, 10) for _ in range(n)] for _ in range(n)]
def performance():
    test_sizes = [64, 128, 256, 512]
    repeats = 3
    naive_times = []
    numpy_times = []
    header = "size  " + "  ".join(f"run{i + 1}(ms)" for i in range(repeats)) + "  avg(ms)"
    print(header)
    print("-" * len(header))
    
    for size in test_sizes:
        naive_execs = []
        numpy_execs = []
        for _ in range(repeats):
            # Generate random matrices
            A = generate_random_matrix(size)
            B = generate_random_matrix(size)
            A_np = np.array(A)
            B_np = np.array(B)
            
            # Test naive method
            start = time.perf_counter()
            matrix_multiply_naive(A, B)
            end = time.perf_counter()
            naive_execs.append((end - start) * 1000)
            
            # Test NumPy method
            start = time.perf_counter()
            matrix_multiply_numpy(A_np, B_np)
            end = time.perf_counter()
            numpy_execs.append((end - start) * 1000)
        
        avg_time = sum(naive_execs) / repeats
        row = f"{size:<4} " + "  ".join(f"{t:>8.3f}" for t in naive_execs) + f"  {avg_time:>8.3f}"
        print(row)
        
        naive_times.append(avg_time)
        numpy_times.append(sum(numpy_execs) / repeats)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(
        test_sizes,
        naive_times,
        marker="o",
        linestyle="-",
        color="r",
        label="Naive O(n³)",
    )
    axes[0].plot(
        test_sizes,
        numpy_times,
        marker="s",
        linestyle="-",
        color="g",
        label="NumPy Optimized",
    )
    axes[0].set_title("Matrix Multiplication Performance (Average of 3 runs)")
    axes[0].set_xlabel("Matrix Size (n x n)")
    axes[0].set_ylabel("Execution Time (ms)")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(
        test_sizes,
        naive_times,
        marker="o",
        linestyle="-",
        color="r",
        label="Naive O(n³)",
    )
    axes[1].plot(
        test_sizes,
        numpy_times,
        marker="s",
        linestyle="-",
        color="g",
        label="NumPy Optimized",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Matrix Multiplication Performance (Log Scale)")
    axes[1].set_xlabel("Matrix Size (n x n)")
    axes[1].set_ylabel("Execution Time (ms) - Log Scale")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    performance()