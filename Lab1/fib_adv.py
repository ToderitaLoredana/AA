import time
import matplotlib.pyplot as plt


def fibonacci_fib11(n):
    """Fast Fibonacci using recursive formula - O(log n) time"""
    n0 = n
    n = abs(n)
    F = {}
    qinx = []  # queue of indexes
    qinx.append(n)
    F[n] = -1  # -1 to mark such values
    
    while qinx:
        k = qinx.pop() >> 1
        if (k - 1) not in F:
            F[k - 1] = -1
            qinx.append(k - 1)
        if k not in F:
            F[k] = -1
            qinx.append(k)
        if (k + 1) not in F:
            F[k + 1] = -1
            qinx.append(k + 1)
    
    # Set base values
    F[0], F[1], F[2] = 0, 1, 1
    
    # Fill the indexes that need values
    keys_sorted = sorted(F.keys())
    for k in keys_sorted[3:]:
        k2 = k >> 1
        f1, f2, f3 = F[k2 - 1], F[k2], F[k2 + 1]
        if k % 2 == 0:
            F[k] = f3 * f3 - f1 * f1
        else:
            F[k] = f3 * f3 + f2 * f2
    
    r = F[n]
    if n0 < 0:
        return negafib(n, r)
    return r


def negafib(n, r):
    """Helper function for negative Fibonacci numbers"""
    if n % 2 == 0:
        return -r
    return r


def fibonacci_iterative(n):
    """Iterative Fibonacci - O(n) time, O(1) space"""
    if n <= 1:
        return 1
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fibonacci_fast_doubling(n):
    """
    Fast Doubling Fibonacci - O(log n) time, O(log n) space
    Uses the mathematical identities:
    F(2k) = F(k) * (2*F(k+1) - F(k))
    F(2k+1) = F(k+1)^2 + F(k)^2
    
    This is an elegant and efficient algorithm based on binary representation of n.
    """
    if n == 0:
        return 0
    if n <= 2:
        return 1
    
    def fib_doubling(k):
        """Returns (F(k), F(k+1)) as a tuple"""
        if k == 0:
            return (0, 1)
        
        # Recursive call for k//2
        f_m, f_m1 = fib_doubling(k // 2)
        
        # F(2m) = F(m) * (2*F(m+1) - F(m))
        c = f_m * (2 * f_m1 - f_m)
        # F(2m+1) = F(m+1)^2 + F(m)^2
        d = f_m * f_m + f_m1 * f_m1
        
        if k % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)
    
    return fib_doubling(n)[0]


def performance():
    test_numbers = [5, 10, 15, 20, 25, 30, 50, 100, 500, 1000, 5000, 10000]
    repeats = 3
    
    fib11_times = []
    fast_doubling_times = []
    iterative_times = []
    
    header = "n  " + "  ".join(f"run{i + 1}(ms)" for i in range(repeats)) + "  avg(ms)"
    print("\n=== FIB11 (FAST RECURSIVE FORMULA) O(log n) ===")
    print(header)
    print("-" * len(header))
    
    for number in test_numbers:
        fib11_execs = []
        
        for _ in range(repeats):
            start = time.perf_counter()
            fibonacci_fib11(number)
            end = time.perf_counter()
            fib11_execs.append((end - start) * 1000)
        
        avg_time = sum(fib11_execs) / repeats
        row = f"{number:<5} " + "  ".join(f"{t:>8.3f}" for t in fib11_execs) + f"  {avg_time:>8.3f}"
        print(row)
        fib11_times.append(avg_time)
    
    print("\n=== FAST DOUBLING O(log n) ===")
    print(header)
    print("-" * len(header))
    
    for number in test_numbers:
        fast_doubling_execs = []
        
        for _ in range(repeats):
            start = time.perf_counter()
            fibonacci_fast_doubling(number)
            end = time.perf_counter()
            fast_doubling_execs.append((end - start) * 1000)
        
        avg_time = sum(fast_doubling_execs) / repeats
        row = f"{number:<5} " + "  ".join(f"{t:>8.3f}" for t in fast_doubling_execs) + f"  {avg_time:>8.3f}"
        print(row)
        fast_doubling_times.append(avg_time)
    
    print("\n=== ITERATIVE FIBONACCI O(n) ===")
    print(header)
    print("-" * len(header))
    
    for number in test_numbers:
        iterative_execs = []
        
        for _ in range(repeats):
            start = time.perf_counter()
            fibonacci_iterative(number)
            end = time.perf_counter()
            iterative_execs.append((end - start) * 1000)
        
        avg_time = sum(iterative_execs) / repeats
        row = f"{number:<5} " + "  ".join(f"{t:>8.3f}" for t in iterative_execs) + f"  {avg_time:>8.3f}"
        print(row)
        iterative_times.append(avg_time)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].plot(
        test_numbers,
        fib11_times,
        marker="o",
        linestyle="-",
        color="g",
        label="FIB11 O(log n)",
        linewidth=2,
        markersize=8,
    )
    axes[0].plot(
        test_numbers,
        fast_doubling_times,
        marker="^",
        linestyle="-",
        color="r",
        label="Fast Doubling O(log n)",
        linewidth=2,
        markersize=8,
    )
    axes[0].plot(
        test_numbers,
        iterative_times,
        marker="s",
        linestyle="-",
        color="b",
        label="Iterative O(n)",
        linewidth=2,
        markersize=8,
    )
    axes[0].set_title("Fibonacci Algorithms: Linear Scale")
    axes[0].set_xlabel("Fibonacci Number Index (n)")
    axes[0].set_ylabel("Execution Time (ms)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Log scale
    axes[1].plot(
        test_numbers,
        fib11_times,
        marker="o",
        linestyle="-",
        color="g",
        label="FIB11 O(log n)",
        linewidth=2,
        markersize=8,
    )
    axes[1].plot(
        test_numbers,
        fast_doubling_times,
        marker="^",
        linestyle="-",
        color="r",
        label="Fast Doubling O(log n)",
        linewidth=2,
        markersize=8,
    )
    axes[1].plot(
        test_numbers,
        iterative_times,
        marker="s",
        linestyle="-",
        color="b",
        label="Iterative O(n)",
        linewidth=2,
        markersize=8,
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Fibonacci Algorithms: Log Scale")
    axes[1].set_xlabel("Fibonacci Number Index (n)")
    axes[1].set_ylabel("Execution Time (ms) - Log Scale")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== NOTES ===")
    print("FIB11: O(log n) time - uses recursive formula with matrix-like computation")
    print("Fast Doubling: O(log n) time - elegant algorithm using F(2k) identities")
    print("Iterative: O(n) time - simple loop approach")
    print("\nBoth logarithmic algorithms are significantly faster for very large Fibonacci numbers!")
    print("Fast Doubling is cleaner and uses the identities:")
    print("  F(2k) = F(k) * (2*F(k+1) - F(k))")
    print("  F(2k+1) = F(k+1)^2 + F(k)^2")


if __name__ == "__main__":
    performance()