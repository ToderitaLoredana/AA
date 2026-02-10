import time
import matplotlib.pyplot as plt

CALL_COUNT = 0
def fibonacci_naive(n):
    global CALL_COUNT
    CALL_COUNT += 1
    if n <= 1:
        return 1
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_memo(n, memo):
    if n in memo:
        return memo[n]
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


def performance():
    test_numbers = [5, 10, 15, 20, 25, 30]
    repeats = 3
    naive_times = []
    naive_calls = []
    memo_times = []

    header = "n  " + "  ".join(f"run{i + 1}(ms)" for i in range(repeats)) + "  avg(ms)"
    print(header)
    print("-" * len(header))

    for number in test_numbers:
        naive_execs = []
        memo_execs = []
        calls_for_n = 0

        for _ in range(repeats):
            global CALL_COUNT
            CALL_COUNT = 0
            start = time.perf_counter()
            fibonacci_naive(number)
            end = time.perf_counter()
            naive_execs.append((end - start) * 1000)
            calls_for_n = CALL_COUNT

            memo = {0: 1, 1: 1}
            start = time.perf_counter()
            fibonacci_memo(number, memo)
            end = time.perf_counter()
            memo_execs.append((end - start) * 1000)

        avg_time = sum(naive_execs) / repeats
        row = f"{number:<2} " + "  ".join(f"{t:>8.3f}" for t in naive_execs) + f"  {avg_time:>8.3f}"
        print(row)

        naive_times.append(avg_time)
        naive_calls.append(calls_for_n)
        memo_times.append(sum(memo_execs) / repeats)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        test_numbers,
        naive_times,
        marker="o",
        linestyle="-",
        color="r",
        label="Naïve Recursion",
    )
    axes[0].plot(
        test_numbers,
        memo_times,
        marker="o",
        linestyle="-",
        color="g",
        label="Memoized Recursion",
    )
    axes[0].set_title("Execution Time (Average of 3 runs)")
    axes[0].set_xlabel("Fibonacci Number Index")
    axes[0].set_ylabel("Execution Time (ms)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        test_numbers,
        naive_calls,
        marker="o",
        linestyle="-",
        color="b",
        label="Naïve Recursion Calls",
    )
    axes[1].set_title("Naïve Recursion Call Count")
    axes[1].set_xlabel("Fibonacci Number Index")
    axes[1].set_ylabel("Function Calls")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    performance()