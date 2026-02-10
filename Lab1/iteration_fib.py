import time
import matplotlib.pyplot as plt

def fibonacci(n):
    if n < 2:
        return n

    p, q, r = 0, 1, 1
    for _ in range(2, n + 1):
        p = q
        q = r
        r = p + q

    return r

def measure_time(func, n):
    start = time.time()
    func(n)
    end = time.time()
    return end - start

def print_results_table(input_values, results):
    header = "n:   " + "  ".join(f"{n:<8}" for n in input_values)
    print(header)

    for run_index, run_times in enumerate(results):
        row = f"{run_index}:  " + "  ".join(f"{t:<8.6f}" for t in run_times)
        print(row)

def plot_results(input_values, results):
    times = results[0]  # first run
    plt.plot(input_values, times, marker='o')
    plt.xlabel("n-th Fibonacci Term")
    plt.ylabel("Time (seconds)")
    plt.title("Iterative Fibonacci Execution Time")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    input = [
        501, 631, 794, 1000, 1259, 1585, 1995,
        2512, 3162, 3981, 5012, 6310, 7943,
        10000, 12589, 15849
    ]

    NUM_RUNS = 3
    results = []

    for run in range(NUM_RUNS):
        run_times = []
        for n in input:
            t = measure_time(fibonacci, n)
            run_times.append(t)
        results.append(run_times)

    print("\nEmpirical Results Table:\n")
    print_results_table(input, results)

    plot_results(input, results)