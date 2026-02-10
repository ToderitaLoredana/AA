import time
import matplotlib.pyplot as plt
import math

def fibonacci_binet(n):
    phi = (1 + math.sqrt(5)) / 2
    phi1 = (1 - math.sqrt(5)) / 2
    return round((phi**n - phi1**n) / math.sqrt(5))
def performance():
    test_numbers = [5, 10, 15, 20, 25, 30, 35, 40, 50, 100, 500, 1000]
    repeats = 3
    binet_times = []
    header = "n  " + "  ".join(f"run{i + 1}(ms)" for i in range(repeats)) + "  avg(ms)"
    print("\n=== BINET FORMULA METHOD ===")
    print(header)
    print("-" * len(header))
    
    for number in test_numbers:
        binet_execs = []
        
        for _ in range(repeats):
            start = time.perf_counter()
            fibonacci_binet(number)
            end = time.perf_counter()
            binet_execs.append((end - start) * 1000)
        
        avg_time = sum(binet_execs) / repeats
        row = f"{number:<5} " + "  ".join(f"{t:>8.3f}" for t in binet_execs) + f"  {avg_time:>8.3f}"
        print(row)
        binet_times.append(avg_time)
    
    # Create plots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        test_numbers,
        binet_times,
        marker="o",
        linestyle="-",
        color="b",
        label="Binet Formula",
        linewidth=2,
        markersize=8,
    )
    ax.set_title("Binet Formula Fibonacci Method")
    ax.set_xlabel("n-th Fibonacci Term")
    ax.set_ylabel("Execution Time (ms)")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    performance()