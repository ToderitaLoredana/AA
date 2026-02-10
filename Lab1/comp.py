import matplotlib.pyplot as plt
import numpy as np
import time

def binets_formula(n):
    """Calculate Fibonacci using Binet's formula"""
    phi = (1 + np.sqrt(5)) / 2
    psi = (1 - np.sqrt(5)) / 2
    return round((phi**n - psi**n) / np.sqrt(5))

def dynamic_programming(n):
    """Calculate Fibonacci using dynamic programming"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def memoization_list(n, memo=None):
    """Calculate Fibonacci using memoization with list"""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = memoization_list(n - 1, memo) + memoization_list(n - 2, memo)
    return memo[n]

def matrix_exponentiation(n):
    """Calculate Fibonacci using matrix exponentiation"""
    def matrix_mult(a, b):
        return [[a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
                [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]]]
    
    def matrix_pow(m, p):
        if p == 1:
            return m
        if p % 2 == 0:
            half = matrix_pow(m, p // 2)
            return matrix_mult(half, half)
        return matrix_mult(m, matrix_pow(m, p - 1))
    
    if n == 0:
        return 0
    result = matrix_pow([[1, 1], [1, 0]], n)
    return result[0][1]

def memoization_dict(n, memo=None):
    """Calculate Fibonacci using memoization with dict"""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = memoization_dict(n - 1, memo) + memoization_dict(n - 2, memo)
    return memo[n]

# Measure performance
algorithms = {
    "Binet's Formula": binets_formula,
    "Dynamic Programming": dynamic_programming,
    "Memoization List": lambda n: memoization_list(n),
    "Matrix Exponentiation": matrix_exponentiation,
    "Memoization Dict": lambda n: memoization_dict(n)
}

n_values = range(0, 36)
results = {name: [] for name in algorithms}

for n in n_values:
    for name, func in algorithms.items():
        start = time.perf_counter()
        func(n)
        elapsed = time.perf_counter() - start
        results[name].append(elapsed)

# Create plot
plt.figure(figsize=(12, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for (name, times), color in zip(results.items(), colors):
    plt.plot(n_values, times, marker='o', label=name, linewidth=2, color=color, markersize=4)

plt.xlabel('Fibonacci Number (n)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('Performance Comparison of Fibonacci Algorithms', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('fibonacci_comparison.png', dpi=300, bbox_inches='tight')
plt.show()