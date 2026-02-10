"""
Lab 1 - Algorithm 1: Fibonacci Number Computation using Recursion
Course: Algorithm Analysis (AA)

This program computes the n-th Fibonacci number using the naive recursive method.
Time Complexity: O(2^n) — exponential
Space Complexity: O(n) — due to recursion stack depth
"""

import time
import sys


def fibonacci_recursive(n):
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def main():
    # Increase recursion limit for larger inputs (use with caution)
    sys.setrecursionlimit(10000)

    # Read input
    try:
        n = int(input("Enter the index n of the Fibonacci number to compute: "))
    except ValueError:
        print("Error: Please enter a valid integer.")
        return

    if n < 0:
        print("Error: n must be a non-negative integer.")
        return

    # Warn the user for large inputs (recursion is exponential)
    if n > 35:
        print(f"Warning: n={n} may take a very long time with naive recursion (O(2^n)).")

    # Compute and measure time
    start_time = time.time()
    result = fibonacci_recursive(n)
    end_time = time.time()

    elapsed_time = end_time - start_time

    # Output
    print(f"\nFibonacci({n}) = {result}")
    print(f"Time taken: {elapsed_time:.6f} seconds")


if __name__ == "__main__":
    main()
