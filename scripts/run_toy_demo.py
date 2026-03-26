import os
import sys
import numpy as np

# 让 Python 能找到 src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from toy_data import generate_hmm_sequence, save_toy_data


def main():
    X, Z, params = generate_hmm_sequence(T=300, seed=123)

    print("=== Toy HMM Demo ===")
    print("X shape:", X.shape)   # expected: (300, 2)
    print("Z shape:", Z.shape)   # expected: (300,)
    print("First 30 Z:", Z[:30])
    print("Unique states:", np.unique(Z))
    print("Transition matrix A:\n", params["A"])
    print("Means:\n", params["means"])

    os.makedirs("data/toy", exist_ok=True)
    save_toy_data("data/toy/toy_sequence.npz", X, Z, params)
    print("Toy data saved to data/toy/toy_sequence.npz")


if __name__ == "__main__":
    main()