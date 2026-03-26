import os
import sys
import numpy as np

#
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath(os.path.join(PROJECT_ROOT, "..")))

from toy_sanity_check.src.toy_data import generate_hmm_sequence, save_toy_data


def main():
    X, Z, params = generate_hmm_sequence(T=300, seed=123)

    print("=== Toy HMM Demo ===")
    print("X shape:", X.shape)   # expected: (300, 2)
    print("Z shape:", Z.shape)   # expected: (300,)
    print("First 30 Z:", Z[:30])
    print("Unique states:", np.unique(Z))
    print("Transition matrix A:\n", params["A"])
    print("Means:\n", params["means"])

    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    save_toy_data(os.path.join(PROJECT_ROOT, "data", "sequence.npz"), X, Z, params)
    print("Toy data saved to toy_sanity_check/data/sequence.npz")


if __name__ == "__main__":
    main()
