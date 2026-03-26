from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.proposal_step2_toy_sanity import summarize_cached_toy_results


def main() -> None:
    summary = summarize_cached_toy_results(PROJECT_ROOT)
    print("Saved Step 2 toy summary to outputs/toy/summary.json")
    print(summary)


if __name__ == "__main__":
    main()
