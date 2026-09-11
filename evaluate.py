#!/usr/bin/env python
"""
Top-Level Evaluation Runner for BenchmarkSuite.
Evaluates tracking graphs against official ground truth metrics.
Usage:
    python evaluate.py --data-dir /kaggle/input/competitions/biohub-cell-tracking-during-development/train \
                       --volumes 6bba_05db0fb1 6bba_05b6850b
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.evaluation.benchmark_suite import BenchmarkSuite

def main():
    parser = argparse.ArgumentParser(description="Biohub Cell Tracking Evaluation Harness")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to train directory")
    parser.add_argument("--volumes", nargs="+", default=["6bba_05db0fb1", "6bba_05b6850b"], help="Volumes to benchmark")
    parser.add_argument("--max-dist", type=float, default=7.0, help="Distance matching threshold in um")
    args = parser.parse_args()

    suite = BenchmarkSuite(train_dir=Path(args.data_dir), max_matching_distance_um=args.max_dist)
    print(f"Benchmark Suite initialized on {len(args.volumes)} target volumes.")

if __name__ == "__main__":
    main()
