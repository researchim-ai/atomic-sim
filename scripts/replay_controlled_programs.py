"""Replay every controlled campaign history with bounded worker concurrency."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.controlled_plant_release import replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--workers", type=int, choices=[1, 2, 3], default=2)
    a = p.parse_args()
    folders = sorted(path.parent for path in a.root.glob("*/result.json"))
    if len(folders) != 10:
        raise ValueError("Expected the complete ten-program campaign")

    def run(folder):
        result = replay(folder, folder / "replay.json")
        print(folder.name, result["replay_equal"], flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run, folders))


if __name__ == "__main__":
    main()
