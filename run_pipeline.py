#!/usr/bin/env python3
"""Run the whole pipeline end to end, in phase order.

    python run_pipeline.py             # everything, using cached pulls
    python run_pipeline.py --refresh   # re-pull the raw data first
    python run_pipeline.py --from train
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

PHASES = ["ingest", "label", "eda", "features", "train", "evaluate"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull raw data instead of using the cache")
    ap.add_argument("--from", dest="start", choices=PHASES, default="ingest")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("pipeline")

    import eda
    import evaluate
    import features
    import ingest
    import label
    import train

    steps = {
        "ingest": lambda: ingest.load_all(refresh=args.refresh),
        "label": lambda: label.main(refresh=False),
        "eda": eda.run,
        "features": lambda: features.main(refresh=False),
        "train": train.main,
        "evaluate": evaluate.run,
    }

    for name in PHASES[PHASES.index(args.start):]:
        log.info("=" * 62)
        log.info("phase: %s", name)
        log.info("=" * 62)
        start = time.time()
        steps[name]()
        log.info("phase %s finished in %.1fs", name, time.time() - start)

    log.info("done. reports are in reports/, models in models/")
    log.info("weekly inference: python src/predict.py --season 2024 --week 10")


if __name__ == "__main__":
    main()
