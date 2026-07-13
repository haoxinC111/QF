from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare_quant.public_research import MiniRacer, _SinaDecoder  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniRacer single-decoder stress test")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tasks", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1 or args.tasks < 1 or args.repeats < 1:
        raise ValueError("workers/tasks/repeats must be positive")
    if MiniRacer is None:
        raise RuntimeError("MiniRacer optional dependency is not installed")

    script = "function d(value) { return JSON.parse(value); }"
    decoded_count = 0
    for repeat in range(args.repeats):
        payloads = [
            json.dumps({"repeat": repeat, "value": value})
            for value in range(args.tasks)
        ]
        with _SinaDecoder(
            queue_size=max(4, args.workers * 2),
            decode_script=script,
        ) as decoder:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers
            ) as executor:
                decoded = list(executor.map(decoder.decode, payloads))
        expected = [
            {"repeat": repeat, "value": value} for value in range(args.tasks)
        ]
        if decoded != expected:
            raise AssertionError(f"decoded payload mismatch in repeat {repeat}")
        decoded_count += len(decoded)

    providers = metadata.packages_distributions().get("py_mini_racer", [])
    print(
        json.dumps(
            {
                "workers": args.workers,
                "tasks_per_repeat": args.tasks,
                "repeats": args.repeats,
                "decoded": decoded_count,
                "providers": sorted(providers),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
