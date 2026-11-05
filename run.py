"""Run from the project directory: python run.py --suite pilot."""
from pathlib import Path
import argparse
import json
import time
from dataclasses import asdict
from gridshift.experiment import suite_configs, execute, fingerprint, random_baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("pilot", "reconstruction", "core", "random"), required=True)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    out = Path(args.output) / args.suite
    out.mkdir(parents=True, exist_ok=True)
    if args.suite == "random":
        random_baselines(out)
        return
    configs = suite_configs(args.suite)
    manifest = {"suite": args.suite, "engine_sha256": fingerprint(),
                "runs": len(configs), "configurations": [asdict(c) for c in configs]}
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text())["engine_sha256"] != manifest["engine_sha256"]:
        raise RuntimeError("source changed since previous manifest; preserve old output and select another directory")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    for index, config in enumerate(configs, 1):
        record = execute(config, out)
        if args.suite == "pilot" or index % 25 == 0 or index == len(configs):
            print(json.dumps({"completed": index, "total": len(configs),
                              "elapsed_seconds": round(time.perf_counter()-started, 3),
                              "last_run": record["run_id"], "suite": args.suite}), flush=True)


if __name__ == "__main__":
    main()
