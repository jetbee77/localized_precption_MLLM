from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config


def main():
    parser = argparse.ArgumentParser(description="Run multiple MLLM prompt groups.")
    parser.add_argument("--config", default="../configs/default.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    groups = cfg.get("mllm_groups", {})
    if not groups:
        raise SystemExit("No mllm_groups found in config.")

    base_cmd = (
        "python /Volumes/t7/python_file/LLM_SVIs_exp/svi_pipeline/scripts/mllm_perception.py "
        f"--config {args.config} "
    )
    if args.input:
        base_cmd += f"--input {args.input} "
    if args.model:
        base_cmd += f"--model {args.model} "
    if args.sleep > 0:
        base_cmd += f"--sleep {args.sleep} "

    for name, info in groups.items():
        output_csv = info.get("output_csv")
        if not output_csv:
            print(f"Skip group {name}: missing output_csv")
            continue
        cmd = (
            base_cmd
            + f"--output {output_csv} "
            + f"--prompt-version {name}"
        )
        print(f"\nRunning group: {name}")
        print(cmd)
        ret = Path.cwd().joinpath(".tmp_cmd.sh")
        ret.write_text(cmd)
        exit_code = __import__("os").system(cmd)
        if exit_code != 0:
            print(f"Group {name} failed with exit code {exit_code}")
            break


if __name__ == "__main__":
    main()
