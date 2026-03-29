#!/usr/bin/env python3
"""
Wrapper script called by wandb sweep agent for each trial.

Bridges wandb sweep args to Hydra config overrides by:
1. Initializing wandb to receive sweep parameters
2. Reading delta/scale from wandb.config
3. Launching main.py via subprocess with Hydra overrides and
   WANDB_RUN_ID/WANDB_RESUME env vars so trainer.py resumes the sweep run
"""

import os
import subprocess
import sys

import wandb


def main():
    # Init wandb — agent sets sweep env vars so this joins the correct sweep run
    wandb.init()

    delta = wandb.config["delta"]
    scale = wandb.config["scale"]
    seed = wandb.config["seed"]
    run_id = wandb.run.id

    # Release the run so the subprocess can resume it
    wandb.finish()

    # Build Hydra overrides
    overrides = [
        f"tokenizer.delta={delta}",
        f"tokenizer.scale={scale}",
        f"training.seed={seed}",
        "tokenizer.enable_ddcl=true",
        "tokenizer.enable_fsq=false",
        f"wandb.name=sweep-delta{delta}-scale{scale}-seed{seed}",
        "wandb.tags=[world models, iris, sweep]",
    ]

    env = os.environ.copy()
    env["WANDB_RUN_ID"] = run_id
    env["WANDB_RESUME"] = "must"

    cmd = [sys.executable, "src/main.py"] + overrides
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
