#!/usr/bin/env python3
"""
Orchestrator for wandb sweep over tokenizer.delta and tokenizer.scale.

Usage:
    # Create sweep and run 1 agent (all 20 grid combos):
    python run_sweep.py

    # Create sweep only (print ID for multi-agent use):
    python run_sweep.py --create-only

    # Join an existing sweep:
    python run_sweep.py --sweep-id <entity/project/sweep_id>

    # Limit runs per agent:
    python run_sweep.py --count 5
"""

import argparse
import sys

import wandb
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Run wandb sweep for tokenizer delta/scale grid search"
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create the sweep and print the ID, don't run an agent",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="Use existing sweep ID instead of creating a new one",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs for this agent (default: run all sweep jobs)",
    )
    args = parser.parse_args()

    if args.sweep_id:
        sweep_path = args.sweep_id
        print(f"Using existing sweep: {sweep_path}")
    else:
        with open("sweep_config.yaml", "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=sweep_config.get("project", "ddcl-applications"))
        entity = wandb.Api().default_entity
        sweep_path = f"{entity}/{sweep_config.get('project', 'ddcl-applications')}/{sweep_id}"

        print(f"Created sweep: {sweep_id}")
        print(f"Full path: {sweep_path}")

    print(f"\nTo run agents manually:\n  wandb agent {sweep_path}")

    if args.create_only:
        print("\n--create-only set. Exiting.")
        sys.exit(0)

    print("\nStarting sweep agent...")
    wandb.agent(sweep_path, count=args.count)


if __name__ == "__main__":
    main()
