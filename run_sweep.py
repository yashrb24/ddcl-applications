#!/usr/bin/env python3
"""
Script to initialize and run wandb sweep for DDCL hyperparameter tuning.

This script supports both single-agent and multi-agent parallel sweep execution.

Usage Examples:

1. Create sweep and run single agent:
    python run_sweep.py

2. Create sweep only (for multi-agent setup):
    python run_sweep.py --create-only
    # Then in separate terminals/tmux/screen:
    wandb agent <sweep_id>  # Agent 1
    wandb agent <sweep_id>  # Agent 2
    wandb agent <sweep_id>  # Agent 3

3. Run specific number of jobs per agent:
    python run_sweep.py --count 3

4. Use existing sweep:
    python run_sweep.py --sweep-id <sweep_id>

Multi-Agent Parallel Execution:
    For fastest results, run multiple agents in parallel. Each agent will
    pull different hyperparameter combinations from the sweep queue.
    
    With tmux:
        tmux new-session -d -s sweep1 'wandb agent <sweep_id>'
        tmux new-session -d -s sweep2 'wandb agent <sweep_id>'
        tmux new-session -d -s sweep3 'wandb agent <sweep_id>'
    
    With GNU parallel:
        parallel -j 3 wandb agent ::: <sweep_id> <sweep_id> <sweep_id>
"""

import wandb
import yaml


def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Initialize and run wandb sweep for DDCL hyperparameter tuning"
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create the sweep and print the ID, don't run agents"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs for this agent (default: run all sweep jobs)"
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="Use existing sweep ID instead of creating a new one"
    )
    
    args = parser.parse_args()
    
    # Create or use existing sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep: {sweep_id}")
    else:
        # Load sweep configuration
        with open("sweep_config.yaml", "r") as f:
            sweep_config = yaml.safe_load(f)
        
        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project="ddcl-vae")
        print(f"Created new sweep with ID: {sweep_id}")
    
    print(f"\nTo run agents manually (supports multiple parallel agents):")
    print(f"  wandb agent {sweep_id}")
    print(f"\nFor parallel execution, run the above command in multiple terminals/tmux/screen sessions")
    
    if args.create_only:
        print("\n--create-only flag set. Exiting without running agent.")
        sys.exit(0)
    
    # Run the sweep agent
    print(f"\nStarting sweep agent...")
    if args.count:
        print(f"  Running {args.count} sweep jobs")
        wandb.agent(sweep_id, count=args.count)
    else:
        print(f"  Running all sweep jobs (9 total for current grid)")
        wandb.agent(sweep_id)


if __name__ == "__main__":
    main()
