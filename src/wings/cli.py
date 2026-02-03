"""Command-line interface for gaussian-state-optimizer."""

import argparse
import sys
from typing import Optional


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Usage:
        gso optimize --qubits 8 --sigma 0.5
        gso benchmark --qubits 12
        gso crossover
        gso info
        gso campaigns list
    """
    parser = argparse.ArgumentParser(
        prog="gso",
        description="Gaussian State Optimizer - GPU-accelerated quantum state preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gso optimize --qubits 8 --sigma 0.5
  gso optimize --qubits 10 --sigma 0.5 --target-infidelity 1e-10 --max-time 3600
  gso benchmark --qubits 12
  gso crossover
  gso info
  gso campaigns list
  gso campaigns load campaign_q8_s0.50_20240101_120000
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================
    # optimize command
    # ========================================
    opt_parser = subparsers.add_parser(
        "optimize",
        help="Run Gaussian state optimization",
        description="Optimize quantum circuit parameters to prepare a Gaussian state",
    )
    opt_parser.add_argument(
        "-q",
        "--qubits",
        type=int,
        default=8,
        help="Number of qubits (default: 8)",
    )
    opt_parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=0.5,
        help="Gaussian width sigma (default: 0.5)",
    )
    opt_parser.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Box size for position grid (default: auto)",
    )
    opt_parser.add_argument(
        "--target-infidelity",
        type=float,
        default=1e-8,
        help="Target infidelity 1-F (default: 1e-8)",
    )
    opt_parser.add_argument(
        "--max-time",
        type=float,
        default=600,
        help="Maximum optimization time in seconds (default: 600)",
    )
    opt_parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    opt_parser.add_argument(
        "--no-custatevec",
        action="store_true",
        help="Disable cuStateVec (use Aer GPU instead)",
    )
    opt_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable result plotting",
    )
    opt_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving results to files",
    )
    opt_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    opt_parser.add_argument(
        "-Q",
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    # ========================================
    # campaign command
    # ========================================
    camp_parser = subparsers.add_parser(
        "campaign",
        help="Run production optimization campaign",
        description="Run large-scale optimization campaign with checkpointing",
    )
    camp_parser.add_argument(
        "-q",
        "--qubits",
        type=int,
        default=8,
        help="Number of qubits (default: 8)",
    )
    camp_parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=0.5,
        help="Gaussian width sigma (default: 0.5)",
    )
    camp_parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=100,
        help="Total optimization runs (default: 100)",
    )
    camp_parser.add_argument(
        "--target-infidelity",
        type=float,
        default=1e-10,
        help="Target infidelity (default: 1e-10)",
    )
    camp_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Campaign name (default: auto-generated)",
    )
    camp_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )

    # ========================================
    # benchmark command
    # ========================================
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark GPU vs CPU performance",
    )
    bench_parser.add_argument(
        "-q",
        "--qubits",
        type=int,
        default=10,
        help="Number of qubits (default: 10)",
    )
    bench_parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=0.5,
        help="Gaussian width (default: 0.5)",
    )

    # ========================================
    # crossover command
    # ========================================
    cross_parser = subparsers.add_parser(
        "crossover",
        help="Find GPU crossover point",
        description="Determine at what qubit count GPU becomes faster than CPU",
    )
    cross_parser.add_argument(
        "--min-qubits",
        type=int,
        default=6,
        help="Minimum qubits to test (default: 6)",
    )
    cross_parser.add_argument(
        "--max-qubits",
        type=int,
        default=18,
        help="Maximum qubits to test (default: 18)",
    )

    # ========================================
    # info command
    # ========================================
    subparsers.add_parser(
        "info",
        help="Show backend and system information",
    )

    # ========================================
    # campaigns subcommand
    # ========================================
    campaigns_parser = subparsers.add_parser(
        "campaigns",
        help="Manage optimization campaigns",
    )
    campaigns_sub = campaigns_parser.add_subparsers(dest="campaigns_command")

    campaigns_sub.add_parser("list", help="List all campaigns")

    load_parser = campaigns_sub.add_parser("load", help="Load campaign results")
    load_parser.add_argument("name", help="Campaign name or path")

    # Parse arguments
    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 0

    # Execute command
    try:
        if parsed.command == "optimize":
            return _cmd_optimize(parsed)
        elif parsed.command == "campaign":
            return _cmd_campaign(parsed)
        elif parsed.command == "benchmark":
            return _cmd_benchmark(parsed)
        elif parsed.command == "crossover":
            return _cmd_crossover(parsed)
        elif parsed.command == "info":
            return _cmd_info(parsed)
        elif parsed.command == "campaigns":
            return _cmd_campaigns(parsed)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_optimize(args) -> int:
    """Handle optimize command."""
    from .convenience import optimize_gaussian_state

    verbose = not args.quiet if hasattr(args, "quiet") else True

    results, optimizer = optimize_gaussian_state(
        n_qubits=args.qubits,
        sigma=args.sigma,
        box_size=args.box_size,
        target_infidelity=args.target_infidelity,
        max_time=args.max_time,
        use_gpu=not args.no_gpu,
        use_custatevec=not args.no_custatevec,
        plot=not args.no_plot,
        save=not args.no_save,
        verbose=verbose,
    )

    # Print final result for scripting
    print(f"\nFinal fidelity: {results['fidelity']:.15f}")
    print(f"Final infidelity: {results['infidelity']:.3e}")

    return 0 if results["fidelity"] >= (1 - args.target_infidelity) else 1


def _cmd_campaign(args) -> int:
    """Handle campaign command."""
    from .campaign import run_production_campaign

    results = run_production_campaign(
        n_qubits=args.qubits,
        sigma=args.sigma,
        total_runs=args.runs,
        target_infidelity=args.target_infidelity,
        campaign_name=args.name,
        resume=args.resume,
    )

    results.print_summary()

    return (
        0
        if results.best_result and results.best_result.fidelity >= (1 - args.target_infidelity)
        else 1
    )


def _cmd_benchmark(args) -> int:
    """Handle benchmark command."""
    from .benchmarks import benchmark_gpu

    benchmark_gpu(
        n_qubits=args.qubits,
        sigma=args.sigma,
        verbose=True,
    )

    return 0


def _cmd_crossover(args) -> int:
    """Handle crossover command."""
    from .benchmarks import find_gpu_crossover

    qubit_range = list(range(args.min_qubits, args.max_qubits + 1, 2))

    find_gpu_crossover(
        qubit_range=qubit_range,
        verbose=True,
    )

    return 0


def _cmd_info(args) -> int:
    """Handle info command."""
    from . import __version__, print_backend_info

    print(f"Gaussian State Optimizer v{__version__}")
    print()
    print_backend_info()

    return 0


def _cmd_campaigns(args) -> int:
    """Handle campaigns subcommands."""
    if args.campaigns_command == "list":
        from .campaign import list_campaigns

        campaigns = list_campaigns()

        if not campaigns:
            print("No campaigns found.")
            return 0

        print(f"{'Name':<50} {'Runs':<8} {'Best Fidelity':<20}")
        print("-" * 80)

        for c in campaigns:
            runs = c.get("total_runs", "N/A")
            fid = c.get("best_fidelity")
            fid_str = f"{fid:.12f}" if fid else "N/A"
            print(f"{c['name']:<50} {runs:<8} {fid_str:<20}")

        return 0

    elif args.campaigns_command == "load":
        from .campaign import load_campaign_results

        results = load_campaign_results(args.name)
        results.print_summary()

        return 0

    else:
        print("Usage: gso campaigns {list|load}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
