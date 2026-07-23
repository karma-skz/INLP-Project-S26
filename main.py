import argparse
import sys

def run_experiment(args):
    print(f"Running experiment for task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Scale/Parameters: {args.scale}")
    if args.task == "factual":
        from src.experiments import evaluator
        print("Delegating to factual parity benchmark via Evaluator...")
        evaluator.run_parity_eval(args.model, use_negated_probes=args.use_negated_probes, max_samples=args.max_samples)
    elif args.task == "reasoning":
        from src.experiments import evaluator
        print("Delegating to reasoning logic benchmark via Evaluator...")
        # Since reasoning uses the same structure, we route it to the parity evaluator (which handles text triplets)
        evaluator.run_parity_eval(args.model, use_negated_probes=args.use_negated_probes, max_samples=args.max_samples)
    elif args.task == "intervention":
        from src.experiments import evaluator
        print("Delegating to causal interventions via Evaluator...")
        evaluator.run_intervention(args.model, intervention_type="patching", max_samples=args.max_samples)

def generate_data(args):
    print(f"Generating data for task: {args.task}")
    if args.task == "factual":
        from src.dataset import dataset_builder
        print("Generating factual parity triplets...")
        # Assume facts are loaded and generated
    elif args.task == "reasoning":
        from src.dataset import dataset_builder
        print("Generating negated reasoning dataset...")
    elif args.task == "download":
        from src.dataset import dataset_builder
        dataset_builder.DatasetDownloader.download_all()
    else:
        print(f"Unknown task: {args.task}")

def main():
    parser = argparse.ArgumentParser(description="Double Negation Elimination Project CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate-data command
    parser_gen = subparsers.add_parser("generate-data", help="Generate datasets")
    parser_gen.add_argument("--task", choices=["factual", "reasoning", "download"], required=True, help="Task type")
    
    # run-experiment command
    parser_run = subparsers.add_parser("run-experiment", help="Run experiments")
    parser_run.add_argument("--task", choices=["factual", "reasoning", "intervention"], required=True, help="Task type")
    parser_run.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model to use")
    parser_run.add_argument("--scale", type=str, default="8B", help="Model scale (e.g. 8B, 70B)")
    parser_run.add_argument("--use_negated_probes", action="store_true", help="Use Negated LAMA probes")
    parser_run.add_argument("--max_samples", type=int, default=100, help="Max samples")

    args = parser.parse_args()

    if args.command == "generate-data":
        generate_data(args)
    elif args.command == "run-experiment":
        run_experiment(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
