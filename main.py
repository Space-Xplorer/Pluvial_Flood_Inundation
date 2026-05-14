from run_batch import parse_args, run_batch


if __name__ == "__main__":
    args = parse_args()
    run_batch(
        num_runs=args.num_runs,
        start_suffix=args.start_suffix,
        register_project=not args.no_project_register,
        template_suffix=f"{args.template_suffix:02d}",
    )
