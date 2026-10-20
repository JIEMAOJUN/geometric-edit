from copy import deepcopy

DATA = {
    "natural_root": "datasets/Natural-Image-Subset",
    "sensitive_root": "datasets/Sensitive-Images",
    "num_edit_samples": None,
    "num_neighborhood_samples": None,
    "edit_ratio_after_anchor": 0.8,
    "num_anchor_samples": 20,
    "num_retain_train_samples": 50,
    "num_retain_eval_samples": 100,
}

ABLATION = {
    "exp_name": "ablation",
    "output_dir": "results",
    "require_group_count": 16,
    "one_step_alpha": 0.2,
    "random_repeats": 5,
}


def add_runtime_arguments(parser):
    parser.add_argument("--exp-name", default="gradgeoedit_vit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--ramp-steps", type=int, required=True)
    parser.add_argument("--fisher-batch-size", type=int, default=8)
    parser.add_argument("--lambda-edit", type=float, required=True)
    parser.add_argument("--lambda-edit-base", type=float, required=True)
    parser.add_argument("--lambda-locality", type=float, required=True)
    parser.add_argument("--lambda-reg", type=float, required=True)
    parser.add_argument("--eval-interval", type=int, default=5)


def config_from_args(args):
    if not 0 < args.budget <= 1:
        raise ValueError("--budget must be a fraction in (0, 1]")
    if args.lr <= 0 or args.steps <= 0 or args.eval_interval <= 0:
        raise ValueError("Learning rate, steps and evaluation interval must be positive")
    if not 0 <= args.warmup_steps <= args.steps or args.ramp_steps <= 0:
        raise ValueError("Invalid warmup or ramp steps")
    if any(value < 0 for value in (
        args.lambda_edit, args.lambda_edit_base, args.lambda_locality, args.lambda_reg
    )):
        raise ValueError("Loss weights must be nonnegative")
    return {
        "exp_name": args.exp_name,
        "seed": args.seed,
        "data": deepcopy(DATA),
        "ablation": deepcopy(ABLATION),
        "edit": {
            "budget": args.budget,
            "lr": args.lr,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "ramp_steps": args.ramp_steps,
            "fisher_batch_size": args.fisher_batch_size,
            "lambda_edit": args.lambda_edit,
            "lambda_edit_base": args.lambda_edit_base,
            "lambda_locality": args.lambda_locality,
            "lambda_reg": args.lambda_reg,
            "eval_interval": args.eval_interval,
        },
    }
