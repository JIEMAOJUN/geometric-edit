import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from core import GradGeoEdit
from datamanager import DatasetManager
from logger import Logger
from run import load_vit_model
from settings import add_runtime_arguments, config_from_args


VARIANTS = {
    "A": {"name": "random_one_step", "importance": "random", "optimizer": "one_step"},
    "B": {"name": "pure_grad_one_step", "importance": "grad", "optimizer": "one_step"},
    "C": {"name": "fisher_one_step", "importance": "fisher", "optimizer": "one_step"},
    "D": {"name": "gis_one_step", "importance": "gis", "optimizer": "one_step"},
    "E": {"name": "gis_static", "importance": "gis", "optimizer": "static"},
    "F": {"name": "gis_two_stage", "importance": "gis", "optimizer": "two_stage"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def build_masks(data, config, device, logger, variant_keys):
    model = None
    editor = None
    fisher = None
    coordinate_multiplier = None
    edit_grad = None
    try:
        model = load_vit_model(device)
        editor = GradGeoEdit(model, device)

        logger.log("Computing edit gradient on S_e")
        edit_grad = editor.compute_edit_grad(data["edit_samples"], data["true_label"])

        importance_modes = {
            VARIANTS[key]["importance"]
            for key in variant_keys
            if VARIANTS[key]["importance"] != "random"
        }
        if "fisher" in importance_modes or "gis" in importance_modes:
            logger.log("Computing Fisher on S_a")
            fisher = editor.compute_fisher(
                data["anchor_samples"],
                batch_size=config["edit"].get("fisher_batch_size", 8),
            )
        if "gis" in importance_modes:
            logger.log("Computing random coordinate multiplier")
            coordinate_multiplier = editor.compute_coordinate_multiplier(fisher)

        masks = {}
        stats = {}
        masks_by_mode = {}
        stats_by_mode = {}
        first_key_by_mode = {}
        for key in variant_keys:
            spec = VARIANTS[key]
            mode = spec["importance"]
            if mode == "random":
                continue
            if mode in masks_by_mode:
                masks[key] = masks_by_mode[mode]
                stats[key] = {
                    **stats_by_mode[mode],
                    "shared_with": first_key_by_mode[mode],
                }
                continue

            importance = editor.build_importance(
                mode,
                edit_grad=edit_grad,
                fisher=fisher,
                coordinate_multiplier=coordinate_multiplier,
            )
            masks[key], stats[key] = editor.generate_topk_mask(
                importance,
                budget=config["edit"]["budget"],
                device="cpu",
            )
            logger.log(
                f"Mask {key}: selected={stats[key]['selected_params']} "
                f"ratio={stats[key]['selected_ratio'] * 100:.4f}%"
            )
            masks_by_mode[mode] = masks[key]
            stats_by_mode[mode] = stats[key]
            first_key_by_mode[mode] = key
            del importance
        edit_grad_cpu = {
            name: value.detach().cpu()
            for name, value in edit_grad.items()
        }
        return masks, stats, edit_grad_cpu
    finally:
        del editor, model, fisher, coordinate_multiplier, edit_grad
        release_cuda()


def run_variant(key, data, mask, edit_grad, config, device, logger):
    model = None
    editor = None
    mask_on_device = None
    try:
        model = load_vit_model(device)
        editor = GradGeoEdit(model, device)
        spec = VARIANTS[key]
        logger.log(f"Running {key}: {spec['name']}")
        if spec["optimizer"] == "one_step":
            metrics = editor.one_step_edit(
                data=data,
                mask=mask,
                edit_grad=edit_grad,
                alpha=config["ablation"]["one_step_alpha"],
            )
        elif spec["optimizer"] == "static":
            mask_on_device = {name: value.to(device) for name, value in mask.items()}
            metrics = editor.train_static(
                data=data,
                mask=mask_on_device,
                lr=config["edit"]["lr"],
                steps=config["edit"]["steps"],
                lambda_edit=config["edit"]["lambda_edit"],
                lambda_locality=config["edit"]["lambda_locality"],
                lambda_reg=config["edit"]["lambda_reg"],
                eval_interval=config["edit"]["eval_interval"],
                logger=logger,
            )
        else:
            mask_on_device = {name: value.to(device) for name, value in mask.items()}
            metrics = editor.train_two_stage(
                data=data,
                mask=mask_on_device,
                lr=config["edit"]["lr"],
                steps=config["edit"]["steps"],
                warmup_steps=config["edit"]["warmup_steps"],
                ramp_steps=config["edit"]["ramp_steps"],
                lambda_edit_base=config["edit"]["lambda_edit_base"],
                lambda_edit_max=config["edit"]["lambda_edit"],
                lambda_locality=config["edit"]["lambda_locality"],
                lambda_reg=config["edit"]["lambda_reg"],
                eval_interval=config["edit"]["eval_interval"],
                logger=logger,
            )
        return {name: float(value) for name, value in metrics.items()}
    finally:
        del editor, model, mask_on_device
        release_cuda()


def summarize_repeats(rows):
    metrics = ["reliability", "locality", "generalization"]
    summary = {}
    for metric in metrics:
        values = np.array([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    summary["repeat_count"] = len(rows)
    summary["repeats"] = rows
    return summary


def aggregate(records):
    summary = {}
    for key in VARIANTS:
        rows = records[key]
        if not rows:
            summary[key] = None
            continue
        metric_rows = rows
        if key == "A":
            metric_rows = [repeat for row in rows for repeat in row["repeats"]]
        values = {
            metric: np.array([row[metric] for row in metric_rows], dtype=np.float64)
            for metric in ["reliability", "locality", "generalization"]
        }
        summary[key] = {
            "name": VARIANTS[key]["name"],
            "count": len(metric_rows),
            "R_mean": float(values["reliability"].mean()),
            "R_std": float(values["reliability"].std(ddof=0)),
            "L_mean": float(values["locality"].mean()),
            "L_std": float(values["locality"].std(ddof=0)),
            "G_mean": float(values["generalization"].mean()),
            "G_std": float(values["generalization"].std(ddof=0)),
        }
        if key == "A":
            summary[key]["group_count"] = len(rows)
            summary[key]["random_repeats_per_group"] = int(rows[0]["repeat_count"])
    return summary


def run(args):
    config = config_from_args(args)

    set_seed(config.get("seed", 42))

    ablation_cfg = config.get("ablation", {})
    one_step_alpha = ablation_cfg["one_step_alpha"]
    random_repeats = ablation_cfg["random_repeats"]
    if one_step_alpha <= 0:
        raise ValueError("ablation.one_step_alpha must be positive")
    if random_repeats <= 0:
        raise ValueError("ablation.random_repeats must be positive")
    logger = Logger(
        ablation_cfg.get("exp_name", "ablation"),
        save_dir=ablation_cfg.get("output_dir", "results"),
    )
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.log(f"Device: {device}")
    logger.log("S_n is held out from loss; S_r_eval is held out from locality loss")
    logger.log(
        f"One-step alpha={one_step_alpha}; random repeats={random_repeats}; "
        "localization uses a Fisher-conditioned random coordinate multiplier"
    )
    variant_keys = args.variant or list(VARIANTS)
    logger.log(f"Selected variants: {', '.join(variant_keys)}")

    manager = DatasetManager(
        natural_root=config["data"]["natural_root"],
        sensitive_root=config["data"]["sensitive_root"],
        random_seed=config.get("seed", 42),
    )
    groups = manager.get_all_groups()
    required_count = ablation_cfg.get("require_group_count")
    if required_count is not None and len(groups) != required_count:
        raise ValueError(f"Expected {required_count} groups, found {len(groups)}")

    selected = list(enumerate(groups))
    if args.group is not None:
        selected = [(index, groups[index]) for index in args.group]

    records = {key: [] for key in VARIANTS}
    group_payloads = []

    for group_index, group_name in selected:
        logger.log("=" * 60)
        logger.log(f"Group {group_index}: {group_name}")
        data = manager.prepare_edit_group(
            group_name=group_name,
            num_anchor_samples=config["data"]["num_anchor_samples"],
            num_edit_samples=config["data"].get("num_edit_samples"),
            num_neighborhood_samples=config["data"].get("num_neighborhood_samples"),
            edit_ratio_after_anchor=config["data"]["edit_ratio_after_anchor"],
            num_retain_train_samples=config["data"]["num_retain_train_samples"],
            num_retain_eval_samples=config["data"]["num_retain_eval_samples"],
        )
        manager.print_group_info(data)

        masks, mask_stats, edit_grad = build_masks(
            data,
            config,
            device,
            logger,
            variant_keys,
        )
        group_results = {}

        if "A" in variant_keys:
            random_rows = []
            random_stats = []
            for repeat_index in range(random_repeats):
                random_seed = (
                    config.get("seed", 42)
                    + group_index * random_repeats
                    + repeat_index
                )
                mask_builder = GradGeoEdit(None, device)
                random_mask, stats = mask_builder.generate_random_mask(
                    edit_grad,
                    budget=config["edit"]["budget"],
                    seed=random_seed,
                    device="cpu",
                )
                logger.log(
                    f"Mask A repeat {repeat_index + 1}/{random_repeats}: "
                    f"seed={random_seed} selected={stats['selected_params']} "
                    f"ratio={stats['selected_ratio'] * 100:.4f}%"
                )
                metrics = run_variant("A", data, random_mask, edit_grad, config, device, logger)
                metrics["seed"] = float(random_seed)
                random_rows.append(metrics)
                random_stats.append(stats)
                logger.log(f"A repeat {repeat_index + 1} final: {metrics}")
                del mask_builder, random_mask, stats, metrics
                release_cuda()

            random_summary = summarize_repeats(random_rows)
            records["A"].append(random_summary)
            group_results["A"] = random_summary
            mask_stats["A"] = {
                "repeat_count": random_repeats,
                "repeats": random_stats,
            }
            logger.log(
                f"A final mean: R={random_summary['reliability']:.4f} "
                f"L={random_summary['locality']:.4f} "
                f"G={random_summary['generalization']:.4f}"
            )

        for key in [key for key in variant_keys if key != "A"]:
            metrics = run_variant(key, data, masks[key], edit_grad, config, device, logger)
            records[key].append(metrics)
            group_results[key] = metrics
            logger.log(f"{key} final: {metrics}")

        payload = {
            "group_index": group_index,
            "group_name": group_name,
            "true_label": data["true_label"],
            "wrong_label": data["wrong_label"],
            "split_sizes": {
                "edit": len(data["edit_samples"]),
                "anchor": len(data["anchor_samples"]),
                "neighborhood_eval": len(data["neighborhood_samples"]),
                "retain_train": len(data["retain_train_samples"]),
                "retain_eval": len(data["retain_eval_samples"]),
            },
            "mask_stats": mask_stats,
            "variants": group_results,
        }
        group_payloads.append(payload)
        with open(logger.get_log_dir() / f"group_{group_index}_{group_name}.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        with open(logger.get_log_dir() / "summary_running.json", "w", encoding="utf-8") as handle:
            json.dump(aggregate(records), handle, indent=2)

        del data, masks, mask_stats, edit_grad, group_results, payload
        release_cuda()

    summary = aggregate(records)
    result = {
        "protocol": {
            "budget": float(config["edit"]["budget"]),
            "one_step_alpha": float(one_step_alpha),
            "random_repeats": int(random_repeats),
            "coordinate_multiplier": "fisher_conditioned_gaussian",
            "selected_variants": variant_keys,
            "variants": VARIANTS,
        },
        "groups": group_payloads,
        "summary": summary,
    }
    with open(logger.get_log_dir() / "ablation_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    logger.save_metrics(summary)

    for key, row in summary.items():
        if row is None:
            logger.log(f"{key}: no result")
            continue
        logger.log(
            f"{key} {row['name']}: "
            f"R={row['R_mean'] * 100:.2f}+/-{row['R_std'] * 100:.2f} "
            f"L={row['L_mean'] * 100:.2f}+/-{row['L_std'] * 100:.2f} "
            f"G={row['G_mean'] * 100:.2f}+/-{row['G_std'] * 100:.2f} "
            f"n={row['count']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="GeoEdit A-F ablation")
    add_runtime_arguments(parser)
    parser.add_argument("--group", type=int, action="append", default=None)
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS),
        action="append",
        default=None,
        help="Run only the selected variant; repeat the option for multiple variants",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
