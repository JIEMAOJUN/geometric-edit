import argparse
import random
import sys

import numpy as np
import torch

from core import GradGeoEdit
from datamanager import DatasetManager
from logger import Logger
from settings import add_runtime_arguments, config_from_args

if sys.platform.startswith("win"):
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def load_vit_model(device="cuda"):
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    print(f"ViT-B/16 loaded")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: dict, group_index: int = 0):
    set_seed(config.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = Logger(config["exp_name"])
    logger.log(f"Device: {device}")

    model = load_vit_model(device)
    data_manager = DatasetManager(
        natural_root=config["data"]["natural_root"],
        sensitive_root=config["data"]["sensitive_root"],
    )

    groups = data_manager.get_all_groups()
    if group_index < 0 or group_index >= len(groups):
        raise ValueError(f"Group index {group_index} out of range (0-{len(groups)-1})")

    group_name = groups[group_index]
    data = data_manager.prepare_edit_group(
        group_name=group_name,
        num_anchor_samples=config["data"]["num_anchor_samples"],
        num_edit_samples=config["data"].get("num_edit_samples"),
        num_neighborhood_samples=config["data"].get("num_neighborhood_samples"),
        edit_ratio_after_anchor=config["data"]["edit_ratio_after_anchor"],
        num_retain_train_samples=config["data"]["num_retain_train_samples"],
        num_retain_eval_samples=config["data"]["num_retain_eval_samples"],
    )
    data_manager.print_group_info(data)

    editor = GradGeoEdit(model, device)
    fisher = editor.compute_fisher(
        data["anchor_samples"],
        batch_size=config["edit"].get("fisher_batch_size", 8),
    )
    coordinate_multiplier = editor.compute_coordinate_multiplier(fisher)
    edit_grad = editor.compute_edit_grad(data["edit_samples"], data["true_label"])
    importance = editor.build_importance(
        "gis",
        edit_grad=edit_grad,
        fisher=fisher,
        coordinate_multiplier=coordinate_multiplier,
    )
    mask, mask_stats = editor.generate_topk_mask(
        importance,
        budget=config["edit"]["budget"],
        device=device,
    )
    logger.log(
        f"Mask selected={mask_stats['selected_params']} "
        f"ratio={mask_stats['selected_ratio'] * 100:.4f}%"
    )

    result = editor.train_two_stage(
        data=data,
        mask=mask,
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

    logger.save_metrics(
        {
            "group_name": group_name,
            "true_label": data["true_label"],
            "wrong_label": data["wrong_label"],
            **result,
        }
    )
    logger.log(f"Results saved to: {logger.get_log_dir()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="GeoEdit run")
    add_runtime_arguments(parser)
    parser.add_argument("--group", type=int, default=0)
    args = parser.parse_args()

    config = config_from_args(args)

    run_experiment(config, args.group)


if __name__ == "__main__":
    main()
