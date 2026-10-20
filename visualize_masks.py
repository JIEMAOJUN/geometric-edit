import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch

from core import GradGeoEdit
from datamanager import DatasetManager
from run import load_vit_model, set_seed
from settings import add_runtime_arguments, config_from_args


def release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def flatten_features(fisher_tensor, multiplier_tensor, grad_tensor):
    fisher_score = (fisher_tensor * grad_tensor).detach().abs().float().cpu().reshape(-1)
    multiplier_score = multiplier_tensor.detach().abs().float().cpu().reshape(-1)
    gis_score = fisher_score * multiplier_score
    return fisher_score, multiplier_score, gis_score


def streaming_topk_ids(fisher, coordinate_multiplier, edit_grad, mode, k):
    top_values = None
    top_ids = None
    offset = 0
    layout = []

    for name in edit_grad:
        fisher_score, multiplier_score, gis_score = flatten_features(
            fisher[name],
            coordinate_multiplier[name],
            edit_grad[name],
        )
        score = fisher_score if mode == "fisher" else gis_score
        local_k = min(k, score.numel())
        values, local_ids = torch.topk(score, local_k, largest=True, sorted=False)
        global_ids = local_ids.to(torch.int64) + offset

        if top_values is None:
            top_values = values
            top_ids = global_ids
        else:
            merged_values = torch.cat([top_values, values])
            merged_ids = torch.cat([top_ids, global_ids])
            keep_k = min(k, merged_values.numel())
            keep_values, keep_pos = torch.topk(merged_values, keep_k, largest=True, sorted=False)
            top_values = keep_values
            top_ids = merged_ids[keep_pos]

        layout.append({"name": name, "offset": offset, "count": score.numel()})
        offset += score.numel()
        del fisher_score, multiplier_score, gis_score, score, values, local_ids, global_ids

    return top_ids.numpy(), layout, offset


def log_feature(values, eps=1e-30):
    return np.log10(np.asarray(values, dtype=np.float64) + eps)


def sample_background(layout, fisher, coordinate_multiplier, edit_grad, max_points, rng):
    xs, ys, zs = [], [], []
    total_params = sum(item["count"] for item in layout)
    for item in layout:
        name = item["name"]
        count = item["count"]
        take = int(max_points * count / max(total_params, 1))
        take = min(max(take, 1), count)
        local_ids = torch.from_numpy(rng.choice(count, size=take, replace=False)).long()
        fisher_score, multiplier_score, gis_score = flatten_features(
            fisher[name],
            coordinate_multiplier[name],
            edit_grad[name],
        )
        xs.append(log_feature(fisher_score[local_ids].numpy()))
        ys.append(log_feature(multiplier_score[local_ids].numpy()))
        zs.append(log_feature(gis_score[local_ids].numpy()))
        del fisher_score, multiplier_score, gis_score, local_ids
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)


def selected_points(layout, fisher, coordinate_multiplier, edit_grad, fisher_ids, gis_ids, max_points, rng):
    fisher_ids = np.unique(fisher_ids)
    gis_ids = np.unique(gis_ids)
    both_ids = np.intersect1d(fisher_ids, gis_ids, assume_unique=True)
    fisher_only_ids = np.setdiff1d(fisher_ids, both_ids, assume_unique=True)
    gis_only_ids = np.setdiff1d(gis_ids, both_ids, assume_unique=True)

    groups = {
        "fisher_only": fisher_only_ids,
        "gis_only": gis_only_ids,
        "overlap": both_ids,
    }

    points = {}
    for label, ids in groups.items():
        if ids.size > max_points:
            ids = rng.choice(ids, size=max_points, replace=False)
        ids = np.sort(ids)
        xs, ys, zs = [], [], []
        for item in layout:
            start = item["offset"]
            end = start + item["count"]
            left = np.searchsorted(ids, start, side="left")
            right = np.searchsorted(ids, end, side="left")
            if left == right:
                continue
            local_ids = torch.from_numpy(ids[left:right] - start).long()
            name = item["name"]
            fisher_score, multiplier_score, gis_score = flatten_features(
                fisher[name],
                coordinate_multiplier[name],
                edit_grad[name],
            )
            xs.append(log_feature(fisher_score[local_ids].numpy()))
            ys.append(log_feature(multiplier_score[local_ids].numpy()))
            zs.append(log_feature(gis_score[local_ids].numpy()))
            del fisher_score, multiplier_score, gis_score, local_ids

        if xs:
            points[label] = {
                "x": np.concatenate(xs),
                "y": np.concatenate(ys),
                "z": np.concatenate(zs),
            }
        else:
            points[label] = {"x": np.array([]), "y": np.array([]), "z": np.array([])}

    return points, {
        "fisher_only": int(fisher_only_ids.size),
        "gis_only": int(gis_only_ids.size),
        "overlap": int(both_ids.size),
        "fisher_total": int(fisher_ids.size),
        "gis_total": int(gis_ids.size),
        "jaccard": float(both_ids.size / max(np.union1d(fisher_ids, gis_ids).size, 1)),
    }


def empty_selection_record():
    return {
        "total_params": 0,
        "fisher_only": 0,
        "overlap": 0,
        "gis_only": 0,
    }


def parameter_architecture_location(name):
    if name == "class_token" or name == "encoder.pos_embedding":
        return "tokens", None
    if name.startswith("conv_proj."):
        return "patch_embedding", None
    if name.startswith("encoder.ln."):
        return "final_norm", None
    if name.startswith("heads.head."):
        return "classifier", None

    match = re.fullmatch(r"encoder\.layers\.encoder_layer_(\d+)\.(.+)", name)
    if match is None:
        raise ValueError(f"Unrecognized ViT parameter name: {name}")

    layer_index = int(match.group(1))
    suffix = match.group(2)
    if suffix.startswith("self_attention."):
        component = "attention"
    elif suffix.startswith("mlp."):
        component = "mlp"
    elif suffix.startswith("ln_1.") or suffix.startswith("ln_2."):
        component = "norm"
    else:
        raise ValueError(f"Unrecognized encoder parameter name: {name}")
    return "encoder", (layer_index, component)


def architecture_selection_stats(layout, fisher_ids, gis_ids):
    fisher_ids = np.sort(np.unique(fisher_ids))
    gis_ids = np.sort(np.unique(gis_ids))
    endpoints = {
        "patch_embedding": empty_selection_record(),
        "tokens": empty_selection_record(),
        "final_norm": empty_selection_record(),
        "classifier": empty_selection_record(),
    }
    layers = {}

    for item in layout:
        start = item["offset"]
        end = start + item["count"]
        fisher_local = fisher_ids[
            np.searchsorted(fisher_ids, start, side="left"):
            np.searchsorted(fisher_ids, end, side="left")
        ]
        gis_local = gis_ids[
            np.searchsorted(gis_ids, start, side="left"):
            np.searchsorted(gis_ids, end, side="left")
        ]
        overlap = np.intersect1d(fisher_local, gis_local, assume_unique=True)
        location, detail = parameter_architecture_location(item["name"])

        if location == "encoder":
            layer_index, component = detail
            if layer_index not in layers:
                layers[layer_index] = {
                    "norm": empty_selection_record(),
                    "attention": empty_selection_record(),
                    "mlp": empty_selection_record(),
                }
            record = layers[layer_index][component]
        else:
            record = endpoints[location]

        record["total_params"] += int(item["count"])
        record["fisher_only"] += int(fisher_local.size - overlap.size)
        record["overlap"] += int(overlap.size)
        record["gis_only"] += int(gis_local.size - overlap.size)

    ordered_layers = {
        str(index): layers[index]
        for index in sorted(layers)
    }
    return {
        "endpoints": endpoints,
        "layers": ordered_layers,
    }


def visualization_colors():
    return {
        "neutral": "#D9DEE5",
        "neutral_dark": "#5A6472",
        "fisher_only": "#D97706",
        "overlap": "#7C3AED",
        "gis_only": "#078B83",
        "paper": "#FFFFFF",
        "ink": "#17212B",
    }


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save_figure_formats(fig, output_path, dpi=320):
    output_path = Path(output_path)
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(
            output_path.with_suffix(suffix),
            dpi=dpi if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor=visualization_colors()["paper"],
        )


def draw_vit_architecture(ax, architecture, summary, panel_title=None):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    colors = visualization_colors()
    layer_items = [(int(index), value) for index, value in architecture["layers"].items()]
    columns = 4

    all_records = list(architecture["endpoints"].values())
    for _, layer in layer_items:
        all_records.extend(layer.values())
    max_density = max(
        (
            (record["fisher_only"] + record["overlap"] + record["gis_only"])
            / max(record["total_params"], 1)
        )
        for record in all_records
    )
    max_density = max(max_density, 1e-12)

    ax.set_facecolor(colors["paper"])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    def draw_bar(x, y, width, height, record):
        ax.add_patch(Rectangle(
            (x, y), width, height,
            facecolor="#EEF1F4", edgecolor="none", zorder=1,
        ))
        total_params = max(record["total_params"], 1)
        cursor = x
        for key in ["fisher_only", "overlap", "gis_only"]:
            segment_width = width * (record[key] / total_params) / max_density
            if segment_width <= 0:
                continue
            ax.add_patch(Rectangle(
                (cursor, y), segment_width, height,
                facecolor=colors[key], edgecolor="none", zorder=2,
            ))
            cursor += segment_width
        ax.add_patch(Rectangle(
            (x, y), width, height,
            facecolor="none", edgecolor=colors["neutral"], linewidth=0.6, zorder=3,
        ))

    def draw_endpoint(x, y, width, title, record):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, 0.86,
            boxstyle="round,pad=0.025,rounding_size=0.06",
            facecolor="#F7F8FA", edgecolor=colors["neutral_dark"], linewidth=0.9,
        ))
        ax.text(
            x + width / 2, y + 0.57, title,
            ha="center", va="center", color=colors["ink"],
            fontweight="bold", fontsize=8.3,
        )
        draw_bar(x + 0.14, y + 0.17, width - 0.28, 0.13, record)

    if panel_title:
        ax.text(
            6.0, 10.35, panel_title,
            ha="center", va="center", color=colors["ink"],
            fontweight="bold", fontsize=12,
        )
    ax.text(
        0.45, 9.88,
        r"Fisher-only: $|\hat{G}_{ii}g_i|$",
        color=colors["fisher_only"], fontweight="bold", va="center", fontsize=10.2,
    )
    ax.text(
        6.25, 9.88,
        r"GIS: $|\hat{G}_{ii}g_i|\,|\hat{\kappa}_i|$",
        color=colors["gis_only"], fontweight="bold", va="center", fontsize=10.2,
    )
    legend_x = 3.05
    for index, (key, label) in enumerate([
        ("fisher_only", "Fisher-only"),
        ("overlap", "Shared"),
        ("gis_only", "GIS-only"),
    ]):
        x = legend_x + index * 2.2
        ax.add_patch(Rectangle((x, 9.35), 0.20, 0.20, facecolor=colors[key], edgecolor="none"))
        ax.text(x + 0.29, 9.45, label, va="center", color=colors["ink"], fontsize=9.0)

    encoder_record = empty_selection_record()
    for _, layer in layer_items:
        for component in layer.values():
            for key in encoder_record:
                encoder_record[key] += component[key]

    endpoint_y = 8.08
    endpoint_width = 1.95
    endpoint_specs = [
        (0.15, "Patch projection", architecture["endpoints"]["patch_embedding"]),
        (2.55, "Class + position", architecture["endpoints"]["tokens"]),
        (4.95, "12 encoder blocks", encoder_record),
        (7.35, "Final LayerNorm", architecture["endpoints"]["final_norm"]),
        (9.75, "Classifier", architecture["endpoints"]["classifier"]),
    ]
    for x, title, record in endpoint_specs:
        draw_endpoint(x, endpoint_y, endpoint_width, title, record)
    for left, right in [
        (2.10, 2.55),
        (4.50, 4.95),
        (6.90, 7.35),
        (9.30, 9.75),
    ]:
        ax.add_patch(FancyArrowPatch(
            (left, endpoint_y + 0.43), (right, endpoint_y + 0.43),
            arrowstyle="-|>", mutation_scale=11,
            linewidth=1.0, color=colors["neutral_dark"],
        ))

    grid_left = 0.35
    grid_top = 7.30
    block_width = 2.65
    block_height = 1.72
    gap_x = 0.22
    gap_y = 0.24
    component_labels = [
        ("norm", "LayerNorm"),
        ("attention", "MHSA"),
        ("mlp", "MLP"),
    ]

    for position, (layer_index, layer) in enumerate(layer_items):
        row = position // columns
        column = position % columns
        x = grid_left + column * (block_width + gap_x)
        y = grid_top - (row + 1) * block_height - row * gap_y
        ax.add_patch(FancyBboxPatch(
            (x, y), block_width, block_height,
            boxstyle="round,pad=0.025,rounding_size=0.07",
            facecolor="#FAFBFC", edgecolor=colors["neutral_dark"], linewidth=0.85,
        ))
        ax.text(
            x + 0.14, y + block_height - 0.24,
            f"Encoder block {layer_index + 1:02d}",
            ha="left", va="center", color=colors["ink"], fontweight="bold", fontsize=9.0,
        )
        for component_index, (component, label) in enumerate(component_labels):
            component_y = y + block_height - 0.60 - component_index * 0.39
            ax.text(
                x + 0.15, component_y,
                label, ha="left", va="center", color=colors["neutral_dark"], fontsize=8.0,
            )
            draw_bar(
                x + 1.25,
                component_y - 0.075,
                block_width - 1.41,
                0.15,
                layer[component],
            )

    ax.text(
        6.0, 0.48,
        f"Same budget: {summary['top_k']:,} parameters per mask   |   "
        f"Shared: {summary['overlap']:,}   |   Jaccard: {summary['jaccard']:.3f}",
        ha="center", va="center", color=colors["ink"], fontsize=9.2,
    )
    ax.text(
        6.0, 0.15,
        "Bar length = selection density",
        ha="center", va="center", color=colors["neutral_dark"], fontsize=8.8,
    )


def plot_vit_architecture_comparison(architecture, summary, output_path):
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 6.5), dpi=320)
    fig.patch.set_facecolor(visualization_colors()["paper"])
    draw_vit_architecture(
        ax,
        architecture,
        summary,
        panel_title="ViT-B/16 mask distribution",
    )
    fig.tight_layout(pad=0.35)
    save_figure_formats(fig, output_path)
    plt.close(fig)


def plot_3d(background, selected, summary, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    bx, by, bz = background
    ax.scatter(bx, by, bz, s=1, c="#9aa0a6", alpha=0.035, depthshade=False, label="sampled parameters")

    styles = {
        "fisher_only": ("#e6862f", "Fisher-only"),
        "gis_only": ("#2f6fdf", "GIS-only"),
        "overlap": ("#8e44ad", "overlap"),
    }
    for key, (color, label) in styles.items():
        pts = selected[key]
        if pts["x"].size:
            ax.scatter(
                pts["x"],
                pts["y"],
                pts["z"],
                s=8,
                c=color,
                alpha=0.72,
                depthshade=False,
                label=f"{label} ({summary[key]:,})",
            )

    ax.set_xlabel("log10 Fisher-corrected edit gradient")
    ax.set_ylabel("log10 random coordinate multiplier")
    ax.set_zlabel("log10 GIS score")
    ax.set_title("Fisher-only vs GIS parameter selection under the same budget")
    ax.view_init(elev=24, azim=-128)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_2d_projection(ax, selected, summary, panel_title=None):
    colors = visualization_colors()
    styles = {
        "fisher_only": (colors["fisher_only"], "Fisher-only"),
        "gis_only": (colors["gis_only"], "GIS-only"),
        "overlap": (colors["overlap"], "Shared"),
    }
    for key, (color, label) in styles.items():
        pts = selected[key]
        if pts["x"].size:
            ax.scatter(
                pts["x"],
                pts["y"],
                s=7,
                c=color,
                alpha=0.55,
                label=f"{label} ({summary[key]:,})",
            )
    ax.set_xlabel(r"$\log_{10}$ Fisher-corrected edit gradient", fontsize=10.5)
    ax.set_ylabel(r"$\log_{10}$ random coordinate multiplier", fontsize=10.5)
    ax.set_title(
        panel_title or "Random coordinate weighting separates GIS from Fisher-only selection",
        fontsize=12,
        fontweight="bold" if panel_title else "normal",
        pad=10,
    )
    ax.tick_params(axis="both", labelsize=9.5)
    ax.legend(fontsize=8.5, loc="lower left", frameon=True)
    ax.grid(alpha=0.18)


def plot_2d_projection(selected, summary, output_path):
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=320)
    draw_2d_projection(ax, selected, summary)
    fig.tight_layout()
    save_figure_formats(fig, output_path)
    plt.close(fig)


def plot_combined_comparison(selected, architecture, summary, output_path):
    plt = configure_matplotlib()
    colors = visualization_colors()
    fig = plt.figure(figsize=(16.2, 7.0), dpi=320, facecolor=colors["paper"])
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.92, 1.38],
        left=0.055,
        right=0.992,
        bottom=0.105,
        top=0.955,
        wspace=0.16,
    )
    scatter_ax = fig.add_subplot(grid[0, 0])
    architecture_ax = fig.add_subplot(grid[0, 1])
    draw_2d_projection(
        scatter_ax,
        selected,
        summary,
        panel_title="(a) Random-coordinate mask separation",
    )
    draw_vit_architecture(
        architecture_ax,
        architecture,
        summary,
        panel_title="(b) ViT-B/16 mask distribution",
    )
    save_figure_formats(fig, output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize Fisher-only and GIS masks")
    add_runtime_arguments(parser)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/mask_visualization")
    parser.add_argument("--max-background-points", type=int, default=60000)
    parser.add_argument("--max-selected-points", type=int, default=30000)
    parser.add_argument(
        "--include-scatter",
        action="store_true",
        help="Also generate the legacy 3D scatter plot",
    )
    args = parser.parse_args()

    config = config_from_args(args)

    set_seed(config.get("seed", 42))
    rng = np.random.default_rng(config.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = DatasetManager(
        natural_root=config["data"]["natural_root"],
        sensitive_root=config["data"]["sensitive_root"],
        random_seed=config.get("seed", 42),
    )
    groups = manager.get_all_groups()
    group_name = groups[args.group]
    data = manager.prepare_edit_group(
        group_name=group_name,
        num_anchor_samples=config["data"]["num_anchor_samples"],
        num_edit_samples=config["data"].get("num_edit_samples"),
        num_neighborhood_samples=config["data"].get("num_neighborhood_samples"),
        edit_ratio_after_anchor=config["data"]["edit_ratio_after_anchor"],
        num_retain_train_samples=config["data"]["num_retain_train_samples"],
        num_retain_eval_samples=config["data"]["num_retain_eval_samples"],
    )

    model = load_vit_model(device)
    editor = GradGeoEdit(model, device)
    fisher = editor.compute_fisher(data["anchor_samples"], batch_size=config["edit"]["fisher_batch_size"])
    coordinate_multiplier = editor.compute_coordinate_multiplier(fisher)
    edit_grad = editor.compute_edit_grad(data["edit_samples"], data["true_label"])

    total_params = sum(param.numel() for param in edit_grad.values())
    top_k = max(1, int(total_params * config["edit"]["budget"]))

    fisher_ids, layout, _ = streaming_topk_ids(fisher, coordinate_multiplier, edit_grad, "fisher", top_k)
    gis_ids, _, _ = streaming_topk_ids(fisher, coordinate_multiplier, edit_grad, "gis", top_k)

    fisher_unique = np.unique(fisher_ids)
    gis_unique = np.unique(gis_ids)
    overlap_ids = np.intersect1d(fisher_unique, gis_unique, assume_unique=True)
    fisher_only_ids = np.setdiff1d(fisher_unique, overlap_ids, assume_unique=True)
    gis_only_ids = np.setdiff1d(gis_unique, overlap_ids, assume_unique=True)
    summary = {
        "fisher_only": int(fisher_only_ids.size),
        "gis_only": int(gis_only_ids.size),
        "overlap": int(overlap_ids.size),
        "fisher_total": int(fisher_unique.size),
        "gis_total": int(gis_unique.size),
        "jaccard": float(
            overlap_ids.size
            / max(np.union1d(fisher_unique, gis_unique).size, 1)
        ),
    }
    architecture = architecture_selection_stats(layout, fisher_ids, gis_ids)

    summary.update({
        "group_index": args.group,
        "group_name": group_name,
        "budget": config["edit"]["budget"],
        "top_k": int(top_k),
        "total_params": int(total_params),
        "architecture": architecture,
    })

    stem = f"group_{args.group}_{group_name}"
    plot_vit_architecture_comparison(
        architecture,
        summary,
        output_dir / f"{stem}_vit_architecture_comparison.png",
    )
    selected, _ = selected_points(
        layout,
        fisher,
        coordinate_multiplier,
        edit_grad,
        fisher_ids,
        gis_ids,
        args.max_selected_points,
        rng,
    )
    plot_2d_projection(
        selected,
        summary,
        output_dir / f"{stem}_coordinate_projection.png",
    )
    plot_combined_comparison(
        selected,
        architecture,
        summary,
        output_dir / f"{stem}_mask_comparison_combined.png",
    )
    if args.include_scatter:
        background = sample_background(
            layout,
            fisher,
            coordinate_multiplier,
            edit_grad,
            args.max_background_points,
            rng,
        )
        plot_3d(background, selected, summary, output_dir / f"{stem}_fisher_vs_gis_3d.png")
    with open(output_dir / f"{stem}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    del model, editor, fisher, coordinate_multiplier, edit_grad
    release_cuda()
    print(f"Saved visualization to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
