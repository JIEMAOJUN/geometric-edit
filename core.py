from copy import deepcopy
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class GradGeoEdit:
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.original_predictions = {}

    def compute_fisher(
        self,
        anchor_samples: List[Dict],
        batch_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
        }

        self.model.eval()
        for sample in anchor_samples:
            x = self._image(sample)
            if "label" not in sample:
                raise KeyError("Anchor samples must contain 'label'")
            y = torch.tensor([sample["label"]], device=self.device)

            self.model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(x), y)
            loss.backward()
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher[name].add_(param.grad.detach().pow(2))

        count = max(len(anchor_samples), 1)
        for name in fisher:
            fisher[name].div_(count).add_(1e-8)

        self.model.zero_grad(set_to_none=True)
        for param in self.model.parameters():
            param.grad = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return fisher

    def compute_coordinate_multiplier(
        self,
        fisher: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        multiplier = {}
        for name, value in fisher.items():
            epsilon = 1e-3
            noise = epsilon * torch.randn_like(value) * 0.1
            perturbed = value + noise
            multiplier[name] = ((perturbed - value) / (epsilon * value.pow(2) + 1e-8)).abs()
        return multiplier

    def compute_edit_grad(
        self,
        edit_samples: List[Dict],
        true_label: int,
    ) -> Dict[str, torch.Tensor]:
        edit_grad = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
        }

        self.model.eval()
        for sample in edit_samples:
            x = self._image(sample)
            y = torch.tensor([true_label], device=self.device)
            self.model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(x), y)
            loss.backward()
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        edit_grad[name].add_(param.grad.detach())

        count = max(len(edit_samples), 1)
        for name in edit_grad:
            edit_grad[name].div_(count)
        self.model.zero_grad(set_to_none=True)
        for param in self.model.parameters():
            param.grad = None
        return edit_grad

    def build_importance(
        self,
        mode: str,
        edit_grad: Dict[str, torch.Tensor],
        fisher: Optional[Dict[str, torch.Tensor]] = None,
        coordinate_multiplier: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if mode == "grad":
            return {name: grad.abs() for name, grad in edit_grad.items()}
        if mode == "fisher":
            if fisher is None:
                raise ValueError("fisher is required for mode='fisher'")
            return {name: (fisher[name] * edit_grad[name]).abs() for name in edit_grad}
        if mode == "gis":
            if fisher is None or coordinate_multiplier is None:
                raise ValueError("fisher and coordinate_multiplier are required for mode='gis'")
            return {
                name: (fisher[name] * edit_grad[name]).abs() * coordinate_multiplier[name].abs()
                for name in edit_grad
            }
        raise ValueError(f"Unknown importance mode: {mode}")

    def generate_topk_mask(
        self,
        importance: Dict[str, torch.Tensor],
        budget: float,
        device: str = "cpu",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        names = list(importance.keys())
        shapes = {}
        counts = {}
        flat_parts = []
        for name in names:
            score = importance[name].detach().float().abs().cpu().reshape(-1)
            shapes[name] = importance[name].shape
            counts[name] = score.numel()
            flat_parts.append(score)

        all_scores = torch.cat(flat_parts)
        total = all_scores.numel()
        selected = max(1, min(total, int(total * budget)))
        top_indices = torch.topk(all_scores, selected, largest=True, sorted=False).indices
        flat_mask = torch.zeros(total, dtype=torch.float32)
        flat_mask[top_indices] = 1.0

        masks = {}
        offset = 0
        for name in names:
            count = counts[name]
            masks[name] = flat_mask[offset:offset + count].reshape(shapes[name]).to(device)
            offset += count

        return masks, {
            "total_params": int(total),
            "selected_params": int(selected),
            "selected_ratio": float(selected / total),
            "budget": float(budget),
        }

    def generate_random_mask(
        self,
        reference: Dict[str, torch.Tensor],
        budget: float,
        seed: int,
        device: str = "cpu",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        names = list(reference.keys())
        shapes = {name: reference[name].shape for name in names}
        counts = {name: reference[name].numel() for name in names}
        total = sum(counts.values())
        selected = max(1, min(total, int(total * budget)))

        indices = random.Random(seed).sample(range(total), selected)
        flat_mask = torch.zeros(total, dtype=torch.float32)
        flat_mask[torch.tensor(indices, dtype=torch.long)] = 1.0

        masks = {}
        offset = 0
        for name in names:
            count = counts[name]
            masks[name] = flat_mask[offset:offset + count].reshape(shapes[name]).to(device)
            offset += count

        return masks, {
            "total_params": int(total),
            "selected_params": int(selected),
            "selected_ratio": float(selected / total),
            "budget": float(budget),
            "seed": int(seed),
        }

    def cache_original_predictions(self, samples: List[Dict]) -> None:
        self.original_predictions = {}
        self.model.eval()
        with torch.no_grad():
            for index, sample in enumerate(samples):
                self.original_predictions[index] = self.model(self._image(sample)).argmax(dim=-1).item()

    def cache_original_logits(self, samples: List[Dict]) -> Dict[int, torch.Tensor]:
        logits = {}
        self.model.eval()
        with torch.no_grad():
            for index, sample in enumerate(samples):
                logits[index] = self.model(self._image(sample)).detach().cpu()
        return logits

    def train_static(
        self,
        data: Dict,
        mask: Dict[str, torch.Tensor],
        lr: float,
        steps: int,
        lambda_edit: float,
        lambda_locality: float,
        lambda_reg: float,
        eval_interval: int = 5,
        logger=None,
    ) -> Dict[str, float]:
        return self._train(
            data=data,
            mask=mask,
            lr=lr,
            steps=steps,
            lambda_edit=lambda_edit,
            lambda_locality=lambda_locality,
            lambda_reg=lambda_reg,
            eval_interval=eval_interval,
            warmup_steps=0,
            ramp_steps=1,
            two_stage=False,
            logger=logger,
        )

    def train_two_stage(
        self,
        data: Dict,
        mask: Dict[str, torch.Tensor],
        lr: float,
        steps: int,
        warmup_steps: int,
        ramp_steps: int,
        lambda_edit_base: float,
        lambda_edit_max: float,
        lambda_locality: float,
        lambda_reg: float,
        eval_interval: int = 5,
        logger=None,
    ) -> Dict[str, float]:
        return self._train(
            data=data,
            mask=mask,
            lr=lr,
            steps=steps,
            lambda_edit=lambda_edit_max,
            lambda_locality=lambda_locality,
            lambda_reg=lambda_reg,
            eval_interval=eval_interval,
            warmup_steps=warmup_steps,
            ramp_steps=ramp_steps,
            two_stage=True,
            lambda_edit_base=lambda_edit_base,
            logger=logger,
        )

    def evaluate(self, data: Dict) -> Dict[str, float]:
        retain_eval = data.get("retain_eval_samples", data.get("non_neighborhood_samples", []))
        return self._evaluate(
            data["edit_samples"],
            data["neighborhood_samples"],
            retain_eval,
            data["true_label"],
        )

    def one_step_edit(
        self,
        data: Dict,
        mask: Dict[str, torch.Tensor],
        edit_grad: Dict[str, torch.Tensor],
        alpha: float,
    ) -> Dict[str, float]:
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        retain_eval = data.get("retain_eval_samples", data.get("non_neighborhood_samples", []))
        self.cache_original_predictions(retain_eval)

        norm_sq = 0.0
        selected = 0
        for name, grad in edit_grad.items():
            if name not in mask:
                continue
            mask_cpu = mask[name].detach().float().cpu()
            grad_cpu = grad.detach().float().cpu()
            masked_grad = mask_cpu * grad_cpu
            norm_sq += float(masked_grad.pow(2).sum())
            selected += int(mask_cpu.sum().item())

        masked_grad_norm = norm_sq ** 0.5
        if masked_grad_norm == 0.0:
            self.original_predictions = {}
            raise ValueError("Masked edit gradient norm is zero")

        scale = alpha / (masked_grad_norm + 1e-12)
        try:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name not in mask or name not in edit_grad:
                        continue
                    masked_grad = (
                        mask[name].to(param.device, dtype=param.dtype)
                        * edit_grad[name].to(param.device, dtype=param.dtype)
                    )
                    param.add_(masked_grad, alpha=-scale)
                    del masked_grad

            metrics = self.evaluate(data)
            return {
                **metrics,
                "one_step_alpha": float(alpha),
                "masked_grad_norm": float(masked_grad_norm),
                "selected_params": int(selected),
            }
        finally:
            self.original_predictions = {}
            self.model.zero_grad(set_to_none=True)
            for param in self.model.parameters():
                param.grad = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _train(
        self,
        data: Dict,
        mask: Dict[str, torch.Tensor],
        lr: float,
        steps: int,
        lambda_edit: float,
        lambda_locality: float,
        lambda_reg: float,
        eval_interval: int,
        warmup_steps: int,
        ramp_steps: int,
        two_stage: bool,
        logger,
        lambda_edit_base: float = 0.0,
    ) -> Dict[str, float]:
        retain_train = data.get("retain_train_samples", data.get("non_neighborhood_samples", []))
        retain_eval = data.get("retain_eval_samples", data.get("non_neighborhood_samples", []))

        self.cache_original_predictions(retain_eval)
        train_logits = self.cache_original_logits(retain_train)
        original_params, mask_device, hooks = self._prepare_masked_training(mask)
        params = [param for name, param in self.model.named_parameters() if name in original_params]
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

        try:
            for step in range(steps):
                self.model.train()
                optimizer.zero_grad(set_to_none=True)

                if two_stage and step < warmup_steps:
                    edit_weight = 0.0
                    loc_weight = lambda_locality
                elif two_stage:
                    ramp = min(1.0, float(step - warmup_steps + 1) / max(ramp_steps, 1))
                    edit_weight = lambda_edit_base + lambda_edit * ramp
                    loc_weight = lambda_locality
                else:
                    edit_weight = lambda_edit
                    loc_weight = lambda_locality

                edit_loss = torch.tensor(0.0, device=self.device)
                if edit_weight > 0:
                    edit_loss = self._backward_edit_loss(data["edit_samples"], data["true_label"], edit_weight)

                loc_loss = self._backward_locality_loss(retain_train, train_logits, loc_weight)
                reg_loss = self._regularization_loss(original_params, mask_device)
                (lambda_reg * reg_loss).backward()

                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                self._restore_unmasked_values(original_params, mask_device)

                if self._should_log(step, steps, eval_interval):
                    metrics = self.evaluate(data)
                    self._log(
                        logger,
                        f"step {step + 1:03d}/{steps} "
                        f"Le={float(edit_loss):.4f} Ll={float(loc_loss):.4f} "
                        f"R={metrics['reliability']:.4f} "
                        f"L={metrics['locality']:.4f} "
                        f"G={metrics['generalization']:.4f}",
                    )

                if torch.cuda.is_available() and (step + 1) % 5 == 0:
                    torch.cuda.empty_cache()
            final_metrics = self.evaluate(data)
        finally:
            for hook in hooks:
                hook.remove()
            for param in self.model.parameters():
                param.requires_grad = True
                param.grad = None
            self.original_predictions = {}
            self.model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return final_metrics

    def _backward_edit_loss(
        self,
        samples: List[Dict],
        true_label: int,
        weight: float,
    ) -> torch.Tensor:
        total = 0.0
        count = max(len(samples), 1)
        for sample in samples:
            x = self._image(sample)
            y = torch.tensor([true_label], device=self.device)
            loss = F.cross_entropy(self.model(x), y) / count
            total += float(loss.detach())
            (weight * loss).backward()
            del x, y, loss
        return torch.tensor(total, device=self.device)

    def _backward_locality_loss(
        self,
        samples: List[Dict],
        original_logits: Dict[int, torch.Tensor],
        weight: float,
    ) -> torch.Tensor:
        if not samples or weight <= 0:
            return torch.tensor(0.0, device=self.device)

        total = 0.0
        count = len(samples)
        for index, sample in enumerate(samples):
            x = self._image(sample)
            logits_new = self.model(x)
            logits_old = original_logits[index].to(self.device)
            loss = F.kl_div(
                F.log_softmax(logits_new, dim=1),
                F.softmax(logits_old, dim=1),
                reduction="batchmean",
            ) / count
            total += float(loss.detach())
            (weight * loss).backward()
            del x, logits_new, logits_old, loss
        return torch.tensor(total, device=self.device)

    def _regularization_loss(
        self,
        original_params: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in original_params:
                delta = (param - original_params[name]) * mask[name]
                loss = loss + delta.pow(2).sum()
        return loss / max(len(original_params), 1)

    def _prepare_masked_training(
        self,
        mask: Dict[str, torch.Tensor],
    ):
        original_params = {}
        mask_device = {}
        for name, param in self.model.named_parameters():
            if name in mask and bool(mask[name].any().item()):
                original_params[name] = param.detach().clone()
                mask_device[name] = mask[name].to(param.device)

        for name, param in self.model.named_parameters():
            param.requires_grad = name in original_params

        hooks = []
        for name, param in self.model.named_parameters():
            if name in mask_device:
                active_mask = mask_device[name]

                def make_hook(mask_tensor):
                    def hook_fn(grad):
                        return grad * mask_tensor
                    return hook_fn

                hooks.append(param.register_hook(make_hook(active_mask)))
        return original_params, mask_device, hooks

    def _restore_unmasked_values(
        self,
        original_params: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name] + (param.data - original_params[name]) * mask[name])

    def _evaluate(
        self,
        edit_samples,
        neighborhood_samples,
        retain_samples,
        true_label,
    ) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            reliability = self._target_accuracy(edit_samples, true_label)
            generalization = self._target_accuracy(neighborhood_samples, true_label)
            locality = self._locality(retain_samples)
        return {
            "reliability": reliability,
            "locality": locality,
            "generalization": generalization,
        }

    def _target_accuracy(self, samples: List[Dict], true_label: int) -> float:
        if not samples:
            return 0.0
        correct = 0
        for sample in samples:
            if self.model(self._image(sample)).argmax(dim=-1).item() == true_label:
                correct += 1
        return correct / len(samples)

    def _locality(self, samples: List[Dict]) -> float:
        if not samples:
            return 0.0
        total = 0
        consistent = 0
        for index, sample in enumerate(samples):
            before = self.original_predictions.get(index)
            if before is None:
                continue
            total += 1
            now = self.model(self._image(sample)).argmax(dim=-1).item()
            if now == before:
                consistent += 1
        return consistent / total if total else 0.0

    def _image(self, sample: Dict) -> torch.Tensor:
        x = sample["image"].to(self.device, non_blocking=True)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def _should_log(step: int, steps: int, interval: int) -> bool:
        return step == 0 or (step + 1) % interval == 0 or step + 1 == steps

    @staticmethod
    def _log(logger, message: str) -> None:
        if logger is None:
            print(message)
        else:
            logger.log(message)


def copy_model_state(model) -> Dict[str, torch.Tensor]:
    return deepcopy(model.state_dict())


def restore_model_state(model, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)
