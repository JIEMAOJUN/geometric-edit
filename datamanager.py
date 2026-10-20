import os
import random
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torchvision import transforms


class DatasetManager:
    def __init__(
        self,
        natural_root: str,
        sensitive_root: str,
        random_seed: int = 42,
    ):
        self.natural_root = natural_root
        self.sensitive_root = sensitive_root
        self.rng = random.Random(random_seed)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.retain_pool = self._load_images(self.sensitive_root, "retain")
        print(f"Loaded {len(self.retain_pool)} retain images")

    def _load_images(self, folder: str, image_type: str) -> List[Dict]:
        samples = []
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Path not found: {folder}")

        for image_name in sorted(os.listdir(folder)):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(folder, image_name)
            try:
                image = Image.open(image_path).convert("RGB")
                samples.append({
                    "image": self.transform(image),
                    "path": image_path,
                    "type": image_type,
                })
            except Exception as exc:
                print(f"Failed to load {image_path}: {exc}")

        self.rng.shuffle(samples)
        return samples

    def get_all_groups(self) -> List[str]:
        groups = []
        for item in os.listdir(self.natural_root):
            path = os.path.join(self.natural_root, item)
            if not os.path.isdir(path):
                continue
            parts = item.split("_")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                groups.append(item)
        return sorted(groups)

    def prepare_edit_group(
        self,
        group_name: str,
        num_anchor_samples: int = 20,
        num_edit_samples: Optional[int] = None,
        num_neighborhood_samples: Optional[int] = None,
        edit_ratio_after_anchor: float = 0.8,
        num_retain_train_samples: int = 50,
        num_retain_eval_samples: int = 100,
    ) -> Dict:
        true_label, wrong_label = self._parse_group_name(group_name)

        group_path = os.path.join(self.natural_root, group_name)
        natural_samples = self._load_images(group_path, "natural")
        if not natural_samples:
            raise ValueError(f"No samples in group {group_name}")

        for sample in natural_samples:
            sample["label"] = true_label
            sample["true_label"] = true_label
            sample["wrong_label"] = wrong_label

        if num_anchor_samples <= 0:
            raise ValueError("num_anchor_samples must be positive")
        if num_anchor_samples >= len(natural_samples):
            raise ValueError(
                f"Group {group_name} has {len(natural_samples)} natural samples, "
                f"but num_anchor_samples={num_anchor_samples}"
            )

        anchor_samples = natural_samples[:num_anchor_samples]
        remaining = natural_samples[num_anchor_samples:]
        edit_count, neighborhood_count = self._resolve_edit_neighborhood_counts(
            remaining_count=len(remaining),
            num_edit_samples=num_edit_samples,
            num_neighborhood_samples=num_neighborhood_samples,
            edit_ratio_after_anchor=edit_ratio_after_anchor,
        )

        retain_needed = num_retain_train_samples + num_retain_eval_samples
        if retain_needed > len(self.retain_pool):
            raise ValueError(
                f"Retain pool has {len(self.retain_pool)} samples, "
                f"but requested train={num_retain_train_samples}, eval={num_retain_eval_samples}"
            )

        retain_train = [
            self._without_labels(sample)
            for sample in self.retain_pool[:num_retain_train_samples]
        ]
        retain_eval = [
            self._without_labels(sample)
            for sample in self.retain_pool[num_retain_train_samples:retain_needed]
        ]

        return {
            "group_name": group_name,
            "true_label": true_label,
            "wrong_label": wrong_label,
            "edit_samples": remaining[:edit_count],
            "anchor_samples": anchor_samples,
            "neighborhood_samples": remaining[edit_count:edit_count + neighborhood_count],
            "retain_train_samples": retain_train,
            "retain_eval_samples": retain_eval,
        }

    def print_group_info(self, data: Dict) -> None:
        print("=" * 60)
        print(f"Group: {data['group_name']}")
        print(f"Labels: true={data['true_label']}, wrong={data['wrong_label']}")
        print(f"S_e edit:          {len(data['edit_samples'])}")
        print(f"S_a anchor:        {len(data['anchor_samples'])}")
        print(f"S_n eval:          {len(data['neighborhood_samples'])}")
        print(f"S_r train:         {len(data['retain_train_samples'])}")
        print(f"S_r eval:          {len(data['retain_eval_samples'])}")
        print("=" * 60)

    @staticmethod
    def _parse_group_name(group_name: str) -> Tuple[int, int]:
        parts = group_name.split("_")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"Invalid group name: {group_name}")
        return int(parts[0]), int(parts[1])

    @staticmethod
    def _without_labels(sample: Dict) -> Dict:
        copied = dict(sample)
        for key in ("label", "true_label", "wrong_label"):
            copied.pop(key, None)
        return copied

    @staticmethod
    def _resolve_edit_neighborhood_counts(
        remaining_count: int,
        num_edit_samples: Optional[int],
        num_neighborhood_samples: Optional[int],
        edit_ratio_after_anchor: float,
    ) -> Tuple[int, int]:
        if remaining_count < 2:
            raise ValueError("At least two samples are required after anchor split")

        if num_edit_samples is None and num_neighborhood_samples is None:
            edit_count = max(1, int(remaining_count * edit_ratio_after_anchor))
            neighborhood_count = remaining_count - edit_count
        elif num_edit_samples is None:
            neighborhood_count = int(num_neighborhood_samples)
            edit_count = remaining_count - neighborhood_count
        elif num_neighborhood_samples is None:
            edit_count = int(num_edit_samples)
            neighborhood_count = remaining_count - edit_count
        else:
            edit_count = int(num_edit_samples)
            neighborhood_count = int(num_neighborhood_samples)

        if edit_count <= 0:
            raise ValueError("edit sample count must be positive")
        if neighborhood_count <= 0:
            raise ValueError("neighborhood sample count must be positive")
        if edit_count + neighborhood_count > remaining_count:
            raise ValueError(
                f"Requested edit={edit_count}, neighborhood={neighborhood_count}, "
                f"but only {remaining_count} samples remain after anchors"
            )
        return edit_count, neighborhood_count
