from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from app.models.schemas import BugDatasetRecord


def _parse_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in re.split(r"[^0-9]+", value):
        if not token:
            continue
        parts.append(int(token))
    return tuple(parts)


class _TokenizedBugDataset(Dataset):
    def __init__(self, encodings: dict[str, list[int]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class _DatasetSubset(Dataset):
    def __init__(self, dataset: _TokenizedBugDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.dataset[self.indices[index]]


class ClassificationService:
    def __init__(
        self,
        *,
        base_checkpoint_path: str,
        finetuned_root_path: str,
        train_epochs: int,
        learning_rate: float,
        batch_size: int,
        max_length: int,
        gradient_accumulation_steps: int,
        early_stopping_patience: int,
        mixed_precision_enabled: bool,
    ) -> None:
        self.base_checkpoint_path = Path(base_checkpoint_path)
        self.finetuned_root_path = Path(finetuned_root_path)
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.early_stopping_patience = max(1, early_stopping_patience)
        self.mixed_precision_enabled = mixed_precision_enabled
        self.enabled = False
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model = None
        self.id_to_label: dict[int, str] = {}
        self.active_checkpoint_path: Path | None = None
        self.training_metadata: dict[str, object] | None = None
        self._prediction_lock = Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.finetuned_root_path.mkdir(parents=True, exist_ok=True)
        latest_checkpoint = self._find_latest_finetuned_checkpoint()
        if latest_checkpoint is not None:
            self.load_checkpoint(latest_checkpoint)

    @property
    def torch_version(self) -> str:
        return str(torch.__version__)

    @property
    def base_checkpoint_available(self) -> bool:
        return self.base_checkpoint_path.exists()

    @property
    def active_checkpoint_name(self) -> str | None:
        active_checkpoint_path = self.active_finetuned_checkpoint_path
        if active_checkpoint_path is None:
            return None
        return active_checkpoint_path.name

    @property
    def active_finetuned_checkpoint_path(self) -> Path | None:
        checkpoint_path = self.active_checkpoint_path
        if checkpoint_path is None:
            return None

        resolved_checkpoint_path = checkpoint_path.resolve()
        if not resolved_checkpoint_path.exists() or not (resolved_checkpoint_path / "config.json").exists():
            return None

        try:
            resolved_checkpoint_path.relative_to(self.finetuned_root_path.resolve())
        except ValueError:
            return None

        return resolved_checkpoint_path

    @property
    def has_active_finetuned_checkpoint(self) -> bool:
        return self.active_finetuned_checkpoint_path is not None

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        resolved_path = Path(checkpoint_path)
        self._validate_checkpoint_runtime(resolved_path)
        tokenizer = AutoTokenizer.from_pretrained(str(resolved_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(resolved_path))
        model.to(self.device)
        model.eval()

        config_labels = getattr(model.config, "id2label", {}) or {}
        id_to_label = {int(key): value for key, value in config_labels.items()}

        with self._prediction_lock:
            self.tokenizer = tokenizer
            self.model = model
            self.id_to_label = id_to_label
            self.active_checkpoint_path = resolved_path.resolve()
            self.enabled = True

    def fine_tune(
        self,
        records: list[BugDatasetRecord],
        source_name: str,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, object]:
        if not self.base_checkpoint_available:
            raise ValueError(
                f"Base checkpoint was not found at '{self.base_checkpoint_path}'."
            )

        self._validate_checkpoint_runtime(self.base_checkpoint_path)

        if len(records) < 2:
            raise ValueError("At least two bug records are required to fine-tune the classifier.")

        developer_names = sorted({record.developer_name for record in records})
        if len(developer_names) < 2:
            raise ValueError(
                "At least two unique developers are required to fine-tune the classifier."
            )

        label_to_id = {developer_name: index for index, developer_name in enumerate(developer_names)}
        id_to_label = {index: developer_name for developer_name, index in label_to_id.items()}
        texts = [self._combine_bug_text(record) for record in records]
        labels = [label_to_id[record.developer_name] for record in records]

        if progress_callback is not None:
            progress_callback(
                "preparing",
                5,
                f"Validated {len(records)} bug records across {len(developer_names)} developers.",
            )

        tokenizer = AutoTokenizer.from_pretrained(str(self.base_checkpoint_path))
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        dataset = _TokenizedBugDataset(encodings, labels)
        collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if self.device.type == "cuda" else None,
        )

        if progress_callback is not None:
            progress_callback("tokenizing", 15, "Tokenized bug reports for fine-tuning.")

        train_indices, validation_indices = self._build_train_validation_split(labels)
        train_dataset = _DatasetSubset(dataset, train_indices)
        validation_dataset = _DatasetSubset(dataset, validation_indices) if validation_indices else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            collate_fn=collator,
            pin_memory=self.device.type == "cuda",
        )
        validation_loader = (
            DataLoader(
                validation_dataset,
                batch_size=min(self.batch_size, len(validation_dataset)),
                shuffle=False,
                collate_fn=collator,
                pin_memory=self.device.type == "cuda",
            )
            if validation_dataset is not None
            else None
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.base_checkpoint_path),
            num_labels=len(label_to_id),
            id2label=id_to_label,
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)

        class_weights = self._compute_class_weights(labels, len(label_to_id)).to(self.device)
        amp_enabled, amp_dtype, use_grad_scaler = self._get_amp_settings()
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

        steps_per_epoch = max(math.ceil(len(train_loader) / self.gradient_accumulation_steps), 1)
        total_steps = max(steps_per_epoch * self.train_epochs, 1)
        warmup_steps = max(total_steps // 10, 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        completed_steps = 0
        best_epoch = 0
        best_metric = float("-inf")
        best_validation_accuracy: float | None = None
        best_validation_loss: float | None = None
        stopped_early = False
        patience_counter = 0
        best_model_state_path: Path | None = None

        with tempfile.TemporaryDirectory(prefix="classifier-best-", dir=str(self.finetuned_root_path)) as tmp_dir:
            best_model_state_path = Path(tmp_dir) / "best-model-state.pt"
            for epoch_index in range(self.train_epochs):
                model.train()
                epoch_loss_total = 0.0
                optimizer.zero_grad(set_to_none=True)

                for batch_index, batch in enumerate(train_loader, start=1):
                    prepared_batch = {key: value.to(self.device) for key, value in batch.items()}
                    labels_tensor = prepared_batch.pop("labels")

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        outputs = model(**prepared_batch)
                        raw_loss = F.cross_entropy(outputs.logits, labels_tensor, weight=class_weights)

                    loss = raw_loss / self.gradient_accumulation_steps
                    epoch_loss_total += float(raw_loss.item())

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    should_step = (
                        batch_index % self.gradient_accumulation_steps == 0
                        or batch_index == len(train_loader)
                    )
                    if should_step:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        completed_steps += 1

                        if progress_callback is not None:
                            progress_callback(
                                "training",
                                15 + ((completed_steps / total_steps) * 70),
                                f"Fine-tuning epoch {epoch_index + 1} of {self.train_epochs}.",
                            )

                average_train_loss = epoch_loss_total / max(len(train_loader), 1)
                validation_metrics = self._evaluate(model, validation_loader, class_weights)
                epoch_score = (
                    validation_metrics["accuracy"]
                    if validation_metrics is not None
                    else -average_train_loss
                )

                if epoch_score > best_metric:
                    best_metric = epoch_score
                    best_epoch = epoch_index + 1
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_state_path)
                    if validation_metrics is not None:
                        best_validation_accuracy = float(validation_metrics["accuracy"])
                        best_validation_loss = float(validation_metrics["loss"])
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        stopped_early = True
                        break

            if best_model_state_path.exists():
                best_state_dict = torch.load(best_model_state_path, map_location=self.device)
                model.load_state_dict(best_state_dict)

        output_dir = self._build_output_dir(source_name)
        output_dir.mkdir(parents=True, exist_ok=False)

        if progress_callback is not None:
            progress_callback("saving", 92, "Saving fine-tuned checkpoint.")

        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        training_summary = {
            "source_name": source_name,
            "base_checkpoint_path": str(self.base_checkpoint_path.resolve()),
            "output_checkpoint_path": str(output_dir.resolve()),
            "epochs": self.train_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "mixed_precision_enabled": amp_enabled,
            "mixed_precision_dtype": str(amp_dtype).replace("torch.", "") if amp_enabled else None,
            "early_stopping_patience": self.early_stopping_patience,
            "stopped_early": stopped_early,
            "trained_records": len(records),
            "developer_count": len(developer_names),
            "training_records": len(train_indices),
            "validation_records": len(validation_indices),
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_loss": best_validation_loss,
            "created_at": datetime.now(UTC).isoformat(),
        }
        summary_path = output_dir / "training_summary.json"
        summary_path.write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

        if progress_callback is not None:
            progress_callback("loading", 98, "Loading fine-tuned checkpoint for recommendations.")

        self.load_checkpoint(output_dir)
        self.training_metadata = training_summary
        self._prune_stale_checkpoints(output_dir)
        return training_summary

    def clear_finetuned_checkpoints(self) -> dict[str, object]:
        deleted_checkpoint_paths: list[str] = []

        for checkpoint_dir in self._list_finetuned_checkpoint_dirs():
            shutil.rmtree(checkpoint_dir, ignore_errors=False)
            deleted_checkpoint_paths.append(str(checkpoint_dir.resolve()))

        self.unload_checkpoint()
        return {
            "deleted_checkpoints": len(deleted_checkpoint_paths),
            "deleted_checkpoint_paths": deleted_checkpoint_paths,
            "base_checkpoint_path": str(self.base_checkpoint_path.resolve()),
            "active_checkpoint_path": None,
        }

    def unload_checkpoint(self) -> None:
        with self._prediction_lock:
            self.tokenizer = None
            self.model = None
            self.id_to_label = {}
            self.active_checkpoint_path = None
            self.enabled = False
            self.training_metadata = None

    def predict(self, text: str, top_k: int) -> list[dict[str, float | str]]:
        with self._prediction_lock:
            tokenizer = self.tokenizer
            model = self.model
            id_to_label = dict(self.id_to_label)

        if not self.has_active_finetuned_checkpoint or tokenizer is None or model is None:
            return []

        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prepared_inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**prepared_inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0]
        limit = min(top_k, probabilities.shape[0])
        scores, indices = torch.topk(probabilities, k=limit)

        predictions: list[dict[str, float | str]] = []
        for score, index in zip(scores.tolist(), indices.tolist(), strict=True):
            developer_name = id_to_label.get(index, str(index))
            predictions.append(
                {
                    "developer_name": developer_name,
                    "model_score": float(score),
                }
            )
        return predictions

    def _find_latest_finetuned_checkpoint(self) -> Path | None:
        checkpoint_dirs = self._list_finetuned_checkpoint_dirs()
        if not checkpoint_dirs:
            return None
        return max(checkpoint_dirs, key=lambda path: path.stat().st_mtime)

    def _list_finetuned_checkpoint_dirs(self) -> list[Path]:
        if not self.finetuned_root_path.exists():
            return []

        return [
            path
            for path in self.finetuned_root_path.iterdir()
            if path.is_dir() and (path / "config.json").exists()
        ]

    def _prune_stale_checkpoints(self, keep_checkpoint: Path) -> None:
        resolved_keep_checkpoint = keep_checkpoint.resolve()
        for checkpoint_dir in self._list_finetuned_checkpoint_dirs():
            if checkpoint_dir.resolve() == resolved_keep_checkpoint:
                continue
            shutil.rmtree(checkpoint_dir, ignore_errors=False)

    def _build_output_dir(self, source_name: str) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        source_stem = Path(source_name).stem or "bug-dataset"
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", source_stem).strip("-").lower() or "bug-dataset"
        return self.finetuned_root_path / f"{timestamp}-{slug}"

    def _build_train_validation_split(self, labels: list[int]) -> tuple[list[int], list[int]]:
        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            label_to_indices[label].append(index)

        validation_indices: list[int] = []
        for label_indices in label_to_indices.values():
            if len(label_indices) < 2:
                continue

            validation_count = max(1, round(len(label_indices) * 0.2))
            validation_count = min(validation_count, len(label_indices) - 1)
            validation_indices.extend(label_indices[-validation_count:])

        validation_index_set = set(validation_indices)
        train_indices = [index for index in range(len(labels)) if index not in validation_index_set]

        if not validation_indices or not train_indices:
            return list(range(len(labels))), []

        return train_indices, sorted(validation_indices)

    @staticmethod
    def _compute_class_weights(labels: list[int], num_labels: int) -> torch.Tensor:
        label_counts = Counter(labels)
        total_records = len(labels)
        weights = [0.0] * num_labels

        for label_index in range(num_labels):
            label_count = label_counts.get(label_index, 1)
            weights[label_index] = total_records / max(label_count * num_labels, 1)

        return torch.tensor(weights, dtype=torch.float32)

    def _evaluate(
        self,
        model,
        data_loader: DataLoader | None,
        class_weights: torch.Tensor,
    ) -> dict[str, float] | None:
        if data_loader is None or len(data_loader.dataset) == 0:
            return None

        model.eval()
        total_loss = 0.0
        total_examples = 0
        total_correct = 0
        amp_enabled, amp_dtype, _ = self._get_amp_settings()

        with torch.no_grad():
            for batch in data_loader:
                prepared_batch = {key: value.to(self.device) for key, value in batch.items()}
                labels_tensor = prepared_batch.pop("labels")
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    outputs = model(**prepared_batch)
                    loss = F.cross_entropy(outputs.logits, labels_tensor, weight=class_weights)
                predictions = torch.argmax(outputs.logits, dim=-1)

                batch_size = int(labels_tensor.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size
                total_correct += int((predictions == labels_tensor).sum().item())

        model.train()
        if total_examples == 0:
            return None

        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }

    def _validate_checkpoint_runtime(self, checkpoint_path: Path) -> None:
        if self._has_safetensors_weights(checkpoint_path):
            return

        if self._torch_meets_minimum((2, 6, 0)):
            return

        raise ValueError(
            "Bug dataset fine-tuning requires torch>=2.6.0 when loading checkpoints stored as pytorch_model.bin. "
            f"The checkpoint at '{checkpoint_path}' does not include safetensors weights, and the current torch version is {self.torch_version}. "
            "Upgrade torch to at least 2.6.0 or replace the checkpoint weights with safetensors. "
            "See CVE-2025-32434 for details."
        )

    @staticmethod
    def _has_safetensors_weights(checkpoint_path: Path) -> bool:
        return any(checkpoint_path.glob("*.safetensors"))

    def _torch_meets_minimum(self, minimum_version: tuple[int, ...]) -> bool:
        current_version = _parse_version(self.torch_version)
        padded_current = current_version + (0,) * max(0, len(minimum_version) - len(current_version))
        return padded_current >= minimum_version

    def _get_amp_settings(self) -> tuple[bool, torch.dtype, bool]:
        if self.device.type != "cuda" or not self.mixed_precision_enabled:
            return False, torch.float32, False

        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False

        return True, torch.float16, True

    @staticmethod
    def _combine_bug_text(record: BugDatasetRecord) -> str:
        return f"Title: {record.title.strip()}\nDescription: {record.description.strip()}"
