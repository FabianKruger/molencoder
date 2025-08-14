from pathlib import Path
import csv
from typing import Any, Dict, List, Optional, Set
from transformers import TrainerCallback, TrainingArguments

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, output_dir: Path, filename: str = "validation_metrics_log.csv") -> None:
        self.filepath: Path = output_dir / filename
        self.fieldnames: Set[str] = set()  # Track all keys seen so far.
        self.rows: List[Dict[str, Any]] = []  # Store all logged rows in memory.

    def on_log(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        if logs is None:
            return

        # Only log entries that contain evaluation metrics
        if not any(key.startswith("eval_") for key in logs.keys()):
            return

        new_keys = set(logs.keys()) - self.fieldnames
        if new_keys:
            self.fieldnames.update(new_keys)

        self.rows.append(logs.copy())

        fieldnames_list = sorted(self.fieldnames)
        # Re-open and rewrite the entire CSV file with updated header and all rows
        with self.filepath.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_list)
            writer.writeheader()
            for row in self.rows:
                filtered_row = {field: row.get(field, "") for field in fieldnames_list}
                writer.writerow(filtered_row)



class BestEpochTracker(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = float("inf")
        self.best_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control  # Nothing to do if no metrics provided

        current_loss = metrics.get("eval_loss")
        if current_loss is not None and current_loss < self.best_eval_loss:
            self.best_eval_loss = current_loss
            self.best_epoch = metrics.get("epoch")
        return control
