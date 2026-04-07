import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import json
import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any


class ResearchTrainer:
    """
    Canonical SpectralNet Trainer (v1.2).

    Changes relative to v1.1:
      - Added best-val checkpoint saving (P1).
        best_checkpoint.pt is saved upon every improvement in val accuracy.
      - save_artifact_lineage() expanded: records seed and final metrics.
      - Added get_summary() for aggregating multi-seed results.
    """

    def __init__(self, model: nn.Module, config: DictConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logging.getLogger("SpectralNet.Trainer")

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = nn.CrossEntropyLoss()

        # Best-val tracking (P1)
        self.best_val_acc: float = 0.0
        self.best_val_epoch: int = 0

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.infra.save_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_optimizer(self):
        opt = self.config.training.optimizer
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
        )

    def _setup_scheduler(self):
        sched = self.config.training.get("scheduler", None)
        if sched is None:
            return None
        if sched.type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sched.T_max, eta_min=sched.eta_min,
            )
        if sched.type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sched.step_size, gamma=sched.gamma,
            )
        return None

    # ------------------------------------------------------------------
    # Audit & Lineage
    # ------------------------------------------------------------------

    def perform_numerical_audit(self, sample_input: torch.Tensor):
        """gradcheck of the entire model on a couple of examples in float64."""
        self.logger.info("🚀 Starting Numerical Audit (gradcheck)...")
        model_double = self.model.to(torch.float64)
        x_double = sample_input.to(torch.float64).requires_grad_(True)
        try:
            passed = torch.autograd.gradcheck(
                model_double, (x_double,), eps=1e-6, atol=1e-4, nondet_tol=1e-7,
            )
            if passed:
                self.logger.info("✅ Numerical Audit passed.")
            self.model.float()
            self.logger.info("✅ Model converted back to float32.")
        except Exception as e:
            self.logger.error(f"❌ Numerical Audit FAILED: {e}")
            if self.config.training.strict_audit:
                raise RuntimeError("Gradcheck failed.")

    def save_artifact_lineage(self, final_metrics: Dict[str, Any] = None):
        """Saves a digital fingerprint of the run."""
        lineage = {
            "run_id":          self.run_id,
            "seed":            self.config.training.seed,
            "config":          OmegaConf.to_container(self.config),
            "device":          self.device,
            "timestamp":       datetime.now().isoformat(),
            "best_val_acc":    self.best_val_acc,
            "best_val_epoch":  self.best_val_epoch,
        }
        if final_metrics:
            lineage["final_metrics"] = final_metrics
        path = os.path.join(self.output_dir, "lineage.json")
        with open(path, "w") as f:
            json.dump(lineage, f, indent=4)
        self.logger.info(f"💾 Lineage saved: {path}")

    # ------------------------------------------------------------------
    # Train / Validate
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            if self.config.training.get("clip_grad", False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(loader)

    def validate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        correct, val_loss = 0, 0.0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        return {
            "loss":     val_loss / len(loader),
            "accuracy": 100.0 * correct / len(loader.dataset),
            "lr":       self.optimizer.param_groups[0]["lr"],
        }

    # ------------------------------------------------------------------
    # Best checkpoint (P1)
    # ------------------------------------------------------------------

    def update_best(self, epoch: int, metrics: Dict[str, Any]):
        """Saves a checkpoint if the current val accuracy is the best so far."""
        acc = metrics["accuracy"]
        if acc > self.best_val_acc:
            self.best_val_acc = acc
            self.best_val_epoch = epoch
            path = os.path.join(self.output_dir, "best_checkpoint.pt")
            torch.save(
                {
                    "epoch":           epoch,
                    "val_acc":         acc,
                    "model_state":     self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "seed":            self.config.training.seed,
                },
                path,
            )
            self.logger.info(
                f"💾 Best checkpoint @ epoch {epoch:03d} | "
                f"Val Acc: {acc:.2f}% → {path}"
            )

    # ------------------------------------------------------------------
    # Summary for multi-seed aggregation
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Returns the final metrics of the run."""
        return {
            "seed":           self.config.training.seed,
            "run_id":         self.run_id,
            "best_val_acc":   self.best_val_acc,
            "best_val_epoch": self.best_val_epoch,
            "output_dir":     self.output_dir,
        }
