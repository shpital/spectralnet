import hydra
from omegaconf import DictConfig
import torch
import logging

from src.spectralnet.data.loaders import get_loaders, get_num_classes
from src.spectralnet.models.spectralnet_s import SpectralNetS
from src.spectralnet.training.loop import ResearchTrainer


@hydra.main(
    config_path="../../../configs",
    config_name="experiment/exp005_modes8_multiseed",
    version_base=None,
)
def main(cfg: DictConfig):
    logger = logging.getLogger("SpectralNet")
    logger.info(f"📊 Starting experiment: {cfg.name} | seed={cfg.training.seed}")

    # --- Reproducibility ---
    # Full seed control: Python, NumPy, PyTorch CPU/CUDA
    seed = cfg.training.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic ops — slightly slower, but guarantee reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥  Device: {device}")

    # --- Data ---
    dataset_name = cfg.dataset.name
    train_loader, test_loader = get_loaders(
        dataset_name=dataset_name,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.infra.num_workers,
        seed=seed,
    )

    # --- Model ---
    # num_classes is passed from the dataset — the head is built for the target dataset
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["num_classes"] = get_num_classes(dataset_name)
    model_cfg = DictConfig(model_cfg)
    model = SpectralNetS(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🏗  Model: {cfg.model.name} | params: {n_params:,}")

    # --- Trainer ---
    trainer = ResearchTrainer(model, cfg, device)

    # --- Numerical Audit ---
    # gradcheck of the entire model on a couple of examples
    # (SpectralRemizovLayer is already checked in isolation in P0 — this is an additional
    # check of the entire forward pass including ResolventAggregation and head)
    sample_data, _ = next(iter(train_loader))
    trainer.perform_numerical_audit(sample_data[:2].to(device))

    # --- Lineage (before training) ---
    trainer.save_artifact_lineage()

    # --- Main Loop ---
    logger.info("🟢 Starting training...")
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        metrics    = trainer.validate(test_loader)

        # Save best checkpoint upon improvement (P1)
        trainer.update_best(epoch, metrics)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.2f}% | "
            f"LR: {metrics['lr']:.2e} | "
            f"Best: {trainer.best_val_acc:.2f}% (ep {trainer.best_val_epoch:03d})"
        )

    # --- Final summary ---
    summary = trainer.get_summary()
    logger.info(
        f"🏁 Run completed | seed={seed} | "
        f"Best Val Acc: {summary['best_val_acc']:.2f}% @ epoch {summary['best_val_epoch']}"
    )

    # Update lineage with final metrics
    trainer.save_artifact_lineage(
        final_metrics={
            "best_val_acc":   summary["best_val_acc"],
            "best_val_epoch": summary["best_val_epoch"],
        }
    )


if __name__ == "__main__":
    main()
