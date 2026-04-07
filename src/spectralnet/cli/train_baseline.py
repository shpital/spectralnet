import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torchvision.models as tvm
import logging
from src.spectralnet.data.loaders import get_loaders, get_num_classes
from src.spectralnet.training.loop import ResearchTrainer


def build_baseline(cfg) -> nn.Module:
    name = cfg.model.baseline_type
    num_classes = get_num_classes(cfg.dataset.name)

    if name == "resnet18":
        m = tvm.resnet18(weights=None, num_classes=num_classes)
    elif name == "mobilenetv2":
        m = tvm.mobilenet_v2(weights=None, num_classes=num_classes)
    elif name == "shufflenetv2":
        m = tvm.shufflenet_v2_x0_5(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown baseline_type: {name}")
    return m


@hydra.main(
    config_path="../../../configs",
    config_name="experiment/exp_baseline_cifar10",
    version_base=None,
)
def main(cfg: DictConfig):
    logger = logging.getLogger("SpectralNet")
    logger.info(f"📊 Starting experiment: {cfg.name} | seed={cfg.training.seed}")

    seed = cfg.training.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥  Device: {device}")

    dataset_name = cfg.dataset.name
    train_loader, test_loader = get_loaders(
        dataset_name=dataset_name,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.infra.num_workers,
        seed=seed,
    )

    model = build_baseline(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🏗  Model: {cfg.model.baseline_type} | params: {n_params:,}")

    trainer = ResearchTrainer(model, cfg, device)
    trainer.save_artifact_lineage()

    logger.info("🟢 Starting training...")
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        metrics    = trainer.validate(test_loader)
        trainer.update_best(epoch, metrics)
        logger.info(
            f"Epoch {epoch:03d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.2f}% | "
            f"LR: {metrics['lr']:.2e} | "
            f"Best: {trainer.best_val_acc:.2f}% (ep {trainer.best_val_epoch:03d})"
        )

    summary = trainer.get_summary()
    logger.info(
        f"🏁 Run completed | seed={seed} | "
        f"Best Val Acc: {summary['best_val_acc']:.2f}% @ epoch {summary['best_val_epoch']}"
    )
    trainer.save_artifact_lineage(
        final_metrics={
            "best_val_acc":   summary["best_val_acc"],
            "best_val_epoch": summary["best_val_epoch"],
        }
    )


if __name__ == "__main__":
    main()
