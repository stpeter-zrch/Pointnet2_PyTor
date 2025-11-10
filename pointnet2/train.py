import os
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pointnet2.data.RadarPDWDataset import RadarPDWDataset
from pytorch_lightning.loggers import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update({k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()})
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v
        return res
    return _to_dot_dict(hparams)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    # === 加载PDW数据集 ===
    train_set = RadarPDWDataset('data/pdw_data.npz', split='train')
    val_set = RadarPDWDataset('data/pdw_data.npz', split='val')

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))

    # 创建模型保存目录
    save_dir = os.path.join("outputs", cfg.task_model.name)
    os.makedirs(save_dir, exist_ok=True)

    # 提前定义 EarlyStopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    # ✅ 使用新版 API
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        filename="{epoch}-{val_loss:.2f}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        verbose=True
    )

    # ✅ 新版 Trainer 参数
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=list(cfg.gpus),
        max_epochs=cfg.get("epochs", 100),
        callbacks=[early_stop_callback, checkpoint_callback],
        strategy=cfg.distrib_backend,
        logger=TensorBoardLogger(save_dir)
    )


    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
