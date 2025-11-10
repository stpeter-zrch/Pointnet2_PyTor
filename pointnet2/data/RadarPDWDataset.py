import numpy as np
import torch
from torch.utils.data import Dataset

class RadarPDWDataset(Dataset):
    """
    雷达信号分选数据集。
    每个样本是一个点云，由 PDW 特征 [TOA, PW, CF, PA] 组成。
    标签是每个点的辐射源类别。
    """
    def __init__(self, data_path, split='train', normalize=True):
        data = np.load(data_path, allow_pickle=True)
        self.points = data[f'{split}_points']   # shape: [num_samples, num_points, 4]
        self.labels = data[f'{split}_labels']   # shape: [num_samples, num_points]
        self.normalize = normalize

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pts = self.points[idx].astype(np.float32)
        lbl = self.labels[idx].astype(np.int64)

        if self.normalize:
            pts[:, :3] = (pts[:, :3] - pts[:, :3].min(axis=0)) / \
                         (pts[:, :3].max(axis=0) - pts[:, :3].min(axis=0) + 1e-6)

        coords = torch.from_numpy(pts[:, :3])   # TOA, PW, CF
        feats = torch.from_numpy(pts[:, 3:])    # PA
        lbl = torch.from_numpy(lbl)

        return coords, feats, lbl
