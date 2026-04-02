import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict

# mapping CDL code → index de classe
CDL_TO_IDX_ARK = {
    1:  0,   # Corn
    2:  1,   # Cotton
    5:  2,   # Soybean
    3:  3,   # Rice
    -1: 4,   # Others (tous les codes non listés)
}

CDL_TO_IDX_CAL = {
    69: 0,   # Grapes
    3:  1,   # Rice
    36: 2,   # Alfalfa
    75: 3,   # Almonds
    74: 4,   # Pistachios
    -1: 5,   # Others
}

class CropDataset(Dataset):
    """
    Dataset PyTorch pour MCTNet
    
    Charge les arrays numpy et retourne des tenseurs PyTorch.
    
    Chaque sample :
        x     : (T, n_bands) = (36, 10) — float32
        mask  : (T,)         = (36,)    — float32, 1=valide 0=manquant
        label : ()           — int64
    """
    def __init__(
        self,
        bands_path:  str,
        masks_path:  str,
        labels_path: str,
        state:       str = 'arkansas',   # 'arkansas' ou 'california'
        indices:     np.ndarray = None,  # pour train/val/test split
        normalize:   bool = True,
        norm_stats:  Dict = None,        # {'mean': array, 'std': array}
    ):
        # chargement
        bands  = np.load(bands_path).astype(np.float32)   # (N, T, 10)
        masks  = np.load(masks_path).astype(np.float32)   # (N, T)
        labels_raw = np.load(labels_path)                 # (N,)

        # mapping CDL → index classe
        cdl_map = CDL_TO_IDX_ARK if state == 'arkansas' else CDL_TO_IDX_CAL
        labels  = np.array([
            cdl_map.get(int(l), cdl_map[-1])
            for l in labels_raw
        ], dtype=np.int64)

        # filtre minority classes < 5% → "Others"
        # (déjà fait pendant le sampling mais on vérifie)
        n_total = len(labels)
        for cls_idx in np.unique(labels):
            if cls_idx == cdl_map[-1]:
                continue
            if np.sum(labels == cls_idx) / n_total < 0.05:
                labels[labels == cls_idx] = cdl_map[-1]

        # normalisation par bande : divise par 10000 comme dans le papier
        # (les valeurs Sentinel-2 sont en réflectance ×10000)
        if normalize:
            if norm_stats is not None:
                mean = norm_stats['mean']   # (10,)
                std  = norm_stats['std']    # (10,)
                bands = (bands - mean) / (std + 1e-8)
            else:
                # normalisation simple du papier : /10000
                bands = bands / 10000.0

        # sous-ensemble selon indices
        if indices is not None:
            bands  = bands[indices]
            masks  = masks[indices]
            labels = labels[indices]

        self.bands  = torch.from_numpy(bands)
        self.masks  = torch.from_numpy(masks)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        return self.bands[idx], self.masks[idx], self.labels[idx]


def make_splits(
    n_total:    int,
    n_per_class: int = 300,
    val_ratio:   float = 0.2,
    labels:      np.ndarray = None,
    seed:        int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crée les splits train/val/test exactement comme le papier :
    - 300 samples par classe pour train+val (80/20)
    - reste pour test
    
    Returns: (train_idx, val_idx, test_idx)
    """
    np.random.seed(seed)
    train_idx = []
    val_idx   = []
    test_idx  = []

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)

        # 300 pour train+val
        selected  = cls_indices[:n_per_class]
        remaining = cls_indices[n_per_class:]

        n_val   = int(n_per_class * val_ratio)   # 60
        n_train = n_per_class - n_val            # 240

        train_idx.extend(selected[:n_train].tolist())
        val_idx.extend(selected[n_train:].tolist())
        test_idx.extend(remaining.tolist())

    return (
        np.array(train_idx),
        np.array(val_idx),
        np.array(test_idx)
    )