from collections import defaultdict
from pathlib import Path
import trimesh
import numpy as np
from tqdm import tqdm
from torchvision.datasets import DatasetFolder


def create_dataset(src_dir, n_samples, n_points, out_dir, train):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    else:
        print(f'{out_dir} already exists.')
        return

    n_class_samples = defaultdict(lambda: 1)
    src_paths = list(Path(src_dir).glob('*/{}/*.off'.format('train' if train else 'test')))
    for src_path in tqdm(src_paths):
        class_name = src_path.parts[-3]
        class_dir = out_dir / class_name
        if not class_dir.exists():
            class_dir.mkdir()

        mesh = trimesh.load(src_path)
        for _ in range(n_samples):
            sampled_points = mesh.sample(n_points)
            np.save(class_dir / f'{n_class_samples[class_name]:08}.npy', sampled_points)
            n_class_samples[class_name] += 1


def load_sample(path):
    return np.load(path)


class ModelNet40(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(ModelNet40, self).__init__(root, load_sample, ('npy', ), transform, target_transform)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
