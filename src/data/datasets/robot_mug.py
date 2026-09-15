from env import CACHE, ROBOT_MUG_PATH
from pathlib import Path
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from src.data.utils import split_dataset
from typing import Dict, List

class RobotMugDataset(Dataset):
    def __init__(self,
                 test_size: float = 0.2,
                 ftune_size: float = 0.0,
                 ftune_val_size: float = 0.0):
        
        self.test_size = test_size
        self.ftune_size = ftune_size
        self.ftune_val_size = ftune_val_size

        self.c_info = {'names': None, 'cardinality': None}
        self.y_info = {'names': ['success'], 'cardinality': [2]}
        self.data = {}

    def load_ground_truth_graph(self):
        return None

    def split(self, ckpt_config=None):
        data_root = ROBOT_MUG_PATH

        episodes = {}
        for folder in sorted(os.listdir(data_root)):
            parts = folder.split('_')  # ['rollout', '0000', 'failure', 'left']
            if len(parts) != 4 or parts[0] != 'rollout':
                continue
            ep_num = parts[1]
            label = 0 if parts[2] == 'failure' else 1

            if ep_num not in episodes:
                episodes[ep_num] = {'label': label, 'frames': []}

            folder_path = os.path.join(data_root, folder)
            for frame in sorted(Path(folder_path).glob("*.jpg")):
                episodes[ep_num]['frames'].append(str(frame))

        rows = [
            {'episode': ep_num,
            'frames': info['frames'],
            'label': info['label']}
            for ep_num, info in sorted(episodes.items())
        ]

        print(f"Loaded {len(rows)} episodes")
        print(f"Success: {sum(r['label'] for r in rows)}, Failure: {sum(1-r['label'] for r in rows)}")

        full_dataset = _RobotMugDataset(rows)
        self.data['train'], data_val_test = split_dataset(full_dataset, 0.3)
        self.data['val'], data_test = split_dataset(data_val_test, self.test_size)
        self.data['val'].split_type = 'val'
        self.data['test'] = data_test
        self.data['test'].split_type = 'test'


class _RobotMugDataset(Dataset):
    def __init__(self, rows: List[Dict], transform_config=None):
        super().__init__()
        self.rows = rows
        self.X = None
        self.c = None
        self.y = None
        self.graph = {}
        self.split_type = 'train'

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.rows)

    def register_graph(self, graph):
        self.graph = graph

    def __getitem__(self, index):
        if self.X is None:
            row = self.rows[index]
            frame_path = row['frames'][len(row['frames']) // 2]  # middle frame
            image = Image.open(frame_path).convert("RGB")
            image = self.transform(image)
            label = torch.Tensor([row['label']])
            c = torch.zeros_like(label)
        else:
            image = self.X[index]
            c = self.c[index]
            label = self.y[index]

        return {"x": image, "c": c, "y": label, "graph": self.graph}

    def collate_fn(self, instances: List):
        images = torch.stack([ins["x"] for ins in instances])
        c = torch.stack([ins["c"] for ins in instances])
        labels = torch.stack([ins["y"] for ins in instances])
        graph = instances[0]["graph"]
        return {"x": images, "c": c, "y": labels, "graph": graph}