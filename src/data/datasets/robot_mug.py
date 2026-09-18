import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class SplitData(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.c          = torch.tensor(X, dtype=torch.float32)
        self.y          = torch.tensor(y, dtype=torch.long).unsqueeze(1)
        self.X          = self.c
        self.complete_c = self.c
        self.graph      = []
        self.split_type = ""

    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.c)

    def __getitem__(self, idx):
        return {
            'x':          self.X[idx],
            'c':          self.c[idx],
            'complete_c': self.complete_c[idx],
            'y':          self.y[idx],
            'graph':      self.graph,
        }


class RobotDataset:
    def __init__(self, dag_name, task_name, csv_path,
                 val_size=0.1, test_size=0.2,
                 ftune_size=0., ftune_val_size=0., **kwargs):
        self.dag_name  = dag_name
        self.task_name = task_name

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[task_name])

        drop_cols = ["episode_idx"]

        # only drop success if it's not the task_name
        if task_name != "success":
            drop_cols.append("success")

        concept_cols = [c for c in df.columns if c not in drop_cols]


        # ── filter to keep_concepts if specified ──────────────────────────────────────
        if kwargs.get("keep_concepts"):
            keep = set(kwargs["keep_concepts"])
            concept_cols = [c for c in concept_cols if c in keep or c == task_name]
            print(f"Filtered to {len(concept_cols)} concepts: {concept_cols}")

        for col in concept_cols:
            df[col] = df[col].map(
                lambda x: 1 if str(x).strip().lower() in ("1", "true", "yes") else 0
            )

        self.feature_names = [c for c in concept_cols if c != task_name]
        X = df[self.feature_names].values.astype(float)
        y = df[task_name].values.astype(float)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_size / (1.0 - test_size),
                random_state=42, stratify=y_train)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_size / (1.0 - test_size),
                random_state=42)

        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.X_test,  self.y_test  = X_test,  y_test

        self.data = {
            'train': SplitData(X_train, y_train),
            'val':   SplitData(X_val,   y_val),
            'test':  SplitData(X_test,  y_test),
        }
        self.data['train'].split_type = 'train'
        self.data['val'].split_type   = 'val'
        self.data['test'].split_type  = 'test'

        self.c_info = {
            'names':       self.feature_names,
            'cardinality': [2] * len(self.feature_names),
        }
        self.c_info_complete = self.c_info
        self.y_info = {
            'names':       [task_name],
            'cardinality': [2],
        }

        print(f"Dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
        print(f"Concepts ({len(self.feature_names)}): {self.feature_names}")
        print(f"Task: {self.y_info['names']}")
        print(f"Success rate: {y.mean():.2f} ({int(y.sum())} success, {int((1-y).sum())} failure)")

    def load_ground_truth_graph(self):
        return None

    def get_concept_names(self):
        return self.feature_names