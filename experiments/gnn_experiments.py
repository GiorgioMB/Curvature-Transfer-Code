import os
import time
import argparse
import random
import copy
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool
from torch_geometric.datasets import ZINC, LRGBDataset, HeterophilousGraphDataset
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
import optuna

from curvature_rewiring import SDRFRewiring, BORFRewiring

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Architectures
# -----------------------------
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, arch="GCN", task="graph", dropout=0.5):
        super().__init__()
        self.task = task
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        
        if arch == "GCN":
            self.layers.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.layers.append(GCNConv(hidden_channels, hidden_channels if task == "graph" else out_channels))
        elif arch == "GIN":
            self.layers.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels)
            )))
            for _ in range(num_layers - 2):
                self.layers.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels)
                )))
            self.layers.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels if task == "graph" else out_channels)
            )))
            
        if task == "graph":
            self.pool = global_mean_pool
            self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1 or self.task == "graph":
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.task == "graph":
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = self.pool(x, batch)
            x = self.lin(x)
        return x

# -----------------------------
# Evaluation Logistics
# -----------------------------
def get_dataset(name):
    if name == "ZINC":
        d_train = ZINC(root="data/ZINC", split="train", subset=True)
        d_val = ZINC(root="data/ZINC", split="val", subset=True)
        d_test = ZINC(root="data/ZINC", split="test", subset=True)
        return list(d_train) + list(d_val) + list(d_test), "graph", "mae", 1, 1
    elif name == "Peptides-func":
        dataset = LRGBDataset(root="data/LRGB", name="Peptides-func")
        return list(dataset), "graph", "ap", dataset.num_node_features, 10
    elif name == "Peptides-struct":
        dataset = LRGBDataset(root="data/LRGB", name="Peptides-struct")
        return list(dataset), "graph", "mae", dataset.num_node_features, 11
    elif name == "minesweeper":
        dataset = HeterophilousGraphDataset(root="data/Heterophilous", name="minesweeper")
        return dataset[0], "node", "rocauc", dataset.num_node_features, 1
    raise ValueError(f"Unknown dataset: {name}")

def compute_metric(y_true, y_pred, metric_name):
    y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    if metric_name == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric_name == "ap":
        valid = ~np.isnan(y_true)
        if not np.any(valid): return 0.0
        y_true[np.isnan(y_true)] = 0
        return average_precision_score(y_true, y_pred)
    elif metric_name == "rocauc":
        if len(np.unique(y_true)) == 1: return 0.5
        return roc_auc_score(y_true, y_pred)

# -----------------------------
# Objective Execution
# -----------------------------
def run_cv_fold(model_params, dataset, task, metric_name, cv_split, device, epochs, optim_name):
    kf = KFold(n_splits=cv_split, shuffle=True)
    scores = []
    
    if task == "graph":
        data_list = dataset
        for train_idx, test_idx in kf.split(data_list):
            train_data = [data_list[i].to(device) for i in train_idx]
            test_data = [data_list[i].to(device) for i in test_idx]
            
            from torch_geometric.loader import DataLoader
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
            
            model = GNN(**model_params).to(device)
            optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=model_params.get("lr", 0.01))
            criterion = torch.nn.L1Loss() if metric_name == "mae" else torch.nn.BCEWithLogitsLoss()
            
            for _ in range(epochs):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(batch.x.float(), batch.edge_index, batch.batch)
                    y = batch.y.view(out.shape).float()
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in test_loader:
                    out = model(batch.x.float(), batch.edge_index, batch.batch)
                    y_true.append(batch.y.view(out.shape))
                    y_pred.append(out)
            
            score = compute_metric(torch.cat(y_true), torch.cat(y_pred), metric_name)
            scores.append(score)
            
    elif task == "node":
        data = dataset.to(device)
        indices = np.arange(data.num_nodes)
        for train_idx, test_idx in kf.split(indices):
            model = GNN(**model_params).to(device)
            optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=model_params.get("lr", 0.01))
            criterion = torch.nn.BCEWithLogitsLoss()
            
            train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
            
            for _ in range(epochs):
                model.train()
                optimizer.zero_grad()
                out = model(data.x.float(), data.edge_index)
                y = data.y.view(-1, 1).float()
                loss = criterion(out[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                out = model(data.x.float(), data.edge_index)
                y = data.y.view(-1, 1).float()
                score = compute_metric(y[test_mask], out[test_mask], metric_name)
                scores.append(score)
                
    return np.mean(scores)

def execute_trials(dataset_name, seed, n_trials, epochs, optimizer_name, cv_split, arch_name, n_reps):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nEvaluating Base Dataset: {dataset_name} | Arch: {arch_name}")
    raw_dataset, task, metric_name, in_c, out_c = get_dataset(dataset_name)
    
    print(f"\nRunning {n_trials} Optuna trials on Base Dataset to find optimal hyperparameters...")
    
    def objective(trial):
        params = {
            "in_channels": in_c,
            "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 128]),
            "out_channels": out_c,
            "num_layers": trial.suggest_int("num_layers", 2, 5),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "arch": arch_name,
            "task": task
        }
        return run_cv_fold(params, raw_dataset, task, metric_name, cv_split, device, epochs, optimizer_name)

    direction = "minimize" if metric_name == "mae" else "maximize"
    study = optuna.create_study(direction=direction)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params.update({"in_channels": in_c, "out_channels": out_c, "arch": arch_name, "task": task})
    print(f"Optimal Hyperparameters: {best_params}")

  
    pipelines = {
        "Base": None,
        "BORF (OR)": BORFRewiring(metric="c_OR", max_iters=3, n_jobs=-1),
        "SDRF (OR)": SDRFRewiring(metric="c_OR", max_iters=5, n_jobs=-1),
        "BORF (Bounds)": BORFRewiring(metric="bounds", max_iters=3, n_jobs=-1),
        "SDRF (Bounds)": SDRFRewiring(metric="bounds", max_iters=5, n_jobs=-1),
    }
    
    cache_data = {}
    time_log = {}
    
    for pipe_name, transform in pipelines.items():
        if transform is None:
            cache_data[pipe_name] = raw_dataset
            time_log[pipe_name] = 0.0
            continue
            
        print(f"Applying {pipe_name} Rewiring...")
        t0 = time.perf_counter()
        
        if task == "node":
            cache_data[pipe_name] = transform(copy.deepcopy(raw_dataset))
        else:
            cache_data[pipe_name] = [transform(copy.deepcopy(g)) for g in raw_dataset]
            
        t1 = time.perf_counter()
        time_log[pipe_name] = t1 - t0
        print(f"[{pipe_name}] Edge Topologies Updated | Exec Time: {t1 - t0:.2f}s")

    # 3. Final Candidate Testing (N Repetitions on Best Model)
    print("\nStarting Empirical Evaluation...")
    final_scores = {p: [] for p in pipelines.keys()}
    
    for rep in range(n_reps):
        print(f"\n--- Repetition {rep + 1}/{n_reps} ---")
        for pipe_name, target_dataset in cache_data.items():
            score = run_cv_fold(best_params, target_dataset, task, metric_name, cv_split, device, epochs, optimizer_name)
            final_scores[pipe_name].append(score)
            print(f"{pipe_name:15s} | {metric_name.upper()}: {score:.4f}")

    print(f"\n{'='*50}\nFINAL AGGREGATE SCORES ({n_reps} Repetitions)\n{'='*50}")
    for pipe_name in pipelines.keys():
        avg = np.mean(final_scores[pipe_name])
        std = np.std(final_scores[pipe_name])
        print(f"{pipe_name:15s} | Avg {metric_name.upper()}: {avg:.4f} ± {std:.4f} | Rewire Time: {time_log[pipe_name]:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Topology Rewiring Benchmarks")
    parser.add_argument("--dataset", type=str, required=True, choices=["ZINC", "Peptides-func", "Peptides-struct", "minesweeper"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials on Base model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--cv-split", type=int, default=5, help="Number of K-Fold splits")
    parser.add_argument("--arch", type=str, default="GCN", choices=["GCN", "GIN"])
    parser.add_argument("--reps", type=int, default=3, help="Number of evaluation repetitions per topology")
    
    args = parser.parse_args()
    
    execute_trials(
        args.dataset, args.seed, args.trials, args.epochs, 
        args.optimizer, args.cv_split, args.arch, args.reps
    )
