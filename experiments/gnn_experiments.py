import os
import time
import argparse
import random
import copy
import numpy as np
import json
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigsh
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool, global_mean_pool
from torch_geometric.datasets import ZINC, LRGBDataset, HeterophilousGraphDataset
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
import optuna

from curvature_rewiring import SDRFRewiring, BORFRewiring, FoSRRewiring


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_topological_metrics(data_list):
    if not isinstance(data_list, list):
        data_list = [data_list]

    diams, avg_sps, fiedlers = [], [], []
    for data in data_list:
        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        dist_matrix = shortest_path(csgraph=A, directed=False, unweighted=True)
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        valid_dists = upper_tri[np.isfinite(upper_tri)]

        if len(valid_dists) > 0:
            diams.append(np.max(valid_dists))
            avg_sps.append(np.mean(valid_dists))
        else:
            diams.append(0.0)
            avg_sps.append(0.0)

        deg = np.array(A.sum(axis=1)).flatten()
        L = sp.diags(deg) - A
        try:
            evals = eigsh(L.astype(float), k=2, which='SA', return_eigenvectors=False)
            fiedlers.append(evals[1])
        except Exception:
            fiedlers.append(0.0)

    return np.mean(diams), np.mean(avg_sps), np.mean(fiedlers)


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

        elif arch == "GAT":
            heads = 4
            head_dim = hidden_channels // heads
            self.layers.append(GATConv(in_channels, head_dim, heads=heads, concat=True))
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_channels, head_dim, heads=heads, concat=True))

            if task == "graph":
                self.layers.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False))
            else:
                self.layers.append(GATConv(hidden_channels, out_channels, heads=1, concat=False))

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


def run_cv_fold(model_params, dataset, task, metric_name, cv_split, epochs, optim_name, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_params = model_params.copy()
    lr = model_params.pop("lr", 0.01)

    kf = KFold(n_splits=cv_split, shuffle=True, random_state=seed)
    scores = []

    if task == "graph":
        for train_idx, test_idx in kf.split(dataset):
            train_data = [dataset[i].to(device) for i in train_idx]
            test_data = [dataset[i].to(device) for i in test_idx]

            from torch_geometric.loader import DataLoader
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

            model = GNN(**model_params).to(device)
            optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=lr)
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
            optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=lr)
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

def execute_trials(dataset_name, seed, n_trials, epochs, optimizer_name, cv_split, arch_name, n_reps, out_dir, max_iters_rewiring, max_workers=4):
    checkpoint_file = os.path.join(out_dir, "gnn_results.json")
    results = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                pass

    run_key = f"{dataset_name}_{arch_name}"
    if run_key not in results:
        results[run_key] = {}

    print(f"\nProcessing Dataset: {dataset_name} | Arch: {arch_name}")
    raw_dataset, task, metric_name, in_c, out_c = get_dataset(dataset_name)

    pipelines = {
        "Base": None,
        "FoSR": FoSRRewiring(max_iters=max_iters_rewiring),
        "BORF (OR)": BORFRewiring(metric="c_OR", max_iters=max_iters_rewiring, n_jobs=-1),
        "SDRF (OR)": SDRFRewiring(metric="c_OR", max_iters=max_iters_rewiring, n_jobs=-1),
        "BORF (Bounds)": BORFRewiring(metric="bounds", max_iters=max_iters_rewiring, n_jobs=-1),
        "SDRF (Bounds)": SDRFRewiring(metric="bounds", max_iters=max_iters_rewiring, n_jobs=-1),
    }

    cache_data = {}
    
    for pipe_name, transform in pipelines.items():
        if pipe_name in results[run_key] and len(results[run_key][pipe_name].get("scores", [])) >= n_reps:
            print(f"Skipping {pipe_name} - already evaluated.")
            continue
            
        print(f"\nPreparing topology: {pipe_name}")
        if transform is None:
            cache_data[pipe_name] = raw_dataset
            rewire_time = 0.0
        else:
            t0 = time.perf_counter()
            if task == "node":
                cache_data[pipe_name] = transform(copy.deepcopy(raw_dataset))
            else:
                cache_data[pipe_name] = [transform(copy.deepcopy(g)) for g in raw_dataset]
            rewire_time = time.perf_counter() - t0

        diam, avg_sp, fiedler = compute_topological_metrics(cache_data[pipe_name])
        topology_metrics = {"Diameter": float(diam), "Avg_SP": float(avg_sp), "Fiedler": float(fiedler)}
        
        print(f"Optuna hyperparameter search for {pipe_name} ({n_trials} trials)...")
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
            return run_cv_fold(params, cache_data[pipe_name], task, metric_name, cv_split, epochs, optimizer_name, seed=seed)
        

        direction = "minimize" if metric_name == "mae" else "maximize"
        study = optuna.create_study(direction=direction)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params.update({"in_channels": in_c, "out_channels": out_c, "arch": arch_name, "task": task})

        print(f"Executing {n_reps} parallel repetitions for {pipe_name}...")
        scores = []
        target_dataset = cache_data[pipe_name]
        
        worker_func = partial(
            run_cv_fold,
            best_params, target_dataset, task, metric_name, cv_split, epochs, optimizer_name
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_rep = {
                executor.submit(worker_func, seed=seed + rep): rep 
                for rep in range(n_reps)
            }
            for future in as_completed(future_to_rep):
                scores.append(future.result())

        avg, std = float(np.mean(scores)), float(np.std(scores))
        
        results[run_key][pipe_name] = {
            "scores": scores,
            "avg": avg,
            "std": std,
            "rewire_time": rewire_time,
            "topology": topology_metrics,
            "best_params": best_params
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Completed {pipe_name} | Avg {metric_name.upper()}: {avg:.4f} ± {std:.4f}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Parallel GNN Topology Rewiring Benchmarks")
    parser.add_argument("--dataset", type=str, required=True, choices=["ZINC", "Peptides-func", "Peptides-struct", "minesweeper"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials per topology")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--cv-split", type=int, default=5, help="Number of K-Fold splits")
    parser.add_argument("--arch", type=str, default="GCN", choices=["GCN", "GIN", "GAT"])
    parser.add_argument("--reps", type=int, default=3, help="Number of parallel evaluation repetitions")
    parser.add_argument("--out-dir", type=str, default="out", help="Directory to save gnn_results.json")
    parser.add_argument("--max-iters-rewiring", type=int, default=5, help="Maximum iterations for all rewiring methods")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker processes")

    args = parser.parse_args()

    execute_trials(
        args.dataset, args.seed, args.trials, args.epochs, 
        args.optimizer, args.cv_split, args.arch, args.reps, 
        args.out_dir, args.max_iters_rewiring, args.workers
    )
