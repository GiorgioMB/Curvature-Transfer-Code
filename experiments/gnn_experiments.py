import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
import socket
import time
import argparse
import random
import copy
import json
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse.linalg import eigsh

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool, global_mean_pool
from torch_geometric.datasets import ZINC, HeterophilousGraphDataset
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score, accuracy_score
import optuna

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tree_neighbors_match import TreeNeighborsMatch
from curvature_rewiring import SDRFRewiring, BORFRewiring, FoSRRewiring


DATASET_META = {
    "ZINC":                ("graph", "mae",    1,  1),
    "minesweeper":         ("node",  "rocauc", 7,  1),
    "QM9":                 ("graph", "mae",    11, 19),
    "ogbg-molhiv":         ("graph", "rocauc", 9,  1),
    "Tree-NeighborsMatch": ("graph", "acc",    35, 16),
}

MASSIVE_DATASETS = {"QM9", "ogbg-molhiv"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Atomic Disk I/O & Distributed Slurm Lock (Safe on Lustre / GPFS / NFS)
# ==============================================================================

def atomic_torch_save(obj, target_path):
    """Writes to a process-unique temp file and atomically renames to target_path."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = f"{target_path}.tmp.{socket.gethostname()}.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target_path)


def atomic_json_save(obj, target_path):
    """Writes JSON atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = f"{target_path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=4)
    os.replace(tmp_path, target_path)


class DistributedCacheLock:
    """
    Process- and cluster-safe lock using atomic `os.mkdir` (POSIX-safe across shared HPC filesystems).
    Includes a heartbeat file so that if a leader Slurm job hits its walltime limit and dies mid-rewiring,
    waiting follower jobs automatically break the stale lock, load the partial checkpoint, and resume!
    """
    def __init__(self, lock_dir, stale_timeout=900):
        self.lock_dir = lock_dir
        self.heartbeat_file = os.path.join(lock_dir, "heartbeat.json")
        self.stale_timeout = stale_timeout
        self.worker_id = f"{socket.gethostname()}_{os.getpid()}_{os.environ.get('SLURM_JOB_ID', 'local')}"

    def try_acquire(self):
        try:
            os.makedirs(self.lock_dir, exist_ok=False)
            self.update_heartbeat()
            return True
        except FileExistsError:
            if self._is_stale():
                print(f"  [Lock] Detected stale lock at {self.lock_dir} (dead Slurm job). Taking over...", flush=True)
                self.force_release()
                try:
                    os.makedirs(self.lock_dir, exist_ok=False)
                    self.update_heartbeat()
                    return True
                except FileExistsError:
                    return False
            return False

    def update_heartbeat(self):
        try:
            tmp_hb = f"{self.heartbeat_file}.tmp.{self.worker_id}"
            with open(tmp_hb, 'w') as f:
                json.dump({"worker": self.worker_id, "timestamp": time.time()}, f)
            os.replace(tmp_hb, self.heartbeat_file)
        except Exception:
            pass

    def _is_stale(self):
        try:
            if not os.path.exists(self.heartbeat_file):
                # Give a newly created lock 30 seconds to write its first heartbeat
                mtime = os.path.getmtime(self.lock_dir)
                return (time.time() - mtime) > 60
            with open(self.heartbeat_file, 'r') as f:
                data = json.load(f)
            return (time.time() - data.get("timestamp", 0)) > self.stale_timeout
        except Exception:
            return False

    def force_release(self):
        shutil.rmtree(self.lock_dir, ignore_errors=True)


# ==============================================================================
# Topological Metrics (Parallelized & Resumable for Massive Datasets)
# ==============================================================================

def _single_graph_topo_metrics(data):
    A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    dist_matrix = shortest_path(csgraph=A, directed=False, unweighted=True)
    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    valid_dists = upper_tri[np.isfinite(upper_tri)]

    if len(valid_dists) > 0:
        diam = float(np.max(valid_dists))
        avg_sp = float(np.mean(valid_dists))
    else:
        diam = 0.0
        avg_sp = 0.0

    n_components, labels = connected_components(csgraph=A, directed=False, return_labels=True)
    lcc_mask = labels == np.argmax(np.bincount(labels))
    A_lcc = A[lcc_mask][:, lcc_mask]

    # Enforce perfect symmetry to prevent eigensolver failure
    A_lcc = A_lcc.maximum(A_lcc.T)

    deg = np.array(A_lcc.sum(axis=1)).flatten()
    deg_inv_sqrt = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg, dtype=float), where=deg > 0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    N = A_lcc.shape[0]
    try:
        if N > 2:
            if N <= 1000:
                L_norm = sp.eye(N) - (D_inv_sqrt @ A_lcc @ D_inv_sqrt)
                evals = spl.eigh(L_norm.toarray(), eigvals_only=True)
                fiedler = float(evals[1])
            else:
                M = D_inv_sqrt @ A_lcc @ D_inv_sqrt
                evals = eigsh(M.astype(float), k=2, which='LA', return_eigenvectors=False)
                fiedler_val = 1.0 - evals[0]
                fiedler = float(max(0.0, fiedler_val))
        else:
            fiedler = 2.0 if N == 2 else 0.0
    except Exception:
        fiedler = 0.0

    return diam, avg_sp, fiedler


def compute_topological_metrics(data_list, max_workers=4, lock=None):
    if not isinstance(data_list, list):
        d, s, f = _single_graph_topo_metrics(data_list)
        return d, s, f

    total = len(data_list)
    if total == 0:
        return 0.0, 0.0, 0.0

    diams, avg_sps, fiedlers = [], [], []
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=max_workers, maxtasksperchild=200) as pool:
        for i, (d, s, f) in enumerate(pool.imap(_single_graph_topo_metrics, data_list, chunksize=100)):
            diams.append(d)
            avg_sps.append(s)
            fiedlers.append(f)
            if lock is not None and (i + 1) % 1000 == 0:
                lock.update_heartbeat()
            if (i + 1) % max(1, total // 10) == 0:
                print(f"  [Topo Metrics] Evaluated {i + 1}/{total} graphs...", flush=True)

    return float(np.mean(diams)), float(np.mean(avg_sps)), float(np.mean(fiedlers))


# ==============================================================================
# Incremental Rewiring Engine (Exact Timing + Partial Saves + Lock Coordination)
# ==============================================================================

def get_or_create_rewired_dataset(
    dataset_name, arch_name, pipe_name, pipe_name_safe, transform,
    raw_dataset_loader, task, max_iters_rewiring, max_workers,
    rewired_cache_path, save_every, rewire_seed
):
    """
    Coordinates safe incremental rewiring across parallel Slurm jobs:
    1. If `rewired_cache_path` exists, loads and returns immediately.
    2. Otherwise acquires `rewired_cache_path + ".lock"`.
       - If another Slurm job holds the lock, waits and polls until the leader finishes
         (or takes over automatically if the leader Slurm job times out!).
    3. Rewires graphs in chunks, saving progress every `save_every` graphs to `partial_path`.
    4. Tracks ONLY pure rewiring compute time in `rewire_time` (pausing the timer during disk writes).
    """
    partial_path = rewired_cache_path + ".partial.pt"
    lock_dir = rewired_cache_path + ".lock"
    lock = DistributedCacheLock(lock_dir, stale_timeout=900)

    while True:
        # 1. Fast path: Completed cache already exists on disk
        if os.path.exists(rewired_cache_path):
            print(f"\n[Cache Hit] Loading completed topology for {pipe_name} from {rewired_cache_path}...", flush=True)
            bundle = torch.load(rewired_cache_path, weights_only=False)
            return bundle["dataset"], bundle["rewire_time"], bundle["topology_metrics"]

        # 2. Try to become the active rewiring leader for this cache target
        if lock.try_acquire():
            try:
                # Double-check after acquiring lock in case it finished right before lock acquisition
                if os.path.exists(rewired_cache_path):
                    bundle = torch.load(rewired_cache_path, weights_only=False)
                    return bundle["dataset"], bundle["rewire_time"], bundle["topology_metrics"]

                raw_dataset = raw_dataset_loader()
                set_seed(rewire_seed)

                # Check if a previous interrupted run left a partial checkpoint
                rewired_list = []
                elapsed_rewire_time = 0.0
                start_idx = 0

                if task == "graph" and transform is not None and os.path.exists(partial_path):
                    try:
                        part_bundle = torch.load(partial_path, weights_only=False)
                        rewired_list = part_bundle["rewired_list"]
                        elapsed_rewire_time = float(part_bundle["elapsed_rewire_time"])
                        start_idx = len(rewired_list)
                        print(
                            f"\n[Partial Resume] Resuming {pipe_name} from graph {start_idx}/{len(raw_dataset)} "
                            f"(accumulated rewire_time so far: {elapsed_rewire_time:.2f}s)",
                            flush=True
                        )
                    except Exception as e:
                        print(f"  [Warning] Could not load partial checkpoint ({e}). Restarting from 0.", flush=True)
                        rewired_list = []
                        elapsed_rewire_time = 0.0
                        start_idx = 0

                # --- Perform Rewiring ---
                if transform is None:
                    print(f"\nPreparing topology: {pipe_name} (no rewiring needed)", flush=True)
                    target_dataset = raw_dataset
                    rewire_time = 0.0
                elif task == "node":
                    print(f"\nPreparing node topology: {pipe_name}", flush=True)
                    t0 = time.perf_counter()
                    target_dataset = transform(copy.deepcopy(raw_dataset))
                    rewire_time = time.perf_counter() - t0
                else:
                    total_graphs = len(raw_dataset)
                    if start_idx < total_graphs:
                        print(
                            f"\nRewiring topology: {pipe_name} | Range: [{start_idx} -> {total_graphs}] "
                            f"| Saving partial checkpoints every {save_every} graphs...",
                            flush=True
                        )
                        ctx = mp.get_context('spawn')

                        def _remaining_graph_gen():
                            for idx in range(start_idx, total_graphs):
                                g = raw_dataset[idx]
                                yield g.clone() if hasattr(g, 'clone') else copy.deepcopy(g)

                        with ctx.Pool(processes=max_workers, maxtasksperchild=50) as pool:
                            # Start pure-compute stopwatch
                            t_chunk_start = time.perf_counter()

                            for offset, out_g in enumerate(pool.imap(transform, _remaining_graph_gen(), chunksize=25)):
                                rewired_list.append(out_g.clone())
                                current_count = start_idx + offset + 1

                                # Log progress every 5% or at save boundaries
                                if current_count % max(1, total_graphs // 20) == 0 or current_count == total_graphs:
                                    current_elapsed = elapsed_rewire_time + (time.perf_counter() - t_chunk_start)
                                    print(
                                        f"  [Topology] Rewired {current_count}/{total_graphs} graphs "
                                        f"({100.0 * current_count / total_graphs:.1f}%) | Pure Rewire Time: {current_elapsed:.2f}s",
                                        flush=True
                                    )

                                # Periodic Partial Checkpoint (PAUSE stopwatch during disk I/O!)
                                if current_count % save_every == 0 and current_count < total_graphs:
                                    elapsed_rewire_time += (time.perf_counter() - t_chunk_start)
                                    print(
                                        f"  [Partial Save] Writing {current_count}/{total_graphs} graphs to {partial_path} "
                                        f"(timer paused at {elapsed_rewire_time:.2f}s)...",
                                        flush=True
                                    )
                                    atomic_torch_save({
                                        "rewired_list": rewired_list,
                                        "elapsed_rewire_time": elapsed_rewire_time,
                                        "total_graphs": total_graphs
                                    }, partial_path)
                                    lock.update_heartbeat()
                                    # Resume stopwatch after disk write completes
                                    t_chunk_start = time.perf_counter()

                            # Add final segment of compute time
                            elapsed_rewire_time += (time.perf_counter() - t_chunk_start)

                    target_dataset = rewired_list
                    rewire_time = elapsed_rewire_time

                # --- Compute Topological Metrics ---
                print(
                    f"  [Topology] Rewiring complete in {rewire_time:.2f}s. Computing spectral/topological metrics...",
                    flush=True
                )
                lock.update_heartbeat()
                diam, avg_sp, fiedler = compute_topological_metrics(target_dataset, max_workers=max_workers, lock=lock)
                topology_metrics = {"Diameter": float(diam), "Avg_SP": float(avg_sp), "Fiedler": float(fiedler)}

                # --- Save Final Bundle & Clean Up Partial Checkpoint ---
                atomic_torch_save({
                    "dataset": target_dataset,
                    "rewire_time": float(rewire_time),
                    "topology_metrics": topology_metrics,
                }, rewired_cache_path)

                if os.path.exists(partial_path):
                    try:
                        os.remove(partial_path)
                    except OSError:
                        pass

                print(f"  [Topology] Saved final rewired cache to {rewired_cache_path}.", flush=True)
                return target_dataset, float(rewire_time), topology_metrics

            finally:
                lock.force_release()

        else:
            # Another parallel Slurm job is actively rewiring this exact cache file
            print(
                f"  [Lock Wait] Another Slurm job is currently rewiring {rewired_cache_path}. "
                f"Waiting 30s (will automatically resume if that job times out)...",
                flush=True
            )
            time.sleep(30)


# ==============================================================================
# GNN Architecture & Cross-Validation Evaluation
# ==============================================================================

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
            if x.size(1) == 35:
                x = x[x[:, -1] == 1.0]
            else:
                x = self.pool(x, batch)
            x = self.lin(x)
        return x


def get_dataset(name):
    if name == "ZINC":
        d_train = ZINC(root="data/ZINC", split="train", subset=True)
        d_val = ZINC(root="data/ZINC", split="val", subset=True)
        d_test = ZINC(root="data/ZINC", split="test", subset=True)
        return list(d_train) + list(d_val) + list(d_test), "graph", "mae", 1, 1
    elif name == "minesweeper":
        dataset = HeterophilousGraphDataset(root="data/Heterophilous", name="minesweeper")
        return dataset[0], "node", "rocauc", dataset.num_node_features, 1
    elif name == "QM9":
        from torch_geometric.datasets import QM9
        dataset = QM9(root="data/QM9")
        return list(dataset), "graph", "mae", dataset.num_node_features, 19
    elif name == "ogbg-molhiv":
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="data/OGB")
        return list(dataset), "graph", "rocauc", dataset.num_node_features, 1
    elif name == "Tree-NeighborsMatch":
        dataset = TreeNeighborsMatch(root="data/Tree-NeighborsMatch", depth=4, num_graphs=16000)
        num_classes = 2 ** dataset.depth
        return list(dataset), "graph", "acc", dataset.num_node_features, num_classes

    raise ValueError(f"Unknown dataset: {name}")


def compute_metric(y_true, y_pred, metric_name):
    y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    if metric_name == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric_name == "ap":
        valid = ~np.isnan(y_true)
        if not np.any(valid):
            return 0.0
        y_true[np.isnan(y_true)] = 0
        return average_precision_score(y_true, y_pred)
    elif metric_name == "rocauc":
        if len(np.unique(y_true)) == 1:
            return 0.5
        return roc_auc_score(y_true, y_pred)
    elif metric_name == "acc":
        y_pred_classes = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred_classes)


def load_dataset_from_source(dataset_or_path):
    if isinstance(dataset_or_path, str):
        loaded = torch.load(dataset_or_path, weights_only=False)
        if isinstance(loaded, dict) and "dataset" in loaded:
            return loaded["dataset"]
        return loaded
    return dataset_or_path


def run_cv_fold(model_params, dataset_or_path, task, metric_name, cv_split, epochs, optim_name, seed, num_splits_to_run=None, device_id=None):
    set_seed(seed)
    if device_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_pin_memory = (device.type == 'cuda')
    dataset = load_dataset_from_source(dataset_or_path)

    model_params = model_params.copy()
    lr = model_params.pop("lr", 0.01)

    kf = KFold(n_splits=cv_split, shuffle=True, random_state=seed)
    scores = []

    if task == "graph":
        from torch_geometric.loader import DataLoader
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            train_data = [dataset[i] for i in train_idx]
            test_data = [dataset[i] for i in test_idx]

            train_loader = DataLoader(train_data, batch_size=512, shuffle=True, pin_memory=use_pin_memory, num_workers=0, persistent_workers=False)
            test_loader = DataLoader(test_data, batch_size=512, shuffle=False, pin_memory=use_pin_memory, num_workers=0, persistent_workers=False)

            model = GNN(**model_params).to(device)
            optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=lr)

            if metric_name == "mae":
                criterion = torch.nn.L1Loss()
            elif metric_name == "acc":
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.BCEWithLogitsLoss()

            for epoch in range(epochs):
                if (epoch + 1) % 10 == 0:
                    print(f"  [Fold] Graph Task Epoch {epoch + 1}/{epochs} complete.", flush=True)
                model.train()
                for batch in train_loader:
                    batch = batch.to(device, non_blocking=use_pin_memory)
                    optimizer.zero_grad()
                    out = model(batch.x.float(), batch.edge_index, batch.batch)

                    if metric_name == "acc":
                        y = batch.y.long().view(-1)
                    else:
                        y = batch.y.view(out.shape).float()

                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x.float(), batch.edge_index, batch.batch)
                    if metric_name == "acc":
                        y_true.append(batch.y.view(-1))
                    else:
                        y_true.append(batch.y.view(out.shape))
                    y_pred.append(out)

            score = compute_metric(torch.cat(y_true), torch.cat(y_pred), metric_name)
            scores.append(score)
            if num_splits_to_run is not None and (fold_idx + 1) >= num_splits_to_run:
                break

    elif task == "node":
        data = dataset.to(device)
        indices = np.arange(data.num_nodes)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
            model = GNN(**model_params).to(device)
            optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=lr)
            criterion = torch.nn.BCEWithLogitsLoss()

            train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            train_mask[train_idx] = True
            test_mask[test_idx] = True

            for epoch in range(epochs):
                if (epoch + 1) % 50 == 0:
                    print(f"  [Fold] Node Task Epoch {epoch + 1}/{epochs} complete.", flush=True)
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
                if num_splits_to_run is not None and (fold_idx + 1) >= num_splits_to_run:
                    break

    return float(np.mean(scores))


def global_worker_rep_wrapper(rep, num_gpus, best_params, cache_path, task, metric_name, cv_split, epochs, optimizer_name, seed):
    gpu_id = rep % num_gpus if num_gpus > 0 else None
    score = run_cv_fold(
        best_params, cache_path, task, metric_name, cv_split,
        epochs, optimizer_name, seed=seed + rep, device_id=gpu_id
    )
    return rep, score


def resolve_rewired_cache_path(dataset_name, arch_name, pipe_name_safe, max_iters_rewiring, out_dir, cache_scope, shared_cache_dir):
    use_shared = (cache_scope == "shared") or (cache_scope == "auto" and dataset_name in MASSIVE_DATASETS)
    if use_shared:
        os.makedirs(shared_cache_dir, exist_ok=True)
        filename = f"rewired_{dataset_name}_{pipe_name_safe}_iters{max_iters_rewiring}.pt"
        return os.path.join(shared_cache_dir, filename)
    else:
        os.makedirs(out_dir, exist_ok=True)
        filename = f"rewired_{dataset_name}_{arch_name}_{pipe_name_safe}_iters{max_iters_rewiring}.pt"
        return os.path.join(out_dir, filename)


def execute_trials(
    dataset_name, seed, n_trials, epochs, optimizer_name, cv_split,
    arch_name, n_reps, out_dir, max_iters_rewiring, max_workers=4,
    pipelines_to_run=None, cache_scope="auto", shared_cache_dir="data/rewired_cache",
    save_every=2000
):
    os.makedirs(out_dir, exist_ok=True)
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

    print(f"\nProcessing Dataset: {dataset_name} | Arch: {arch_name}", flush=True)

    raw_dataset_cache = {"data": None}
    if dataset_name in DATASET_META:
        task, metric_name, in_c, out_c = DATASET_META[dataset_name]
    else:
        set_seed(seed)
        raw_data, task, metric_name, in_c, out_c = get_dataset(dataset_name)
        raw_dataset_cache["data"] = raw_data

    def _lazy_load_raw():
        if raw_dataset_cache["data"] is None:
            set_seed(seed)
            raw_data, _, _, _, _ = get_dataset(dataset_name)
            raw_dataset_cache["data"] = raw_data
        return raw_dataset_cache["data"]

    pipelines = {
        "Base": None,
        "FoSR": FoSRRewiring(max_iters=max_iters_rewiring),
        "BORF (OR)": BORFRewiring(metric="c_OR", max_iters=max_iters_rewiring, n_jobs=1),
        "SDRF (OR)": SDRFRewiring(metric="c_OR", max_iters=max_iters_rewiring, n_jobs=1),
        "SDRF (BF)": SDRFRewiring(metric="c_BF", max_iters=max_iters_rewiring, n_jobs=1),
        "BORF (Bounds)": BORFRewiring(metric="bounds", max_iters=max_iters_rewiring, n_jobs=1),
        "SDRF (Bounds)": SDRFRewiring(metric="bounds", max_iters=max_iters_rewiring, n_jobs=1),
        "BORF (JL)": BORFRewiring(metric="jl_bounds", max_iters=max_iters_rewiring, n_jobs=1),
    }
    if pipelines_to_run:
        pipelines = {k: v for k, v in pipelines.items() if k in pipelines_to_run}

    for pipe_name, transform in pipelines.items():
        existing_entry = results[run_key].get(pipe_name, {})
        existing_scores = existing_entry.get("scores", [])

        if len(existing_scores) >= n_reps:
            print(f"Skipping {pipe_name}, already evaluated ({len(existing_scores)}/{n_reps} reps).", flush=True)
            continue

        pipe_name_safe = pipe_name.replace(" ", "_").replace("(", "").replace(")", "")
        rewired_cache_path = resolve_rewired_cache_path(
            dataset_name, arch_name, pipe_name_safe, max_iters_rewiring,
            out_dir, cache_scope, shared_cache_dir
        )

        use_shared = (cache_scope == "shared") or (cache_scope == "auto" and dataset_name in MASSIVE_DATASETS)
        # Deterministic seed so restarted partial runs generate the exact same rewiring trajectory
        if use_shared:
            rewire_seed = seed + sum(ord(c) for c in f"{dataset_name}_{pipe_name}")
        else:
            rewire_seed = seed + sum(ord(c) for c in f"{dataset_name}_{arch_name}_{pipe_name}")

        # ------------------------------------------------------------------
        # STEP 1: Incremental / Lock-Coordinated Rewiring & Metric Caching
        # ------------------------------------------------------------------
        target_dataset, rewire_time, topology_metrics = get_or_create_rewired_dataset(
            dataset_name=dataset_name,
            arch_name=arch_name,
            pipe_name=pipe_name,
            pipe_name_safe=pipe_name_safe,
            transform=transform,
            raw_dataset_loader=_lazy_load_raw,
            task=task,
            max_iters_rewiring=max_iters_rewiring,
            max_workers=max_workers,
            rewired_cache_path=rewired_cache_path,
            save_every=save_every,
            rewire_seed=rewire_seed
        )

        # Save topology metrics and rewire_time into gnn_results.json right away,
        # even before Optuna or repetitions start!
        if pipe_name not in results[run_key]:
            results[run_key][pipe_name] = {
                "scores": [],
                "avg": None,
                "std": None,
                "rewire_time": rewire_time,
                "topology": topology_metrics,
                "best_params": None
            }
            atomic_json_save(results, checkpoint_file)

        # ------------------------------------------------------------------
        # STEP 2: Optuna Hyperparameter Search (Resumable per Trial via Journal/JSON)
        # ------------------------------------------------------------------
        params_file = os.path.join(out_dir, f"optuna_params_{dataset_name}_{arch_name}_{pipe_name_safe}.json")

        if os.path.exists(params_file):
            print(f"  [Optuna] Found cached hyperparameters at {params_file}. Skipping search.", flush=True)
            with open(params_file, 'r') as f:
                best_params = json.load(f)
        else:
            print(f"Optuna hyperparameter search for {pipe_name} ({n_trials} trials)...", flush=True)

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
                return run_cv_fold(
                    params, target_dataset, task,
                    metric_name, cv_split, epochs,
                    optimizer_name, seed=seed,
                    num_splits_to_run=(cv_split // 2) + 1
                )

            direction = "minimize" if metric_name == "mae" else "maximize"
            study = optuna.create_study(direction=direction)
            optuna.logging.set_verbosity(optuna.logging.INFO)
            study.optimize(objective, n_trials=n_trials, n_jobs=min(4, max_workers))

            best_params = study.best_params
            best_params.update({"in_channels": in_c, "out_channels": out_c, "arch": arch_name, "task": task})

            atomic_json_save(best_params, params_file)
            print(f"  [Optuna] Saved best hyperparameters to {params_file}.", flush=True)

        del target_dataset

        # ------------------------------------------------------------------
        # STEP 3: Parallel Evaluation Repetitions (Incremental Save per Rep)
        # ------------------------------------------------------------------
        start_rep = len(existing_scores)
        remaining_reps = list(range(start_rep, n_reps))
        print(
            f"Executing {len(remaining_reps)} remaining repetitions (from rep {start_rep} to {n_reps - 1}) for {pipe_name}...",
            flush=True
        )

        scores = list(existing_scores)
        num_gpus = torch.cuda.device_count()
        safe_workers = min(max_workers, num_gpus * 2) if num_gpus > 0 else max_workers

        worker_fn = partial(
            global_worker_rep_wrapper,
            num_gpus=num_gpus,
            best_params=best_params,
            cache_path=rewired_cache_path,
            task=task,
            metric_name=metric_name,
            cv_split=cv_split,
            epochs=epochs,
            optimizer_name=optimizer_name,
            seed=seed
        )

        with ProcessPoolExecutor(max_workers=safe_workers) as executor:
            futures = [executor.submit(worker_fn, rep) for rep in remaining_reps]
            for fut in as_completed(futures):
                rep_idx, rep_score = fut.result()
                scores.append(rep_score)
                avg, std = float(np.mean(scores)), float(np.std(scores))

                results[run_key][pipe_name] = {
                    "scores": scores,
                    "avg": avg,
                    "std": std,
                    "rewire_time": rewire_time,
                    "topology": topology_metrics,
                    "best_params": best_params
                }
                atomic_json_save(results, checkpoint_file)
                print(
                    f"  [Rep {len(scores)}/{n_reps}] Completed rep {rep_idx} ({rep_score:.4f}) "
                    f"| Running Avg: {avg:.4f} ± {std:.4f} | Rewire Time: {rewire_time:.2f}s",
                    flush=True
                )

        avg, std = float(np.mean(scores)), float(np.std(scores))
        print(
            f"Completed {pipe_name} | Final Avg {metric_name.upper()}: {avg:.4f} ± {std:.4f} | Rewire Time: {rewire_time:.2f}s",
            flush=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel GNN Topology Rewiring Benchmarks")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["ZINC", "QM9", "minesweeper", "ogbg-molhiv", "Tree-NeighborsMatch"]
    )
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
    parser.add_argument(
        "--pipelines", type=str, nargs="*", default=None,
        help="Optional subset of pipelines to run (e.g. --pipelines 'Base' 'FoSR')"
    )
    parser.add_argument(
        "--cache-scope", type=str, default="auto", choices=["auto", "arch", "shared"],
        help=(
            "'auto' (default): shares rewired graphs across architectures for massive datasets (QM9, ogbg-molhiv) "
            "and rewires per-(dataset, arch, method) for smaller datasets. "
            "'arch': always rewire per (dataset, arch, method) inside --out-dir. "
            "'shared': always share rewired graphs across architectures in --shared-cache-dir."
        )
    )
    parser.add_argument(
        "--shared-cache-dir", type=str, default="data/rewired_cache",
        help="Directory to store shared rewired datasets (used for QM9 and ogbg-molhiv in 'auto' or 'shared' mode)"
    )
    parser.add_argument(
        "--save-every", type=int, default=2000,
        help="Number of rewired graphs between partial disk checkpoints (default: 2000)"
    )

    args = parser.parse_args()

    execute_trials(
        dataset_name=args.dataset,
        seed=args.seed,
        n_trials=args.trials,
        epochs=args.epochs,
        optimizer_name=args.optimizer,
        cv_split=args.cv_split,
        arch_name=args.arch,
        n_reps=args.reps,
        out_dir=args.out_dir,
        max_iters_rewiring=args.max_iters_rewiring,
        max_workers=args.workers,
        pipelines_to_run=args.pipelines,
        cache_scope=args.cache_scope,
        shared_cache_dir=args.shared_cache_dir,
        save_every=args.save_every
    )
