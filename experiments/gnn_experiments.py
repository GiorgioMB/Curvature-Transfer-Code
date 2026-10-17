"""Rewiring case study (Appendix A): GNNs on original and rewired graphs.

For one (dataset, architecture) this runs every rewiring pipeline, tunes the
GNN hyperparameters of each pipeline with Optuna, and evaluates the selected
configuration by repeated K-fold cross-validation.

Everything is checkpointed as soon as it is computed:

  rewiring       graph tasks: the rewired graphs are saved every --save-every
                 graphs or --save-seconds seconds (``*.partial.pt``); node tasks:
                 the rewiring loop saves its state after every iteration
                 (``*.iter.pkl``). Only pure rewiring compute time is counted.
                 The rewired dataset is saved (``*.rewired.pt``) before the
                 topological metrics are computed.
  topology       per-graph metrics are saved incrementally (``*.topo.pkl``).
  Optuna         every trial is stored in a journal file (``optuna/*.journal``)
                 as soon as it finishes; a restart runs only the missing trials.
                 A trial interrupted mid-way is re-enqueued with the same
                 hyperparameters, so it reuses its finished folds.
  folds          every finished CV fold (tuning and evaluation) is saved
                 (``fold_cache/``) under a key that hashes the dataset,
                 pipeline, hyperparameters, seed and fold index;
  epochs         a fold that is still training saves model, optimizer, epoch,
                 and RNG states every --ckpt-seconds seconds (``train_ckpt/``)
                 and resumes from them;
  repetitions    every finished repetition is written to the results file.

Every fold is seeded from (seed, fold index), so a fold gives the same result
whether or not the folds before it were restored from the cache.

Results of this (dataset, arch) go to ``<out>/parts/<dataset>_<arch>.json``
(one writer per file, so parallel Slurm jobs on different datasets never
overwrite each other); ``<out>/gnn_results.json`` is the merge of all parts
(see ``merge_results.py``) and is what ``stats.py`` and the paper read.

Assumption: one process at a time works on a given (dataset, arch, pipeline);
this is enforced with a lock directory with a heartbeat.
"""
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
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
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
from torch_geometric.nn import GCNConv, GINEConv, GATConv, global_mean_pool
from torch_geometric.datasets import ZINC, HeterophilousGraphDataset
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score, accuracy_score
import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from tree_neighbors_match import TreeNeighborsMatch  # noqa: E402
from curvature_transfer import _io  # noqa: E402
from curvature_transfer.pyg import SDRFRewiring, BORFRewiring, FoSRRewiring, mark_original  # noqa: E402
import merge_results  # noqa: E402

# name: (task, metric, in_channels, out_channels)
DATASET_META = {
    "ZINC":                ("graph", "mae",    1,  1),
    "minesweeper":         ("node",  "rocauc", 7,  1),
    "QM9":                 ("graph", "mae",    11, 19),
    "ogbg-molhiv":         ("graph", "rocauc", 9,  1),
    "Tree-NeighborsMatch": ("graph", "acc",    35, 16),
}
# Tree-NeighborsMatch is read out at the root node (Alon & Yahav, 2021);
# the root carries a 1 in the last input feature.
ROOT_READOUT = {"Tree-NeighborsMatch"}
MASSIVE_DATASETS = {"QM9", "ogbg-molhiv"}
# Pipeline names as used by stats.py and paper_artifacts.py.
PIPELINES = ["Original", "FoSR", "BORF (OR)", "SDRF (OR)", "SDRF (BF)", "BORF (Bounds)", "SDRF (Bounds)",
             "BORF (JL)", "BORF (Sharpened)"]
# Evaluation repetitions use seeds offset from the tuning seed, so that no
# repetition reuses the K-fold partition on which Optuna selected hyperparameters.
EVAL_SEED_OFFSET = 1000


def set_seed(seed):
    seed = int(seed) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fold_seed(seed, fold_idx):
    return (int(seed) * 1000003 + 7919 * (int(fold_idx) + 1)) % (2 ** 31 - 1)


def safe_name(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "")


# ==============================================================================
# Atomic Disk I/O & Distributed Lock (Safe on Lustre / GPFS / NFS)
# ==============================================================================

def atomic_torch_save(obj, target_path):
    """Writes to a process-unique temp file and atomically renames to target_path."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = _io.tmp_path(target_path)
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target_path)


def torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except (FileNotFoundError, EOFError, RuntimeError, OSError) as e:
        if os.path.exists(path):
            print(f"  [Warning] unreadable checkpoint {path} ({e}); ignoring it", flush=True)
        return None


class DistributedCacheLock:
    """Process- and cluster-safe lock using atomic ``os.mkdir`` with a heartbeat.

    While held, a background thread refreshes the heartbeat every
    ``beat_every`` seconds; a lock whose heartbeat is older than
    ``stale_timeout`` belongs to a dead job and is taken over.
    """

    def __init__(self, lock_dir, stale_timeout=900, beat_every=60):
        self.lock_dir = lock_dir
        self.heartbeat_file = os.path.join(lock_dir, "heartbeat.json")
        self.stale_timeout = stale_timeout
        self.beat_every = beat_every
        self.worker_id = f"{socket.gethostname()}_{os.getpid()}_{os.environ.get('SLURM_JOB_ID', 'local')}"
        self._stop = None

    def try_acquire(self):
        try:
            os.makedirs(self.lock_dir, exist_ok=False)
            self.update_heartbeat()
            return True
        except FileExistsError:
            if self._is_stale():
                print(f"  [Lock] Detected stale lock at {self.lock_dir} (dead job). Taking over...", flush=True)
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
            _io.atomic_json({"worker": self.worker_id, "timestamp": time.time()}, self.heartbeat_file)
        except Exception:
            pass

    def _is_stale(self):
        try:
            if not os.path.exists(self.heartbeat_file):
                return (time.time() - os.path.getmtime(self.lock_dir)) > 60
            data = _io.read_json(self.heartbeat_file, {})
            return (time.time() - data.get("timestamp", 0)) > self.stale_timeout
        except Exception:
            return False

    def _beat(self):
        while not self._stop.wait(self.beat_every):
            self.update_heartbeat()

    def start_heartbeat(self):
        self._stop = threading.Event()
        threading.Thread(target=self._beat, daemon=True).start()

    def force_release(self):
        if self._stop is not None:
            self._stop.set()
        shutil.rmtree(self.lock_dir, ignore_errors=True)


@contextmanager
def held(lock_dir, what):
    """Hold ``lock_dir`` for the duration of the block (waits for a live holder)."""
    lock = DistributedCacheLock(lock_dir)
    while not lock.try_acquire():
        print(f"  [Lock Wait] another job holds {what}; waiting 30s (taken over if that job dies)...", flush=True)
        time.sleep(30)
    lock.start_heartbeat()
    try:
        yield lock
    finally:
        lock.force_release()


# ==============================================================================
# Topological Metrics (incremental)
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
    A_lcc = A_lcc.maximum(A_lcc.T)  # perfect symmetry for the eigensolver
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
                fiedler = float(max(0.0, 1.0 - evals[0]))
        else:
            fiedler = 2.0 if N == 2 else 0.0
    except Exception:
        fiedler = 0.0
    return diam, avg_sp, fiedler


def compute_topological_metrics(data_list, max_workers=4, partial_path=None, save_every=1000, save_seconds=300):
    """Mean diameter, mean average shortest path, mean Fiedler value. For a
    list of graphs, per-graph values are saved to ``partial_path`` every
    ``save_every`` graphs or ``save_seconds`` seconds and reused on restart."""
    if not isinstance(data_list, list):
        return _single_graph_topo_metrics(data_list)
    total = len(data_list)
    if total == 0:
        return 0.0, 0.0, 0.0
    vals = []
    if partial_path:
        st = _io.read_pickle(partial_path)
        if isinstance(st, dict) and st.get("total") == total:
            vals = list(st["vals"])
            print(f"  [Topo Metrics] resuming at {len(vals)}/{total} graphs", flush=True)
    if len(vals) < total:
        ctx = mp.get_context('spawn')
        last = time.time()
        with ctx.Pool(processes=max_workers, maxtasksperchild=200) as pool:
            for v in pool.imap(_single_graph_topo_metrics, data_list[len(vals):], chunksize=100):
                vals.append(v)
                i = len(vals)
                if partial_path and (i % save_every == 0 or time.time() - last > save_seconds) and i < total:
                    _io.atomic_pickle({"total": total, "vals": vals}, partial_path)
                    last = time.time()
                if i % max(1, total // 10) == 0:
                    print(f"  [Topo Metrics] Evaluated {i}/{total} graphs...", flush=True)
    a = np.asarray(vals, float)
    return float(a[:, 0].mean()), float(a[:, 1].mean()), float(a[:, 2].mean())


# ==============================================================================
# Incremental rewiring (exact compute timing, partial saves, lock coordination)
# ==============================================================================

def get_or_create_rewired_dataset(dataset_name, pipe_name, transform, raw_dataset_loader, task, max_workers,
                                  rewired_cache_path, save_every, save_seconds, rewire_seed):
    """Return (dataset, rewire_time, topology_metrics), computing and caching
    them resumably (see the module docstring). Every graph of the returned
    dataset has ``edge_attr`` = added-edge flag (0 for original edges)."""
    final = rewired_cache_path
    rewired_path = rewired_cache_path + ".rewired.pt"
    partial_path = rewired_cache_path + ".partial.pt"
    iter_path = rewired_cache_path + ".iter.pkl"
    topo_path = rewired_cache_path + ".topo.pkl"

    b = torch_load(final) if os.path.exists(final) else None
    if b is not None:
        print(f"\n[Cache Hit] {pipe_name}: {final}", flush=True)
        return b["dataset"], b["rewire_time"], b["topology_metrics"]

    with held(rewired_cache_path + ".lock", final):
        b = torch_load(final) if os.path.exists(final) else None
        if b is not None:
            return b["dataset"], b["rewire_time"], b["topology_metrics"]

        b = torch_load(rewired_path) if os.path.exists(rewired_path) else None
        if b is not None:
            target_dataset, rewire_time = b["dataset"], float(b["rewire_time"])
            print(f"\n[Resume] {pipe_name}: rewiring already done ({rewire_time:.1f}s); computing metrics", flush=True)
        else:
            raw_dataset = raw_dataset_loader()
            set_seed(rewire_seed)
            if transform is None:
                print(f"\nPreparing topology: {pipe_name} (no rewiring)", flush=True)
                if task == "node":
                    target_dataset = mark_original(raw_dataset.clone())
                else:
                    target_dataset = [mark_original(g.clone()) for g in raw_dataset]
                rewire_time = 0.0
            elif task == "node":
                print(f"\nRewiring node graph: {pipe_name} (state saved after every iteration)", flush=True)
                transform.checkpoint_path = iter_path
                t0 = time.perf_counter()
                target_dataset = transform(copy.deepcopy(raw_dataset))
                wall = time.perf_counter() - t0
                rewire_time = float(transform.last_compute_time if transform.last_compute_time is not None else wall)
                transform.checkpoint_path = None
            else:
                target_dataset, rewire_time = _rewire_graph_list(raw_dataset, transform, pipe_name, max_workers,
                                                                 partial_path, save_every, save_seconds)
            atomic_torch_save({"dataset": target_dataset, "rewire_time": float(rewire_time)}, rewired_path)
            for p in (partial_path, iter_path):
                if os.path.exists(p):
                    os.remove(p)

        print(f"  [Topology] Rewiring done in {rewire_time:.2f}s. Computing topological metrics...", flush=True)
        diam, avg_sp, fiedler = compute_topological_metrics(target_dataset, max_workers=max_workers,
                                                            partial_path=topo_path)
        topology_metrics = {"Diameter": float(diam), "Avg_SP": float(avg_sp), "Fiedler": float(fiedler)}
        atomic_torch_save({"dataset": target_dataset, "rewire_time": float(rewire_time),
                           "topology_metrics": topology_metrics}, final)
        for p in (rewired_path, topo_path):
            if os.path.exists(p):
                os.remove(p)
        print(f"  [Topology] Saved {final}", flush=True)
        return target_dataset, float(rewire_time), topology_metrics


def _rewire_graph_list(raw_dataset, transform, pipe_name, max_workers, partial_path, save_every, save_seconds):
    rewired_list, elapsed = [], 0.0
    total = len(raw_dataset)
    part = torch_load(partial_path) if os.path.exists(partial_path) else None
    if part is not None and part.get("total_graphs") == total:
        rewired_list = part["rewired_list"]
        elapsed = float(part["elapsed_rewire_time"])
        print(f"\n[Partial Resume] {pipe_name}: graph {len(rewired_list)}/{total} "
              f"(rewire time so far {elapsed:.2f}s)", flush=True)
    start = len(rewired_list)
    if start >= total:
        return rewired_list, elapsed
    print(f"\nRewiring {pipe_name}: graphs [{start} -> {total}], saving every {save_every} graphs "
          f"or {save_seconds}s", flush=True)

    def _gen():
        for idx in range(start, total):
            g = raw_dataset[idx]
            yield g.clone() if hasattr(g, 'clone') else copy.deepcopy(g)

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=max_workers, maxtasksperchild=50) as pool:
        t0 = time.perf_counter()
        last_save = time.time()
        for out_g in pool.imap(transform, _gen(), chunksize=25):
            rewired_list.append(out_g.clone())
            cur = len(rewired_list)
            if cur % max(1, total // 20) == 0 or cur == total:
                print(f"  [Topology] Rewired {cur}/{total} ({100.0 * cur / total:.1f}%) | "
                      f"rewire time {elapsed + time.perf_counter() - t0:.2f}s", flush=True)
            if cur < total and (cur % save_every == 0 or time.time() - last_save > save_seconds):
                elapsed += time.perf_counter() - t0  # pause the stopwatch during I/O
                atomic_torch_save({"rewired_list": rewired_list, "elapsed_rewire_time": elapsed,
                                   "total_graphs": total}, partial_path)
                last_save = time.time()
                t0 = time.perf_counter()
        elapsed += time.perf_counter() - t0
    return rewired_list, elapsed


# ==============================================================================
# GNN architecture
# ==============================================================================

class GNN(torch.nn.Module):
    """GCN (GCNConv; ignores edge features), GIN (GINEConv) and GAT (GATConv)
    with a 1-dimensional edge feature (1 = edge added by rewiring)."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, arch="GCN", task="graph",
                 dropout=0.5, readout="mean"):
        super().__init__()
        self.task, self.arch, self.dropout, self.readout = task, arch, dropout, readout
        self.layers = torch.nn.ModuleList()
        last = hidden_channels if task == "graph" else out_channels
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [last]
        for a, b in zip(dims[:-1], dims[1:]):
            if arch == "GCN":
                self.layers.append(GCNConv(a, b))
            elif arch == "GIN":
                mlp = torch.nn.Sequential(torch.nn.Linear(a, hidden_channels), torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_channels, b))
                self.layers.append(GINEConv(mlp, edge_dim=1))
            elif arch == "GAT":
                is_last = b == last and len(self.layers) == num_layers - 1
                if is_last:
                    self.layers.append(GATConv(a, b, heads=1, concat=False, edge_dim=1))
                else:
                    heads = 4
                    self.layers.append(GATConv(a, b // heads, heads=heads, concat=True, edge_dim=1))
            else:
                raise ValueError(f"unknown arch {arch}")
        if task == "graph":
            self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        root = (x[:, -1] == 1.0) if (self.task == "graph" and self.readout == "root") else None
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.shape[1], 1), dtype=x.dtype, device=x.device)
        edge_attr = edge_attr.float().view(-1, 1)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index) if self.arch == "GCN" else layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1 or self.task == "graph":
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.task == "graph":
            if root is not None:
                x = x[root]  # one root per graph, in batch order
            else:
                if batch is None:
                    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                x = global_mean_pool(x, batch)
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
        return list(dataset), "graph", "acc", dataset.num_node_features, 2 ** dataset.depth
    raise ValueError(f"Unknown dataset: {name}")


def compute_metric(y_true, y_pred, metric_name):
    y_true, y_pred = y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    if metric_name == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric_name == "ap":
        if not np.any(~np.isnan(y_true)):
            return 0.0
        y_true[np.isnan(y_true)] = 0
        return average_precision_score(y_true, y_pred)
    elif metric_name == "rocauc":
        if len(np.unique(y_true)) == 1:
            return 0.5
        return roc_auc_score(y_true, y_pred)
    elif metric_name == "acc":
        return accuracy_score(y_true, np.argmax(y_pred, axis=1))


def load_dataset_from_source(dataset_or_path):
    if isinstance(dataset_or_path, str):
        loaded = torch.load(dataset_or_path, weights_only=False)
        if isinstance(loaded, dict) and "dataset" in loaded:
            return loaded["dataset"]
        return loaded
    return dataset_or_path


# ==============================================================================
# Cross-validation with fold cache and training checkpoints
# ==============================================================================

def _rng_state():
    st = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _train_fold(model_params, lr, optim_name, epochs, device, train_step, ckpt_path, ckpt_seconds, label):
    """Train one fold, saving a resumable checkpoint every ``ckpt_seconds``."""
    model = GNN(**model_params).to(device)
    optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=lr)
    start = 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch_load(ckpt_path)
        if ck is not None:
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            _set_rng_state(ck["rng"]); start = int(ck["epoch"])
            print(f"  [Fold] {label}: resuming at epoch {start}/{epochs}", flush=True)
    last = time.time()
    for epoch in range(start, epochs):
        model.train()
        train_step(model, optimizer)
        if (epoch + 1) % 10 == 0:
            print(f"  [Fold] {label}: epoch {epoch + 1}/{epochs}", flush=True)
        if ckpt_path and epoch + 1 < epochs and time.time() - last > ckpt_seconds:
            atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "rng": _rng_state(), "epoch": epoch + 1}, ckpt_path)
            last = time.time()
    return model


def run_cv_fold(model_params, dataset_or_path, task, metric_name, cv_split, epochs, optim_name, seed,
                num_splits_to_run=None, device_id=None, return_folds=False, cache_dir=None, cache_tag="",
                ckpt_seconds=600):
    """K-fold CV score. With ``cache_dir``, every finished fold is stored and
    reused, and folds in training are checkpointed (see the module docstring)."""
    if device_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_pin_memory = (device.type == 'cuda')
    model_params = model_params.copy()
    lr = model_params.pop("lr", 0.01)
    dataset = None  # loaded lazily: all folds may be cached

    def ds():
        nonlocal dataset
        if dataset is None:
            dataset = load_dataset_from_source(dataset_or_path)
        return dataset

    n_items = None
    scores = []
    n_folds = cv_split if num_splits_to_run is None else min(cv_split, num_splits_to_run)
    for fold_idx in range(n_folds):
        key = _io.fingerprint(cache_tag, model_params, lr, task, metric_name, cv_split, epochs, optim_name,
                              int(seed), fold_idx, "v2")
        fpath = os.path.join(cache_dir, "fold_cache", f"{key}.json") if cache_dir else None
        cached = _io.read_json(fpath) if fpath else None
        if cached is not None and "score" in cached:
            scores.append(float(cached["score"]))
            continue
        data = ds()
        if n_items is None:
            n_items = data.num_nodes if task == "node" else len(data)
        train_idx, test_idx = list(KFold(n_splits=cv_split, shuffle=True, random_state=seed).split(np.arange(n_items)))[fold_idx]
        set_seed(fold_seed(seed, fold_idx))
        ckpt = os.path.join(cache_dir, "train_ckpt", f"{key}.pt") if cache_dir else None
        label = f"{cache_tag} fold {fold_idx + 1}/{n_folds}"

        if task == "graph":
            from torch_geometric.loader import DataLoader
            train_loader = DataLoader([data[i] for i in train_idx], batch_size=512, shuffle=True,
                                      pin_memory=use_pin_memory, num_workers=0)
            test_loader = DataLoader([data[i] for i in test_idx], batch_size=512, shuffle=False,
                                     pin_memory=use_pin_memory, num_workers=0)
            criterion = (torch.nn.L1Loss() if metric_name == "mae" else
                         torch.nn.CrossEntropyLoss() if metric_name == "acc" else torch.nn.BCEWithLogitsLoss())

            def step(model, optimizer):
                for batch in train_loader:
                    batch = batch.to(device, non_blocking=use_pin_memory)
                    optimizer.zero_grad()
                    out = model(batch.x.float(), batch.edge_index, edge_attr=getattr(batch, 'edge_attr', None),
                                batch=batch.batch)
                    y = batch.y.long().view(-1) if metric_name == "acc" else batch.y.view(out.shape).float()
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

            model = _train_fold(model_params, lr, optim_name, epochs, device, step, ckpt, ckpt_seconds, label)
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x.float(), batch.edge_index, edge_attr=getattr(batch, 'edge_attr', None),
                                batch=batch.batch)
                    y_true.append(batch.y.view(-1) if metric_name == "acc" else batch.y.view(out.shape))
                    y_pred.append(out)
            score = compute_metric(torch.cat(y_true), torch.cat(y_pred), metric_name)
        else:
            gdata = data.to(device)
            train_mask = torch.zeros(gdata.num_nodes, dtype=torch.bool, device=device)
            test_mask = torch.zeros(gdata.num_nodes, dtype=torch.bool, device=device)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
            criterion = torch.nn.BCEWithLogitsLoss()
            y = gdata.y.view(-1, 1).float()
            ea = getattr(gdata, 'edge_attr', None)

            def step(model, optimizer):
                optimizer.zero_grad()
                out = model(gdata.x.float(), gdata.edge_index, edge_attr=ea)
                loss = criterion(out[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()

            model = _train_fold(model_params, lr, optim_name, epochs, device, step, ckpt, ckpt_seconds, label)
            model.eval()
            with torch.no_grad():
                out = model(gdata.x.float(), gdata.edge_index, edge_attr=ea)
                score = compute_metric(y[test_mask], out[test_mask], metric_name)
        score = float(score)
        scores.append(score)
        if fpath:
            _io.atomic_json({"score": score, "tag": cache_tag, "fold": fold_idx, "seed": int(seed)}, fpath)
        if ckpt and os.path.exists(ckpt):
            os.remove(ckpt)
    if return_folds:
        return float(np.mean(scores)), [float(x) for x in scores]
    return float(np.mean(scores))


def global_worker_rep_wrapper(rep, num_gpus, best_params, cache_path, task, metric_name, cv_split, epochs,
                              optimizer_name, seed, cache_dir, cache_tag, ckpt_seconds, torch_threads):
    if torch_threads:
        torch.set_num_threads(torch_threads)
    gpu_id = rep % num_gpus if num_gpus > 0 else None
    rep_seed = seed + EVAL_SEED_OFFSET + rep
    score, folds = run_cv_fold(best_params, cache_path, task, metric_name, cv_split, epochs, optimizer_name,
                               seed=rep_seed, device_id=gpu_id, return_folds=True, cache_dir=cache_dir,
                               cache_tag=cache_tag, ckpt_seconds=ckpt_seconds)
    return rep, score, folds


def resolve_rewired_cache_path(dataset_name, arch_name, pipe_name_safe, max_iters_rewiring, out_dir,
                               cache_scope, shared_cache_dir):
    use_shared = (cache_scope == "shared") or (cache_scope == "auto" and dataset_name in MASSIVE_DATASETS)
    if use_shared:
        os.makedirs(shared_cache_dir, exist_ok=True)
        return os.path.join(shared_cache_dir, f"rewired_{dataset_name}_{pipe_name_safe}_iters{max_iters_rewiring}.pt")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"rewired_{dataset_name}_{arch_name}_{pipe_name_safe}_iters{max_iters_rewiring}.pt")


# ==============================================================================
# Optuna (journal storage: every finished trial is on disk)
# ==============================================================================

def _journal_storage(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:  # optuna >= 4
        from optuna.storages.journal import JournalFileBackend
        backend = JournalFileBackend(path)
    except ImportError:  # optuna 3.x
        from optuna.storages import JournalFileStorage
        backend = JournalFileStorage(path)
    return optuna.storages.JournalStorage(backend)


def tune(objective, study_name, journal, n_trials, seed, direction, n_jobs):
    """Run Optuna until ``n_trials`` trials are COMPLETE, resuming from the journal."""
    from optuna.trial import TrialState
    study = optuna.create_study(study_name=study_name, storage=_journal_storage(journal), direction=direction,
                                sampler=optuna.samplers.TPESampler(seed=seed), load_if_exists=True)
    # trials left RUNNING by a killed job: mark them failed and re-run the same hyperparameters
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
        if t.params:
            study.enqueue_trial(t.params, skip_if_exists=False)
        print(f"  [Optuna] trial {t.number} was interrupted; re-enqueued its parameters", flush=True)
    done = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    if done < n_trials:
        print(f"  [Optuna] {done}/{n_trials} trials complete; running {n_trials - done} more", flush=True)
        study.optimize(objective, n_trials=n_trials - done, n_jobs=n_jobs)
    else:
        print(f"  [Optuna] {done}/{n_trials} trials complete (from {journal})", flush=True)
    return study


# ==============================================================================
# Main driver
# ==============================================================================

def execute_trials(dataset_name, seed, n_trials, epochs, optimizer_name, cv_split, arch_name, n_reps, out_dir,
                   max_iters_rewiring, max_workers=4, pipelines_to_run=None, cache_scope="auto",
                   shared_cache_dir="data/rewired_cache", save_every=2000, save_seconds=600, ckpt_seconds=600):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    run_key = f"{dataset_name}_{arch_name}"
    part_file = os.path.join(out_dir, "parts", f"{run_key}.json")
    entry_all = _io.read_json(part_file)
    if entry_all is None:  # migrate from an older single-file layout
        entry_all = (_io.read_json(os.path.join(out_dir, "gnn_results.json"), {}) or {}).get(run_key, {})
    results = {run_key: entry_all}

    def save():
        _io.atomic_json(results[run_key], part_file)
        merge_results.merge(out_dir)

    print(f"\nProcessing Dataset: {dataset_name} | Arch: {arch_name}", flush=True)
    task, metric_name, in_c, out_c = DATASET_META[dataset_name]
    raw_cache = {"data": None}

    def _lazy_load_raw():
        if raw_cache["data"] is None:
            set_seed(seed)
            raw_cache["data"] = get_dataset(dataset_name)[0]
        return raw_cache["data"]

    nj = None if task == "node" else 1  # one big graph: use all Numba threads; many small ones: one per worker
    factories = {
        "Original": lambda: None,
        "FoSR": lambda: FoSRRewiring(max_iters=max_iters_rewiring),
        "BORF (OR)": lambda: BORFRewiring(metric="c_OR", max_iters=max_iters_rewiring, n_jobs=nj),
        "SDRF (OR)": lambda: SDRFRewiring(metric="c_OR", max_iters=max_iters_rewiring, n_jobs=nj),
        "SDRF (BF)": lambda: SDRFRewiring(metric="c_BF", max_iters=max_iters_rewiring, n_jobs=nj),
        "BORF (Bounds)": lambda: BORFRewiring(metric="bounds", max_iters=max_iters_rewiring, n_jobs=nj),
        "SDRF (Bounds)": lambda: SDRFRewiring(metric="bounds", max_iters=max_iters_rewiring, n_jobs=nj),
        "BORF (JL)": lambda: BORFRewiring(metric="jl_bounds", max_iters=max_iters_rewiring, n_jobs=nj),
        "BORF (Sharpened)": lambda: BORFRewiring(metric="sharpened", max_iters=max_iters_rewiring, n_jobs=nj),
    }
    names = [p for p in PIPELINES if not pipelines_to_run or p in pipelines_to_run]
    readout = "root" if dataset_name in ROOT_READOUT else "mean"
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)

    for pipe_name in names:
        existing_entry = results[run_key].get(pipe_name, {})
        existing_runs = existing_entry.get("runs", [])
        if len(existing_runs) >= n_reps:
            print(f"Skipping {pipe_name}, already evaluated ({len(existing_runs)}/{n_reps} reps).", flush=True)
            continue
        pname = safe_name(pipe_name)
        rewired_cache_path = resolve_rewired_cache_path(dataset_name, arch_name, pname, max_iters_rewiring,
                                                        out_dir, cache_scope, shared_cache_dir)
        use_shared = (cache_scope == "shared") or (cache_scope == "auto" and dataset_name in MASSIVE_DATASETS)
        tag = f"{dataset_name}_{pipe_name}" if use_shared else f"{dataset_name}_{arch_name}_{pipe_name}"
        rewire_seed = seed + sum(ord(c) for c in tag)  # deterministic, so restarts give the same trajectory

        # ---- 1. rewiring + topology (resumable)
        target_dataset, rewire_time, topology_metrics = get_or_create_rewired_dataset(
            dataset_name, pipe_name, factories[pipe_name](), _lazy_load_raw, task, max_workers,
            rewired_cache_path, save_every, save_seconds, rewire_seed)
        if pipe_name not in results[run_key]:
            results[run_key][pipe_name] = {"scores": [], "runs": [], "cv_split": cv_split, "avg": None, "std": None,
                                           "rewire_time": rewire_time, "topology": topology_metrics,
                                           "best_params": None}
            save()

        # ---- 2. Optuna (resumable per trial and per fold)
        params_file = os.path.join(out_dir, f"optuna_params_{dataset_name}_{arch_name}_{pname}.json")
        best_params = _io.read_json(params_file)
        cache_dir = os.path.join(out_dir, "cache", f"{dataset_name}_{arch_name}_{pname}_iters{max_iters_rewiring}")
        if best_params is None:
            print(f"Optuna hyperparameter search for {pipe_name} ({n_trials} trials)...", flush=True)
            tune_tag = f"tune|{dataset_name}|{arch_name}|{pipe_name}|iters{max_iters_rewiring}"

            def objective(trial):
                params = {
                    "in_channels": in_c,
                    "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 128]),
                    "out_channels": out_c,
                    "num_layers": trial.suggest_int("num_layers", 2, 5),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                    "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    "arch": arch_name, "task": task, "readout": readout,
                }
                return run_cv_fold(params, target_dataset, task, metric_name, cv_split, epochs, optimizer_name,
                                   seed=seed, num_splits_to_run=(cv_split // 2) + 1, cache_dir=cache_dir,
                                   cache_tag=tune_tag, ckpt_seconds=ckpt_seconds)

            with held(params_file + ".lock", f"the Optuna study of {pipe_name}"):
                journal = os.path.join(out_dir, "optuna", f"{dataset_name}_{arch_name}_{pname}.journal")
                study = tune(objective, f"{dataset_name}_{arch_name}_{pname}", journal, n_trials, seed,
                             "minimize" if metric_name == "mae" else "maximize", min(4, max_workers))
                best_params = dict(study.best_params)
                best_params.update({"in_channels": in_c, "out_channels": out_c, "arch": arch_name, "task": task,
                                    "readout": readout})
                _io.atomic_json(best_params, params_file)
            print(f"  [Optuna] Saved best hyperparameters to {params_file}.", flush=True)
        del target_dataset

        # ---- 3. evaluation repetitions (resumable per repetition, fold, and epoch)
        runs = list(existing_runs)
        done = {r["rep"] for r in runs}
        remaining = [r for r in range(n_reps) if r not in done]
        print(f"Executing {len(remaining)} remaining repetitions for {pipe_name}...", flush=True)
        num_gpus = torch.cuda.device_count()
        workers = min(max_workers, num_gpus * 2) if num_gpus > 0 else max_workers
        worker_fn = partial(global_worker_rep_wrapper, num_gpus=num_gpus, best_params=best_params,
                            cache_path=rewired_cache_path, task=task, metric_name=metric_name, cv_split=cv_split,
                            epochs=epochs, optimizer_name=optimizer_name, seed=seed, cache_dir=cache_dir,
                            cache_tag=f"eval|{dataset_name}|{arch_name}|{pipe_name}|iters{max_iters_rewiring}",
                            ckpt_seconds=ckpt_seconds, torch_threads=max(1, cpus // max(1, workers)))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_fn, rep) for rep in remaining]
            for fut in as_completed(futures):
                rep_idx, rep_score, rep_folds = fut.result()
                runs.append({"rep": int(rep_idx), "seed": int(seed + EVAL_SEED_OFFSET + rep_idx),
                             "score": rep_score, "folds": rep_folds})
                runs.sort(key=lambda r: r["rep"])
                scores = [r["score"] for r in runs]
                results[run_key][pipe_name] = {
                    "scores": scores, "runs": runs, "cv_split": cv_split,
                    "avg": float(np.mean(scores)), "std": float(np.std(scores)),
                    "rewire_time": rewire_time, "topology": topology_metrics, "best_params": best_params}
                save()
                print(f"  [Rep {len(scores)}/{n_reps}] rep {rep_idx}: {rep_score:.4f} | running avg "
                      f"{np.mean(scores):.4f} ± {np.std(scores):.4f}", flush=True)
        if runs:
            sc = [r["score"] for r in runs]
            print(f"Completed {pipe_name} | {metric_name.upper()}: {np.mean(sc):.4f} ± {np.std(sc):.4f} | "
                  f"rewire time {rewire_time:.2f}s", flush=True)
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN rewiring case study (resumable)")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_META))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials per topology")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--cv-split", type=int, default=5, help="Number of K-Fold splits")
    parser.add_argument("--arch", type=str, default="GCN", choices=["GCN", "GIN", "GAT"])
    parser.add_argument("--reps", type=int, default=3, help="Number of evaluation repetitions")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--max-iters-rewiring", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker processes")
    parser.add_argument("--pipelines", type=str, nargs="*", default=None, help="subset of pipelines to run")
    parser.add_argument("--cache-scope", type=str, default="auto", choices=["auto", "arch", "shared"],
                        help="'auto': share rewired graphs across architectures for QM9 and ogbg-molhiv")
    parser.add_argument("--shared-cache-dir", type=str, default="data/rewired_cache")
    parser.add_argument("--save-every", type=int, default=2000, help="graphs between partial rewiring checkpoints")
    parser.add_argument("--save-seconds", type=float, default=600, help="seconds between partial rewiring checkpoints")
    parser.add_argument("--ckpt-seconds", type=float, default=600, help="seconds between training checkpoints of a fold")
    args = parser.parse_args()
    execute_trials(dataset_name=args.dataset, seed=args.seed, n_trials=args.trials, epochs=args.epochs,
                   optimizer_name=args.optimizer, cv_split=args.cv_split, arch_name=args.arch, n_reps=args.reps,
                   out_dir=args.out_dir, max_iters_rewiring=args.max_iters_rewiring, max_workers=args.workers,
                   pipelines_to_run=args.pipelines, cache_scope=args.cache_scope,
                   shared_cache_dir=args.shared_cache_dir, save_every=args.save_every,
                   save_seconds=args.save_seconds, ckpt_seconds=args.ckpt_seconds)
