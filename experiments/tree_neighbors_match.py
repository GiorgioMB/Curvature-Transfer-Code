"""
Generates the TREE-NEIGHBORSMATCH synthetic benchmark to evaluate the over-squashing bottleneck in graph neural networks.
References:
    Alon, U., & Yahav, E. (2021). On the bottleneck of graph neural networks and its practical implications. 
    In Proceedings of the 9th International Conference on Learning Representations (ICLR). 
    https://arxiv.org/abs/2006.05205
"""
import torch
from torch_geometric.data import Data, InMemoryDataset
import random
import os

class TreeNeighborsMatch(InMemoryDataset):
    def __init__(self, root, depth=4, num_graphs=16000, transform=None, pre_transform=None):
        self.depth = depth
        self.num_graphs = num_graphs
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'tree_neighbors_match_d{self.depth}_{self.num_graphs}.pt']

    def process(self):
        data_list = []
        num_leaves = 2 ** self.depth
        num_nodes = 2 ** (self.depth + 1) - 1
        
        feature_dim = num_leaves + num_leaves + 3
        
        for _ in range(self.num_graphs):
            # Generate directed tree topology (edges directed toward the root)
            edges = []
            for i in range(2 ** self.depth - 1):
                edges.append([2 * i + 1, i]) 
                edges.append([2 * i + 2, i])
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Permutate labels and counts for the leaves
            labels = list(range(num_leaves))
            counts = list(range(num_leaves))
            random.shuffle(labels)
            random.shuffle(counts)
            
            # Select target matching criteria
            target_idx = random.randint(0, num_leaves - 1)
            target_count = counts[target_idx]
            target_label = labels[target_idx]
            
            # Construct node features
            x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
            
            for i in range(num_nodes):
                if i == 0:
                    # Target node (root)
                    x[i, num_leaves + target_count] = 1.0  # Queried neighbor count
                    x[i, -1] = 1.0                         # Root type indicator
                elif i >= (2 ** self.depth - 1):
                    # Leaf nodes (green nodes)
                    leaf_idx = i - (2 ** self.depth - 1)
                    l = labels[leaf_idx]
                    c = counts[leaf_idx]
                    x[i, l] = 1.0                          # Alphabetical label
                    x[i, num_leaves + c] = 1.0             # Blue neighbor count
                    x[i, -3] = 1.0                         # Leaf type indicator
                else:
                    # Intermediate nodes
                    x[i, -2] = 1.0                         # Intermediate type indicator
                    
            # The classification target is the label of the leaf matching the root's count
            y = torch.tensor([target_label], dtype=torch.long)
            
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
