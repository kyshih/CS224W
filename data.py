import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, Dataset


def sample_tpm_rows(dataframe, percentage_zero_tpm, percentile = 0):
    """
    Randomly samples rows from the dataframe with all rows having TPM > 0
    and X% of rows with TPM = 0.

    Parameters:
    - dataframe (pd.DataFrame): The input dataframe containing a 'TPM' column.
    - percentage_zero_tpm (float): The percentage of rows with TPM = 0 to sample (0 to 100).

    Returns:
    - pd.DataFrame: A dataframe containing the sampled rows.
    """
    # Split the dataframe into two groups
    low_threshold = np.percentile(dataframe['TPM'], percentile)
    non_zero_tpm = dataframe[dataframe['TPM'] > low_threshold]
    zero_tpm = dataframe[dataframe['TPM'] <= 0]
    
    # Calculate the number of rows to sample from zero TPM group
    num_zero_tpm_to_sample = int(len(zero_tpm) * (percentage_zero_tpm / 100))
    
    # Sample the zero TPM rows
    sampled_zero_tpm = zero_tpm.sample(n=num_zero_tpm_to_sample, random_state=42)
    
    # Concatenate all rows with TPM > 0 and the sampled zero TPM rows
    sampled_df = pd.concat([non_zero_tpm, sampled_zero_tpm])
    
    # Shuffle the final dataframe for randomness
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return sampled_df

def remove_duplicate_edges(edge_index):
    """
    Remove duplicate edges from edge_index for an undirected graph.

    Args:
        edge_index (torch.Tensor): Edge index tensor of shape [2, num_edges].

    Returns:
        torch.Tensor: Edge index tensor with duplicate edges removed.
    """
    # Sort each edge (smallest node first) to treat undirected edges uniformly
    sorted_edges = edge_index.sort(dim=0).values  # Sort each column of edge_index

    # Remove duplicate edges
    unique_edges = torch.unique(sorted_edges, dim=1)

    return unique_edges

def build_enhancer_graph(df, W):
    """
    Build a graph where enhancers are connected if they are within a distance W.

    Args:
        df (pd.DataFrame): DataFrame containing enhancer information in BED format.
        W (int): Distance threshold to connect nearby enhancers.

    Returns:
        torch.Tensor: Edge list for PyTorch Geometric.
    """
    counter = 0
    edge_index = []

    embedding_index = []

    for row in df.iterrows():
        embedding_index.append(row[0])
        second_counter = 0

        _, start, end, *_ = row[1]

        for second_row in df.iterrows():
            _, second_start, second_end, *_ = second_row[1]

            if abs(start - second_start) < W and abs(end - second_end) < W:
                edge_index.append([counter, second_counter])

            second_counter += 1
    
        counter += 1
    
    # remove self loops
    edge_index = [edge for edge in edge_index if edge[0] != edge[1]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).T


    if edge_index.shape[0] == 0:
        raise ValueError("No edges found")

    edge_index = remove_duplicate_edges(edge_index)

    # convert embedding index to tensor long
    embedding_index = torch.tensor(embedding_index, dtype=torch.long)

    return edge_index, embedding_index

def add_virtual_node(graph):
    """
    Add a virtual node to an undirected graph, connecting it to all other nodes.

    Args:
        graph (torch_geometric.data.Data): Input graph with `x` (node features) and `edge_index`.

    Returns:
        torch_geometric.data.Data: Graph with added virtual node and edges.
    """
    num_nodes = graph.x.size(0)  # Number of original nodes
    num_features = graph.x.size(1)  # Feature dimension per node

    # Add a virtual node feature (initialized with zeros or learnable parameters)
    virtual_node_feature = torch.zeros((1, num_features), dtype=graph.x.dtype)
    graph.x = torch.cat([graph.x, virtual_node_feature], dim=0)

    # Create edges connecting the virtual node (num_nodes) to all other nodes
    virtual_edges = torch.tensor(
        [[num_nodes] * num_nodes,  # Virtual node -> all other nodes
         list(range(num_nodes))],  # All other nodes -> Virtual node
        dtype=graph.edge_index.dtype,
    )
    
    graph.edge_index = torch.cat([graph.edge_index, virtual_edges], dim=1)

    return graph

class EnhancerGraphDataset(Dataset):
    def __init__(self, dataframes,peaks, embedding, gene_window = 500_000, enhancer_window = 50_000, transform=None):
        """
        Args:
            dataframes (list): List of DataFrames, one for each graph.
            labels (list): List of labels for each graph.
            W (int): Distance threshold for connecting nodes.
        """
        self.dataframes = dataframes
        self.peaks = peaks
        self.peaks = self.peaks.drop_duplicates(subset=peaks.columns[[0, 1, 2]])

        self.embedding = torch.tensor(embedding)

        self.gene_window = gene_window
        self.enhancer_window = enhancer_window

        self.transform = transform

    def len(self):
        return len(self.dataframes)

    
    def indices(self):
        return list(range(len(self.dataframes)))

    def get(self, idx):
        # Build the graph
        _, chrom, start, end, label = self.dataframes.iloc[idx]

        label = np.log1p(label)

        adj_peaks = self.peaks[(self.peaks[1] > start - self.gene_window) & (self.peaks[2] < end + self.gene_window) & (self.peaks[0] == chrom)]
        
        try:
            edge_index, embedding_index = build_enhancer_graph(adj_peaks, self.enhancer_window)
        except:
            return self.get(idx + 1)

        node_features = self.embedding[embedding_index]
        
        # Create a graph data object
        graph = Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))

        # Add a virtual node
        #graph = add_virtual_node(graph)

        return graph


class EnhancerGraphDatasetClassify(Dataset):
    def __init__(self, dataframes,peaks, embedding, gene_window = 500_000, enhancer_window = 50_000, transform=None):
        """
        Args:
            dataframes (list): List of DataFrames, one for each graph.
            labels (list): List of labels for each graph.
            W (int): Distance threshold for connecting nodes.
        """
        self.dataframes = dataframes
        self.peaks = peaks
        self.peaks = self.peaks.drop_duplicates(subset=peaks.columns[[0, 1, 2]])

        self.embedding = torch.tensor(embedding)

        self.gene_window = gene_window
        self.enhancer_window = enhancer_window

        self.transform = transform

    def len(self):
        return len(self.dataframes)

    
    def indices(self):
        return list(range(len(self.dataframes)))

    def get(self, idx):
        # Build the graph
        _, chrom, start, end, label = self.dataframes.iloc[idx]

        label = 1 if label > 0 else 0

        adj_peaks = self.peaks[(self.peaks[1] > start - self.gene_window) & (self.peaks[2] < end + self.gene_window) & (self.peaks[0] == chrom)]
        
        try:
            edge_index, embedding_index = build_enhancer_graph(adj_peaks, self.enhancer_window)
        except:
            return self.get(idx + 1)

        node_features = self.embedding[embedding_index]
        
        # Create a graph data object
        graph = Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))

        # Add a virtual node
        #graph = add_virtual_node(graph)

        return graph
