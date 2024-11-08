import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader

class GeneExpressionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GeneExpressionGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second GCN layer
        x = self.conv2(x, edge_index)
        # Extract 'big' node embeddings for each graph in the batch
        num_nodes_per_graph = torch.bincount(batch)
        cumsum_nodes = torch.cumsum(num_nodes_per_graph, dim=0)
        big_node_indices = cumsum_nodes - 1  # Indices of 'big' nodes
        gene_node_embeddings = x[big_node_indices]
        # Predicted gene expression
        gene_expression_pred = gene_node_embeddings.squeeze()
        return gene_expression_pred

def build_gene_graph(bpnet_embeddings, hic_edges, gene_feature=None):
    """
    Constructs a graph for a single gene.

    Args:
        bpnet_embeddings (Tensor): Shape [num_peaks, embedding_dim], BPNet embeddings for peaks.
        hic_edges (Tensor): Shape [2, num_edges], Hi-C edge indices between peaks.
        gene_feature (Tensor, optional): Shape [embedding_dim], feature for 'big' node. Defaults to zeros.

    Returns:
        Data: PyTorch Geometric Data object representing the gene graph.
    """
    num_peaks = bpnet_embeddings.shape[0]
    embedding_dim = bpnet_embeddings.shape[1]
    
    # Feature for 'big' node
    if gene_feature is None:
        gene_feature = torch.zeros(embedding_dim)
    
    # Combine peak features and 'big' node feature
    node_features = torch.cat([bpnet_embeddings, gene_feature.unsqueeze(0)], dim=0)
    
    # Edges between peaks (Hi-C interactions)
    edge_index = hic_edges
    
    # Edges from peaks to 'big' node
    big_node_index = num_peaks  # Index of 'big' node
    peaks_indices = torch.arange(num_peaks)
    big_node_edges = torch.stack([peaks_indices, torch.full((num_peaks,), big_node_index)], dim=0)
    
    # Combine Hi-C edges and 'big' node edges
    edge_index = torch.cat([edge_index, big_node_edges], dim=1)
    
    # Create Data object
    data = Data(x=node_features, edge_index=edge_index)
    return data

# Example usage:
# Assuming you have preprocessed data for each gene
# gene_list: List of gene identifiers
# bpnet_embeddings_dict: Dict mapping gene_id -> Tensor of BPNet embeddings
# hic_edges_dict: Dict mapping gene_id -> Tensor of Hi-C edge indices
# gene_expression_dict: Dict mapping gene_id -> Actual gene expression value (normalized)

data_list = []
for gene_id in gene_list:
    bpnet_embeddings = bpnet_embeddings_dict[gene_id]  # [num_peaks, embedding_dim]
    hic_edges = hic_edges_dict[gene_id]  # [2, num_edges]
    gene_expression = gene_expression_dict[gene_id]  # Scalar value
    
    # Build graph for the gene
    data = build_gene_graph(bpnet_embeddings, hic_edges)
    data.y = torch.tensor([gene_expression], dtype=torch.float)  # Add target expression value
    data_list.append(data)

# Create DataLoader
loader = DataLoader(data_list, batch_size=32, shuffle=True)

# Initialize model, optimizer, and loss function
embedding_dim = bpnet_embeddings.shape[1]
hidden_channels = 64  # Set as needed
model = GeneExpressionGNN(in_channels=embedding_dim, hidden_channels=hidden_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Training loop
num_epochs = 100  # Set as needed
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1))  # Ensure shapes match
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader)}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for data in test_loader:
        output = model(data)
        predictions.append(output.cpu())
        actuals.append(data.y.view(-1).cpu())
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()
    
    # Mean Squared Error
    mse = np.mean((predictions - actuals) ** 2)
    
    # R-squared Score
    r2 = r2_score(actuals, predictions)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R-squared: {r2:.4f}")