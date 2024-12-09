from bpnetlite import BPNet
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import scatter

class BPExtractor(nn.Module):
    def __init__(self, original_model):
        super(BPExtractor, self).__init__()
        self.original_model = original_model

    def forward(self, X, X_ctl=None, return_intermediate=True):
        start, end = self.original_model.trimming, X.shape[2] - self.original_model.trimming

        # Initial Convolution Block
        X = self.original_model.irelu(self.original_model.iconv(X))

        # Residual Convolutions
        for i in range(self.original_model.n_layers):
            X_conv = self.original_model.rrelus[i](self.original_model.rconvs[i](X))
            X = torch.add(X, X_conv)

        # If X_ctl is provided, concatenate
        if X_ctl is None:
            X_w_ctl = X
        else:
            X_w_ctl = torch.cat([X, X_ctl], dim=1)

        # Profile prediction (y_profile)
        y_profile = self.original_model.fconv(X_w_ctl)[:, :, start:end]

        # Counts prediction (X before linear)
        X_before_linear = torch.mean(X[:, :, start-37:end+37], dim=2)

        if X_ctl is not None:
            X_ctl = torch.sum(X_ctl[:, :, start-37:end+37], dim=(1, 2))
            X_ctl = X_ctl.unsqueeze(-1)
            X_before_linear = torch.cat([X_before_linear, torch.log(X_ctl + 1)], dim=-1)

        # If return_intermediate is True, return X_before_linear
        if return_intermediate:
            return X_before_linear

        # Pass X_before_linear to the linear layer for y_counts
        y_counts = self.original_model.linear(X_before_linear).reshape(X_before_linear.shape[0], 1)

        # Return the original model outputs
        return y_profile, y_counts

class VirtualNodeGNNOriginal(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(VirtualNodeGNN, self).__init__()
        
        # GCN layers for node feature propagation
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Fully connected layer for label prediction using virtual node embedding
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GCN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        # Extract virtual node embedding
        virtual_node_embedding = x[-1]  # The last node is the virtual node

        # Predict label using the virtual node embedding
        out = self.fc(virtual_node_embedding)
        
        return out


from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class VirtualNodeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(VirtualNodeGNN, self).__init__()
        
        # Virtual node embedding and its MLP
        self.virtual_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GCN layers for node feature propagation
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Fully connected layer for prediction
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initialize virtual node embeddings for each graph
        virtual_node = torch.zeros(batch.max().item() + 1, x.size(1)).to(x.device)
        
        # Message passing layers with virtual node
        for conv in self.convs:
            # 1. Add virtual node features to all nodes in respective graphs
            x = x + virtual_node[batch]
            
            # 2. Apply GCN layer
            x = F.relu(conv(x, edge_index))
            
            # 3. Update virtual nodes by aggregating from real nodes
            # Choose one of these:
            virtual_node_temp = global_mean_pool(x, batch)  # mean pooling
            # virtual_node_temp = global_add_pool(x, batch)   # sum pooling
            # virtual_node_temp = global_max_pool(x, batch)   # max pooling
            
            # 4. Apply MLP to update virtual node features
            virtual_node = self.virtual_mlp(virtual_node_temp)
        
        # Use virtual nodes for prediction
        out = self.fc(virtual_node)
        
        return out

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GeneExpressionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GeneExpressionGNN, self).__init__()
        
        # Define GCN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Fully connected layers for graph-level prediction
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        # Extract graph components
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GCN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        # Aggregate node features into a graph-level embedding
        x = global_mean_pool(x, batch)  # Use mean pooling
        
        # Fully connected layers for prediction
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

# class GeneExpressionGNN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
#         super(GeneExpressionGNN, self).__init__()
        
#         # Define GCN layers
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GCNConv(input_dim, hidden_dim))
#         for _ in range(num_layers - 2):
#             self.convs.append(GCNConv(hidden_dim, hidden_dim))
#         self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
#         # Virtual node MLP for learned aggregation
#         self.virtual_mlp = torch.nn.Sequential(
#             torch.nn.Linear(hidden_dim, hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, hidden_dim)
#         )
        
#         # Fully connected layers for graph-level prediction
#         self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, data):
#         # Extract graph components
#         x, edge_index, batch = data.x, data.edge_index, data.batch
        
#         # Initialize virtual node embedding
#         virtual_node = torch.zeros(batch.max().item() + 1, x.size(1)).to(x.device)
        
#         # Apply GCN layers with skip connections
#         for conv in self.convs:
#             x_residual = x  # Save input for skip connection
#             x = F.relu(conv(x, edge_index)) + x_residual  # Skip connection
#             x = x + virtual_node[batch]  # Add virtual node information
            
#             # Update virtual node using aggregated node features
#             virtual_node_temp = global_mean_pool(x, batch)  # Pool node features for each graph
#             virtual_node = self.virtual_mlp(virtual_node_temp) + virtual_node  # Update with residual
        
#         # Use the final virtual node embedding as the graph-level representation
#         graph_representation = virtual_node
        
#         # Fully connected layers for prediction
#         x = F.relu(self.fc1(graph_representation))
#         x = self.fc2(x)
        
#         return x

