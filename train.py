# from bpnetlite import BPNet, ChromBPNet
from tangermeme.io import extract_loci
from tangermeme.predict import predict
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from models import BPExtractor
import data
import torch
import models
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
import gzip
from torch_geometric.utils import scatter
from tqdm import tqdm

import importlib

importlib.reload(data)
importlib.reload(models)

from data import sample_tpm_rows, remove_duplicate_edges, build_enhancer_graph, add_virtual_node, EnhancerGraphDataset,EnhancerGraphDatasetClassify
from models import VirtualNodeGNN, GeneExpressionGNN

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_data(gene_window=50_000, enhancer_window=50_000, batch_size=256):
    peak_file = "/oak/stanford/groups/akundaje/kobbad/CS224W/data/ENCFF439EIO.bed.gz"
    gene_data = "./data/gene_data.tsv"

    gene_data_df = pd.read_csv(gene_data, sep="\t")
    peaks = pd.read_csv(peak_file, sep="\t", header=None)
    peaks = peaks.drop_duplicates(subset=peaks.columns[[0, 1, 2]])
    peaks.head()

    embeddings = np.load("./data/embeddings.npz")
    embeddings = embeddings['embeddings']

    gene_zero = gene_data_df[gene_data_df.TPM == 0]
    gene_nonzero = gene_data_df[gene_data_df.TPM > 0]


    gene_zero = gene_zero.sample(gene_nonzero.shape[0])

    genes = pd.concat([gene_zero, gene_nonzero])

    train_chroms = [f"chr{x}" for x in range(2, 8)] + [f"chr{x}" for x in range(9, 23)]
    val_chroms = ["chr8", "chr10"]
    test_chroms = ["chr1"]

    train_genes = genes[genes["seqname"].isin(train_chroms)]
    val_genes = genes[genes["seqname"].isin(val_chroms)]
    test_genes = genes[genes["seqname"].isin(test_chroms)]

    train_dataset = EnhancerGraphDatasetClassify(train_genes, peaks, embeddings, gene_window=gene_window, enhancer_window=enhancer_window)
    val_dataset = EnhancerGraphDatasetClassify(val_genes, peaks, embeddings, gene_window=gene_window, enhancer_window=enhancer_window)
    test_dataset = EnhancerGraphDatasetClassify(test_genes, peaks, embeddings, gene_window=gene_window, enhancer_window=enhancer_window)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader

def train_model(train_loader, val_loader, output_dir, epochs=200, lr=0.001, num_layers=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, optimizer, and loss
    input_dim = 512
    hidden_dim = 128
    output_dim = 1  # Predicting a single value (e.g., regression)

    model = VirtualNodeGNN(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    #model = GeneExpressionGNN(input_dim, hidden_dim, output_dim)

    optimizer = Adam(model.parameters(), lr=lr)

    #loss_fn = torch.nn.MSELoss()  # Example for regression
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.to(device);

    # Training loop
    train_loss_list = []
    val_accuracy_list = []

    best_val_accuracy = 0
    best_model = None

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        
        # Training loop
        for batch in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            batch.to('cuda')  # Move batch to GPU if available
            
            # Forward pass
            out = model(batch)
            loss = loss_fn(out.squeeze(), batch.y.float().squeeze())  # Match dimensions
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track training loss
            total_loss += loss.item()
            train_loss_list.append(loss.item())
        
        print(f"Epoch {epoch}, Training Loss: {total_loss / len(train_loader)}")
        
        # Validation loop
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        
        with torch.no_grad():
            for val_batch in val_loader:  # Use validation loader
                val_batch.to('cuda')  # Move to GPU if available
                
                # Forward pass
                logits = model(val_batch)  # Get raw logits
                probs = torch.sigmoid(logits)  # Convert logits to probabilities
                predictions = (probs > 0.5).float()  # Threshold probabilities
                
                # Calculate accuracy
                correct += (predictions.squeeze() == val_batch.y).sum().item()
                total += val_batch.y.size(0)
        
        # Compute accuracy
        accuracy = correct / total

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_model = model
            torch.save(model.state_dict(), f"{output_dir}/best_model_layers.pth")
        
        val_accuracy_list.append(accuracy)
        print(f"Epoch {epoch}, Validation Accuracy: {accuracy:.4f}")

    # make a plot of val_accuracy_list and save it to output_dir
    plt.plot(val_accuracy_list)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Epoch")
    plt.savefig(f"{output_dir}/val_accuracy_plot.png")
    plt.close()

    # save the val_accuracy_list to output_dir
    np.save(f"{output_dir}/val_accuracy_list.npy", val_accuracy_list)

    return best_model

def evaluate_model(model, test_loader, output_dir):
    # evaluate on test set
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for test_batch in test_loader:
            test_batch.to('cuda')
            
            # Forward pass
            logits = model(test_batch)  # Get raw logits
            probs = torch.sigmoid(logits)  # Convert logits to probabilities
            predictions = (probs > 0.5).float()  # Threshold probabilities
            
            # Calculate accuracy
            correct += (predictions.squeeze() == test_batch.y).sum().item()
            total += test_batch.y.size(0)
    
    # Compute accuracy
    accuracy = correct / total

    # save the accuracy to output_dir in txt file
    with open(f"{output_dir}/accuracy.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":

    num_layers_list = [1,2,3,4]

    gene_window_list = [50_000, 100_000, 200_000, 500_000]
    enhancer_window_list = [10_000, 20_000, 50_000]

    for num_layers in num_layers_list:
        for gene_window in gene_window_list:
            for enhancer_window in enhancer_window_list:

                if enhancer_window >= gene_window:
                    continue
                
                output_dir = f"./outputs/output_{num_layers}_{gene_window}_{enhancer_window}"
                
                os.makedirs(output_dir, exist_ok=True)

                train_loader, val_loader, test_loader = load_data(gene_window=gene_window, enhancer_window=enhancer_window)
                model = train_model(train_loader, val_loader, output_dir, num_layers=num_layers)
                evaluate_model(model, test_loader, output_dir)


        







