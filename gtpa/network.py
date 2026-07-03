import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    implemented in vanilla PyTorch.
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h shape: (batch_size, num_nodes, in_features)
        # adj shape: (batch_size, num_nodes, num_nodes)
        
        batch_size, num_nodes, _ = h.size()
        Wh = self.W(h) # (batch_size, num_nodes, out_features)
        
        # Prepare for attention computation
        # Repeat Wh to prepare pairs
        Wh_repeated_1 = Wh.repeat_interleave(num_nodes, dim=1) # (batch_size, num_nodes * num_nodes, out_features)
        Wh_repeated_2 = Wh.repeat(1, num_nodes, 1)            # (batch_size, num_nodes * num_nodes, out_features)
        
        all_combinations = torch.cat([Wh_repeated_1, Wh_repeated_2], dim=-1) 
        # Shape: (batch_size, num_nodes * num_nodes, 2 * out_features)
        
        e = self.leakyrelu(self.a(all_combinations)) # (batch_size, num_nodes * num_nodes, 1)
        e = e.view(batch_size, num_nodes, num_nodes) # (batch_size, num_nodes, num_nodes)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.bmm(attention, Wh) # (batch_size, num_nodes, out_features)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class NeuralPropensityNet(nn.Module):
    """
    Uses GAT layers to process agent interactions tatically,
    then outputs propensity logits for the Gillespie simulator.
    """
    def __init__(self, node_in_features=10, hidden_dim=32, num_actions=10, dropout=0.2):
        super(NeuralPropensityNet, self).__init__()
        self.dropout = dropout
        
        # We can use a 2-head GAT layer
        self.gat1 = GraphAttentionLayer(node_in_features, hidden_dim, dropout=dropout, concat=True)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout, concat=False)
        
        # MLP for predicting propensity logits from node embeddings
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, node_features, adj):
        """
        node_features: tensor of shape (batch_size, num_players, node_in_features)
        adj: Delaunay adjacency matrix of shape (batch_size, num_players, num_players)
        """
        x = self.encode(node_features, adj)

        combined = torch.cat([x, node_features], dim=-1)
        logits = self.mlp(combined) # (batch_size, num_players, num_actions)
        logits = torch.clamp(logits, min=-15.0, max=15.0)
        return logits

    def encode(self, node_features, adj):
        """Return latent graph embeddings before the action head."""
        x = self.gat1(node_features, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj)
        return x
