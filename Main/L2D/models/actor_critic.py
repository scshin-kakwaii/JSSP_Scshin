import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
from models.mlp import MLPActor, MLPCritic # Import class cụ thể

class ActorCritic(nn.Module):
    def __init__(self,
                 n_j, n_m,
                 num_layers, learn_eps, neighbor_pooling_type,
                 input_dim, hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor, hidden_dim_actor,
                 num_mlp_layers_critic, hidden_dim_critic,
                 device):
        super(ActorCritic, self).__init__()
        self.n_j = n_j
        self.n_m = n_m
        self.device = device
        self.num_rules = 8
        self.rule_fea_dim = 11

        # 1. Graph Feature Extractor
        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)

        # 2. Actor Scoring Network
        # Input: Global(Hidden) + Candidate(Hidden) + RuleFea(11)
        actor_input_dim = hidden_dim + hidden_dim + self.rule_fea_dim
        
        self.actor_scorer = MLPActor(num_mlp_layers_actor, actor_input_dim, hidden_dim_actor, 1).to(device)

        # 3. Critic
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self, x, graph_pool, padded_nei, adj, candidates, mask, rule_features):
        """
        x: Node features
        candidates: [Batch, n_j]
        mask: [Batch, n_j] (True if job finished)
        rule_features: [Batch, 8, 11]
        """
        # 1. Graph CNN
        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        
        batch_size = h_pooled.shape[0]
        
        # 2. Candidate Pooling
        h_nodes_reshaped = h_nodes.view(batch_size, -1, h_nodes.size(-1)) 
        
        candidate_indices = candidates.unsqueeze(-1).expand(-1, -1, h_nodes.size(-1))
        candidate_embeds = torch.gather(h_nodes_reshaped, 1, candidate_indices)
        
        # Mask finished jobs
        mask_expanded = mask.unsqueeze(-1).expand_as(candidate_embeds)
        candidate_embeds = candidate_embeds.masked_fill(mask_expanded, 0.0)
        
        # Average Pool over active candidates
        num_active = (~mask).sum(dim=1, keepdim=True).float() + 1e-5
        candidate_context = candidate_embeds.sum(dim=1) / num_active 
        
        # 3. Action Scoring
        # Context: [Batch, 1, 2*Hidden]
        context = torch.cat([h_pooled, candidate_context], dim=1).unsqueeze(1)
        context_expanded = context.expand(-1, self.num_rules, -1)
        
        # Actor Input: [Batch, 8, 2*Hidden + 11]
        actor_input = torch.cat([context_expanded, rule_features], dim=-1)
        
        # Flatten for MLP
        actor_input_flat = actor_input.view(-1, actor_input.size(-1))
        scores_flat = self.actor_scorer(actor_input_flat)
        scores = scores_flat.view(batch_size, self.num_rules)
        
        pi = F.softmax(scores, dim=-1)
        v = self.critic(h_pooled)
        
        return pi, v