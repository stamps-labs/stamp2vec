import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.functional import relu



class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1., squared=False, agg='sum'):
        """
        Initalize the loss function with a margin parameter, whether or not to consider
        squared Euclidean distance and how to aggregate the loss in a batch
        """
        super().__init__()
        self.margin = margin
        self.squared = squared
        self.agg = agg
        self.eps = 1e-8

    def get_pairwise_distances(self, embeddings):
        """
        Computing Euclidean distance for all possible pairs of embeddings.
        """
        ab = embeddings.mm(embeddings.t())
        a_squared = ab.diag().unsqueeze(1)
        b_squared = ab.diag().unsqueeze(0)
        distances = a_squared - 2 * ab + b_squared
        distances = relu(distances)

        if not self.squared:
            distances = torch.sqrt(distances + self.eps)

        return distances

    def hardest_triplet_mining(self, dist_mat, labels):
        
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)
        
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        dist_ap, relative_p_inds = torch.max(
            (dist_mat * is_pos), 1, keepdim=True)
        
        dist_an, relative_n_inds = torch.min(
            (dist_mat * is_neg), 1, keepdim=True)

        return dist_ap, dist_an

    def forward(self, embeddings, labels):
        
        distances = self.get_pairwise_distances(embeddings)
        dist_ap, dist_an = self.hardest_triplet_mining(distances, labels)

        triplet_loss = relu(dist_ap - dist_an + self.margin).sum()
        return triplet_loss


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstruction_loss = nn.BCELoss(reduction='sum')
    
    def kl_divergence_loss(self, q_dist):
        return kl_divergence(
            q_dist, Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.stddev))
        ).sum(-1)
    
    
    def forward(self, output, target, encoding):
        loss = self.kl_divergence_loss(encoding).sum() + self.reconstruction_loss(output, target)
        return loss


