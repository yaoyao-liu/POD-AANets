import torch
from torch.nn import functional as F


def squared_euclidian_distance(a, b):
    return torch.cdist(a, b)**2


def cosine_similarity(a, b):
    return torch.mm(F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1).T)


def stable_cosine_distance(a, b, squared=True):
    mat = torch.cat([a, b])

    pairwise_distances_squared = torch.add(
        mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
        torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
    ) - 2 * (torch.mm(mat, torch.t(mat)))
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]
