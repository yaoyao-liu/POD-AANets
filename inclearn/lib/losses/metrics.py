import itertools

import numpy as np
import torch
from torch.nn import functional as F


def triplet_loss(
    features,
    targets,
    squaredl2=False,
    triplet_selection="all",
    margin=0.2,
    factor=1.,
    normalize=False,
    aggreg="mean",
    harmonic_embeddings=None,
    old_features=None,
    memory_flags=None,
    epoch_percent=None
):
    if normalize:
        features = F.normalize(features, dim=1, p=2)

    if harmonic_embeddings and old_features is not None:
        if harmonic_embeddings["select"] == "old":
            old_features = old_features[memory_flags.eq(1.)]
            old_targets = targets[memory_flags.eq(1.)]
        elif harmonic_embeddings["select"] == "all":
            old_targets = targets
        else:
            raise ValueError(
                "Unknown harmonic embeddings selection {}.".format(harmonic_embeddings["select"])
            )

        features = torch.cat((features, old_features))
        targets = torch.cat((targets, old_targets))

    distance_matrix = _pairwise_distance(features, squared=squaredl2)

    if triplet_selection == "all":
        triplet_losses = _select_all_triplets(
            distance_matrix, _get_triplet_mask(targets), margin=margin
        )
        loss = _aggreg_triplet_losses(triplet_losses, aggreg=aggreg)
    elif triplet_selection == "hard":
        triplet_losses = _select_hardest_triplets(distance_matrix, targets, margin=margin)
        loss = _aggreg_triplet_losses(triplet_losses, aggreg=aggreg)
    elif triplet_selection == "all_hard":
        triplet_losses = _select_all_triplets(
            distance_matrix, _get_triplet_mask(targets), margin=margin
        )
        loss_all = _aggreg_triplet_losses(triplet_losses, aggreg=aggreg)
        triplet_losses_hard = _select_hardest_triplets(distance_matrix, targets, margin=margin)
        loss_hard = _aggreg_triplet_losses(triplet_losses_hard, aggreg=aggreg)

        loss = (1 - epoch_percent) * loss_all + epoch_percent * loss_hard
    else:
        raise ValueError("Unknown triplet selection {}.".format(triplet_selection))

    return factor * loss, _get_per_violated_margin(triplet_losses)

def _select_all_triplets(distance_matrix, triplet_mask, margin=0.2):
    anchor_positive_dist = distance_matrix.unsqueeze(2)
    anchor_negative_dist = distance_matrix.unsqueeze(1)

    if margin == "soft":
        all_triplets = torch.log(1 + torch.exp(anchor_positive_dist - anchor_negative_dist))
    else:
        all_triplets = anchor_positive_dist - anchor_negative_dist + margin

    valid_triplets = all_triplets[triplet_mask]

    pos_triplets = valid_triplets.clamp(min=0.)

    return pos_triplets


def _select_hardest_triplets(distance_matrix, targets, margin=0.2):
    mask_anchor_positive = _get_anchor_positive_triplet_mask(targets).float()
    anchor_positive_dist = mask_anchor_positive * distance_matrix
    hardest_positive_dist = anchor_positive_dist.max(dim=1, keepdims=True)[0]
    mask_anchor_negative = _get_anchor_negative_triplet_mask(targets).float()
    max_anchor_negative_dist = distance_matrix.max(dim=1, keepdims=True)[0]
    anchor_negative_dist = distance_matrix + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist = anchor_negative_dist.min(dim=1, keepdims=True)[0]
    triplet_losses = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.)

    return triplet_losses

def _get_triplet_mask(targets):
    indexes_not_equal = ~torch.eye(len(targets)).bool().to(targets.device)

    i_not_j = indexes_not_equal.unsqueeze(2)
    i_not_k = indexes_not_equal.unsqueeze(1)
    j_not_k = indexes_not_equal.unsqueeze(0)

    distinct_indexes = (i_not_j & i_not_k) & j_not_k

    labels_equal = targets.unsqueeze(0) == targets.unsqueeze(1)
    i_eq_j = labels_equal.unsqueeze(2)
    i_eq_k = labels_equal.unsqueeze(1)

    valid_labels = i_eq_j & (~i_eq_k)

    mask = distinct_indexes & valid_labels

    return mask


def _get_anchor_positive_triplet_mask(targets):
    indexes_not_equal = ~torch.eye(len(targets)).bool().to(targets.device)
    labels_equal = targets.unsqueeze(0) == targets.unsqueeze(1)
    mask = indexes_not_equal & labels_equal

    return mask


def _get_anchor_negative_triplet_mask(targets):
    labels_equal = targets.unsqueeze(0) == targets.unsqueeze(1)

    mask = ~labels_equal

    return mask

def _get_per_violated_margin(triplet_losses):
    nb_total = len(triplet_losses)
    nb_violated = len(triplet_losses[triplet_losses > 1e-8])

    return int(100 * nb_violated / nb_total)


def _aggreg_triplet_losses(triplet_losses, aggreg="mean"):
    if aggreg == "mean":
        return triplet_losses.mean()
    elif aggreg == "max":
        return triplet_losses.max()
    elif aggreg == "adamine":
        nb_not_null = len(triplet_losses[triplet_losses > 0.])
        return triplet_losses.sum() / nb_not_null

    raise ValueError("Unknown aggregation method {}.".format(aggreg))


def _triplet(pos_distance, neg_distance, margin, aggreg="mean"):
    triplets = torch.clamp(margin + pos_distance - neg_distance, min=0.)

    if aggreg == "mean":
        return torch.mean(triplets)
    elif aggreg == "sum":
        return torch.sum(triplets)
    elif aggreg == "adamine":
        return torch.sum(triplets) / max(len(triplets[triplets > 0]), 1)

    raise ValueError("Unknown aggregation method for triplet: {}.".format(aggreg))


def _triplet_facenet_sampling(features, targets, semihard=True, distance="l2squared"):
    pairwise_distances = _dense_distance(features, distance_type=distance)

    anchor_indexes, positive_indexes, negative_indexes = [], [], []

    targets = targets.cpu().numpy()
    for target in set(targets.tolist()):
        indexes = np.where(targets == target)[0].tolist()
        neg_indexes = np.where(targets != target)[0].tolist()

        positive_pairs = list(itertools.combinations(indexes, 2))

        _anchors = torch.tensor([pair[0] for pair in positive_pairs])
        _positives = torch.tensor([pair[1] for pair in positive_pairs])
        if semihard:
            ap_dist = pairwise_distances[_anchors, _positives]

            nb_pos = len(indexes)
            nb_neg = len(targets) - nb_pos

            an_dist = pairwise_distances[torch.tensor(indexes).repeat_interleave(nb_neg, 0),
                                         torch.tensor(neg_indexes).repeat(1, nb_pos)[0]]

            anchors = []
            positives = []
            negatives = []
            for i in range(len(ap_dist)):
                if (ap_dist[i] < an_dist[i]).any():
                    negatives.append(
                        neg_indexes[(an_dist[i] == an_dist[i][ap_dist[i] < an_dist[i]].min()
                                    ).argmax().item()]
                    )

                    positives.append(_positives[i])
                    anchors.append(_anchors[i])
        else:
            negatives = np.random.choice(neg_indexes, size=len(_anchors), replace=False).tolist()
            anchors = _anchors.tolist()
            positives = _positives.tolist()

        assert len(negatives) == len(anchors) == len(positives)
        anchor_indexes.extend(anchors)
        positive_indexes.extend(positives)
        negative_indexes.extend(negatives)

    return torch.tensor(anchor_indexes), torch.tensor(positive_indexes
                                                     ), torch.tensor(negative_indexes)


def _triplet_random_sampling(features, targets):
    anchor_indexes, pos_indexes, neg_indexes = [], [], []
    targets = targets.cpu().numpy()

    for target in targets:
        target_indexes = np.where(target == targets)[0]

        poss = np.random.choice(target_indexes, size=2, replace=len(target_indexes) < 2)
        neg = np.random.choice(np.where(target != targets)[0], size=1)

        anchor_indexes.append(poss[0])
        pos_indexes.append(poss[1])
        neg_indexes.append(neg[0])

    assert len(anchor_indexes) == len(pos_indexes) == len(neg_indexes)

    anchor_indexes = torch.tensor(anchor_indexes)
    pos_indexes = torch.tensor(pos_indexes)
    neg_indexes = torch.tensor(neg_indexes)

    return anchor_indexes, pos_indexes, neg_indexes


def _pairwise_distance(a, squared=False):
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (torch.mm(a, torch.t(a)))
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


def _pair_distance(a, b, distance_type="l2"):
    if distance_type == "l2":
        return F.pairwise_distance(a, b, p=2)
    if distance_type == "l2squared":
        return torch.pow(F.pairwise_distance(a, b, p=2), 2)
    elif distance_type == "l1":
        return F.pairwise_distance(a, b, p=1)
    elif distance_type == "cosine":
        return 1 - torch.cosine_similarity(a, b)

    raise ValueError("Unknown distance type {}.".format(distance_type))
