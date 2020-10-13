import torch
import torch.nn.functional as F
from typing import Dict

'''
Do not use this metrics for DP / DDP mode!
'''

class F1_score:
    def __init__(self, average='weighted'):
        assert average in ['weighted',]
        self.average = average

    '''
    matr is confusion matrix
    '''
    def __call__(self, matr: torch.Tensor):
        tn, fp, fn, tp = matr[0, 0], matr[0, 1], matr[1, 0], matr[1, 1]
        precision_0, recall_0 = tn / (tn + fn), tn / (tn + fp)
        precision_1, recall_1 = tp / (tp + fp), tp / (tp + fn)
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
        weight_0 = (tn + fp) / (tn + fp + fn + tp)
        weight_1 = (tp + fn) / (tn + fp + fn + tp)
        total_f1 = weight_0 * f1_0 + weight_1 * f1_1
        return total_f1, [f1_0, f1_1]


class Recall_at_k:
    def __init__(self, k=1, selected_positives=10):
        assert k in [1, 3, 5]
        self.k = k
        self.selected_positives = selected_positives
        self.tp = []
        self.fn = []

    '''
    batch = {'anchor': [batch_size, output_dim], 'positive': [batch_size, output_dim]}
    '''
    def __call__(self, batch: Dict[str, torch.Tensor]):
        batch_size, output_dim = batch['anchor'].shape
        assert batch['anchor'].shape == batch['positive'].shape
        assert batch_size-1 > self.selected_positives
        anchor, positive = batch['anchor'], batch['positive']
        # dist shape [batch_size(anchor), batch_size(positive)]
        # dist is cosine similarity between every achor and every positive
        dist = F.normalize(anchor) @ F.normalize(positive).t()
        # truth positives for each anchors with shape [batch_size, 1]
        ground_truth = torch.diag(dist).unsqueeze(1)
        mask = torch.ones(batch_size, batch_size) - torch.diag(torch.ones(batch_size))
        mask = mask.type_as(dist)
        # new distance tensor without ground truth values
        dist = dist[mask == 1.].view(batch_size, batch_size - 1)

        # generate permutation
        idx = torch.randperm(dist.nelement()).type_as(dist).long()
        # permute distance tensor
        dist = dist.view(-1)[idx].view(batch_size, batch_size - 1)
        # select 9 random positives for each anchor, shape [batch_size, 9]
        dist = dist[:, :self.selected_positives - 1]
        # shape [batch_size, 10]
        # at each row, first element is ground truth distance
        dist = torch.cat([ground_truth, dist], dim=1)
        # get probabilities
        dist = torch.softmax(dist, dim=-1)
        topk_inds = dist.topk(self.k, dim=-1)[1]
        # true positive is count of non zero elements at topk_inds
        # false negative = batch_size - non zero
        tp = topk_inds[topk_inds == 0].shape[0]
        fn = batch_size - tp
        self.tp.append(tp)
        self.fn.append(fn)
        return [tp, fn]

    def compute_metric(self):
        tp_at_k = sum(self.tp)
        fn_at_k = sum(self.fn)
        recall_at_k = tp_at_k / (tp_at_k + fn_at_k)
        return recall_at_k





