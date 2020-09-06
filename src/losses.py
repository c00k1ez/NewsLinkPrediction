import torch
import torch.nn.functional as F
from typing import Dict, Union


class NegariveSampler:
    def __init__(self, margin=None, number_of_neg_samples=1):
        self.margin = margin
        self.number_of_neg_samples = number_of_neg_samples
        assert number_of_neg_samples == 1 and margin is None or number_of_neg_samples >= 1 and margin is not None

    def negative_samples(
            self,
            data: Dict[str, Union[torch.FloatTensor, torch.cuda.FloatTensor]]
            ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        anchor = data['anchor'] # shape [batch_size, embedding_size]
        pos = data['positive'] # shape [batch_size, embedding_size]
        batch_size, embedding_size = pos.shape
        anchor = anchor / anchor.norm(dim=1).unsqueeze(1).repeat(1, embedding_size)
        pos = pos / pos.norm(dim=1).unsqueeze(1).repeat(1, embedding_size)
        dist_matrix = torch.mm(anchor, pos.t()).type_as(pos) # shape [batch_size, batch_size]
        #dist_matrix = torch.cdist(anchor, pos).cpu() # shape [batch_size, batch_size]
        
        mask = torch.ones(anchor.shape[0], pos.shape[0]).type_as(pos) - torch.diag(torch.ones(pos.shape[0])).type_as(pos) # shape [batch_size, batch_size]
        dist_matrix = dist_matrix * mask
        if self.margin is None:
            neg_samples = dist_matrix.max(dim=1)[1] # choose the closest examples as negative samples
        else:
            margin = torch.FloatTensor([[self.margin]]).repeat(batch_size, batch_size).type_as(pos)
            neg_samples = torch.abs(dist_matrix - margin).type_as(pos)
            neg_samples = neg_samples * mask
            neg_samples[neg_samples == 0.] = float('-inf')
            neg_samples = neg_samples.max(dim=1)[1]
        neg_samples = neg_samples.unsqueeze(1).repeat(1, pos.shape[1])
        # neg_samples shape [batch_size, embedding_size]
        neg = torch.gather(pos, index=neg_samples, dim=0)
        neg = neg.type_as(pos)
        return neg



class OnlineTripletLoss(torch.nn.Module):
    def __init__(self, margin: int, sampler=NegariveSampler, reduction='mean', return_logits=False):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        assert reduction in ['mean', 'none', 'sum']
        self.reduction = reduction
        self.return_logits = return_logits
        self.sampler = sampler(margin)

    def _distance(self, t1, t2):
        return torch.cosine_similarity(t1, t2)

    # model_outputs = {'anchor': torch.FloatTensor, 'positive': torch.FloatTensor}
    def forward(self, model_outputs: Dict[str, Union[torch.FloatTensor, torch.cuda.FloatTensor]]):
        anchor = model_outputs['anchor'] # shape [batch_size, embedding_size]
        pos = model_outputs['positive'] # shape [batch_size, embedding_size]
        
        batch_size, emb_dim = pos.size()

        neg = self.sampler.negative_samples(model_outputs) # shape [batch_size, embedding_size]

        margin = torch.FloatTensor([self.margin]).repeat(batch_size) # shape [batch_size]
        margin = margin.type_as(pos)

        anchor_pos_dist = self._distance(anchor, pos)
        anchor_neg_dist = self._distance(anchor, neg)
        
        loss = F.relu(margin + anchor_pos_dist - anchor_neg_dist) # shape [batch_size,]

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss 


class OnlineBCELoss(torch.nn.Module):
    def __init__(self, margin=None, sampler=NegariveSampler, reduction='mean'):
        super(OnlineBCELoss, self).__init__()
        self.sampler = sampler(margin)
        self.reduction = reduction
        assert reduction in ['mean', 'sum', 'none']
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def _distance(self, t1, t2):
        return torch.cosine_similarity(t1, t2)

    # model_outputs = {'anchor': torch.FloatTensor, 'positive': torch.FloatTensor}
    def forward(self, model_outputs: Dict[str, Union[torch.FloatTensor, torch.cuda.FloatTensor]]):
        anchor = model_outputs['anchor'] # shape [batch_size, embedding_size]
        pos = model_outputs['positive'] # shape [batch_size, embedding_size]
        neg = self.sampler.negative_samples(model_outputs) # shape [batch_size, embedding_size]

        batch_size, emb_dim = pos.size()

        anchor_pos_dist = self._distance(anchor, pos) # shape [batch_size,]
        anchor_neg_dist = self._distance(anchor, neg) # shape [batch_size,]
        dists = torch.cat([anchor_pos_dist, anchor_neg_dist]).type_as(pos)
        labels = torch.FloatTensor([1] * batch_size + [0] * batch_size).type_as(pos)
        return self.bce(dists, labels)


class SoftmaxLoss(torch.nn.Module):
    '''
    Read more about this loss here https://arxiv.org/abs/1902.08564
    '''
    def __init__(self, margin=None, norm_vectors=True, reduction='mean'):
        super(SoftmaxLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.norm_vectors = norm_vectors
        assert reduction in ['mean', 'sum', 'none']
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

    # model_outputs = {'anchor': torch.FloatTensor, 'positive': torch.FloatTensor}
    def forward(self, model_outputs: Dict[str, Union[torch.FloatTensor, torch.cuda.FloatTensor]]):
        anchor = model_outputs['anchor'] # shape [batch_size, embedding_size]
        pos = model_outputs['positive'] # shape [batch_size, embedding_size]

        batch_size, emb_dim = pos.size()
        
        if self.norm_vectors:
            anchor = anchor / anchor.norm(dim=1).unsqueeze(1).repeat(1, emb_dim)
            pos = pos / pos.norm(dim=1).unsqueeze(1).repeat(1, emb_dim)

        scores = anchor @ pos.t()

        if self.margin is not None:
            scores -= (torch.eye(batch_size) * self.margin).type_as(pos)
        
        targets = torch.arange(batch_size).type_as(pos).long()
        loss = self.criterion(scores, targets)

        return loss



