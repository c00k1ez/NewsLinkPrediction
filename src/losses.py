import torch
import torch.nn.functional as F
from typing import Dict, Union


class SemiHardNegariveSampler:
    def __init__(self, margin=None):
        self.margin = margin

    def negative_samples(self, 
                         data: Dict[str, Union[torch.FloatTensor, torch.cuda.FloatTensor]]
                        ) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]:
        anchor = data['anchor'] # shape [batch_size, embedding_size]
        pos = data['positive'] # shape [batch_size, embedding_size]
        dist_matrix = torch.cdist(anchor, pos).cpu() # shape [batch_size, batch_size]
        
        mask = torch.ones(anchor.shape[0], pos.shape[0]) - torch.diag(torch.ones(pos.shape[0])) # shape [batch_size, batch_size]
        dist_matrix = dist_matrix * mask
        neg_samples = dist_matrix.max(dim=1)[1] # choose the closest examples as negative samples
        neg_samples = neg_samples.unsqueeze(1).repeat(1, pos.shape[1])
        # neg_samples shape [batch_size, embedding_size]
        neg = torch.gather(pos.cpu(), index=neg_samples, dim=0)
        neg = neg.type_as(pos)
        return neg



class OnlineTripletLoss(torch.nn.Module):
    def __init__(self, margin: int, sampler=SemiHardNegariveSampler(), reduction='mean', return_logits=False):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        assert reduction in ['mean', 'none', 'sum']
        self.reduction = reduction
        self.return_logits = return_logits
        self.sampler = sampler

    def _distance(self, t1, t2):
        return (t1 - t2).norm(dim=1)

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

