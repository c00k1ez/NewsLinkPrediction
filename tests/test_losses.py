import torch 

import pytest 

from src.losses import OnlineTripletLoss, SemiHardNegariveSampler, OnlineBCELoss

class TestLosses:
    def test_sampler(self):
        sampler = SemiHardNegariveSampler()
        bs = 5
        emb_size = 20
        test_data = {
            'anchor': torch.rand((bs, emb_size)),
            'positive': torch.rand((bs, emb_size))
        }
        assert list(sampler.negative_samples(test_data).shape) == [bs, emb_size]

    def test_online_triplet_loss(self):
        sampler = SemiHardNegariveSampler()
        criterion = OnlineTripletLoss(margin=1, sampler=sampler)
        bs = 5
        emb_size = 20
        test_data = {
            'anchor': torch.rand((bs, emb_size)),
            'positive': torch.rand((bs, emb_size))
        }
        t = criterion(test_data)
        assert (type(t.item()) == float) and (torch.isnan(t).item() is False)

    def test_online_bce_loss(self):
        sampler = SemiHardNegariveSampler()
        criterion = OnlineBCELoss(sampler=sampler)
        bs = 5
        emb_size = 20
        test_data = {
            'anchor': torch.rand((bs, emb_size)),
            'positive': torch.rand((bs, emb_size))
        }
        t = criterion(test_data)

        assert (type(t.item()) == float) and (torch.isnan(t).item() is False)