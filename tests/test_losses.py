import pytest
import torch

from src.losses import NegariveSampler, OnlineBCELoss, OnlineTripletLoss, SoftmaxLoss


class TestLosses:
    def test_sampler(self):
        sampler = NegariveSampler()
        bs = 5
        emb_size = 20
        test_data = {"anchor": torch.rand((bs, emb_size)), "positive": torch.rand((bs, emb_size))}
        neg = sampler.negative_samples(test_data)
        assert list(neg.shape) == [bs, emb_size]

    def test_online_triplet_loss(self):
        sampler = NegariveSampler
        criterion = OnlineTripletLoss(margin=1, sampler=sampler)
        bs = 5
        emb_size = 20
        test_data = {"anchor": torch.rand((bs, emb_size)), "positive": torch.rand((bs, emb_size))}
        t = criterion(test_data)
        assert (type(t.item()) == float) and (torch.isnan(t).item() is False)

    def test_online_bce_loss(self):
        sampler = NegariveSampler
        criterion = OnlineBCELoss(sampler=sampler, number_of_neg_samples=2)
        bs = 5
        emb_size = 20
        test_data = {"anchor": torch.rand((bs, emb_size)), "positive": torch.rand((bs, emb_size))}
        t = criterion(test_data)

        assert (type(t.item()) == float) and (torch.isnan(t).item() is False)

    def test_semihard_sampler(self):
        sampler = NegariveSampler(0.3)
        bs = 5
        emb_size = 20
        test_data = {"anchor": torch.rand((bs, emb_size)), "positive": torch.rand((bs, emb_size))}
        neg = sampler.negative_samples(test_data)
        assert list(neg.shape) == [bs, 1, emb_size]

    def test_softmax_loss(self):
        criterion = SoftmaxLoss(margin=0.5)
        bs = 5
        emb_size = 20

        test_data = {"anchor": torch.rand((bs, emb_size)), "positive": torch.rand((bs, emb_size))}
        t = criterion(test_data)

        assert (type(t.item()) == float) and (torch.isnan(t).item() is False)
