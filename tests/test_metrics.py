import torch

import pytorch_lightning as pl

from src.metrics import F1_score, Recall_at_k

import pytest


class TestRecall:
    @pytest.mark.parametrize("k,curr_recall", [(1, 0.208), (3, 0.375), (5, 0.5)])
    def test_recall(self, k, curr_recall):
        pl.seed_everything(42)
        selected_positives = 10
        metric = Recall_at_k(k=k, selected_positives=selected_positives)
        batches = [{'anchor': torch.rand(12, 10), 'positive': torch.rand(12, 10)} for _ in range(2)]
        results = []
        for batch in batches:
            results.append(metric(batch))
        recall = metric.compute_metric()
        assert pytest.approx(recall, 0.01) == curr_recall




