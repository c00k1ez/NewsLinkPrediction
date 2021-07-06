import json

import torch
import transformers

from src.dataset import PairsDataset, collate_fn


class TestDataset:
    def test_train_dataset(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        with open("./data/data_sample.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = PairsDataset(data, tokenizer, news_pad_len=30, broadcast_pad_len=30)
        assert len(ds) == 101

    def test_data_sample(self):
        with open("./data/data_sample.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        news_len = []
        for key in data.keys():
            sample = data[key]
            news_len.append(len(sample["news"]))
        neg_pairs = sum([(len(data) - 1) * n for n in news_len])
        pos_pairs = sum(news_len)
        assert (neg_pairs == 404) and (pos_pairs == 101)

    def test_val_data(self):
        with open("./data/test.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        news_len = []
        for key in data.keys():
            sample = data[key]
            news_len.append(len(sample["news"]))
        neg_pairs = sum([(len(data) - 1) * n for n in news_len])
        pos_pairs = sum(news_len)
        assert (neg_pairs == 2184) and (pos_pairs == 156)

    def test_val_dataset(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        with open("./data/data_sample.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = PairsDataset(data, tokenizer, mode="test_full", news_pad_len=30, broadcast_pad_len=30)
        pos_cnt = 0
        for i in range(len(ds)):
            pos_cnt += ds[i]["label"][0].item()
        assert (len(ds) == 505) and (pos_cnt == 101)

    def test_collate_fn(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        with open("./data/data_sample.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = PairsDataset(data, tokenizer, mode="train", news_pad_len=512, broadcast_pad_len=512)
        loader = torch.utils.data.DataLoader(ds, batch_size=100, collate_fn=collate_fn)
        batch = next(iter(loader))
        assert batch["news"].size(-1) == 478
