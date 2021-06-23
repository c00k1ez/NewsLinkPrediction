from typing import Dict, List, Tuple, Union

import torch
from transformers import BertTokenizer


class PairsDataset(torch.utils.data.Dataset):
    def __init__(
        self, data, tokenizer: BertTokenizer, news_pad_len: int = 512, broadcast_pad_len: int = 512, mode="train"
    ):
        self.mode = mode
        self.data = self._preprocess(data)
        self.tokenizer = tokenizer
        self.news_pad_len = news_pad_len
        self.broadcast_pad_len = broadcast_pad_len
        assert mode in ["train", "test_full", "test"]

    # generate only positive samples
    def _train_mode(self, data):
        cleaned_data = []
        match_ids = list(data.keys())
        for match_id in match_ids:
            samples = []  # list of dicts {'broadcast': str, 'news': str}
            broadcast = data[match_id]["broadcast"]
            news = data[match_id]["news"]
            # merge broadcast to one string
            merged_br = "\n".join([broadcast[key] for key in broadcast.keys()])
            # transform data sample to pairs
            for news_id in news.keys():
                news_sample = news[news_id]["title"] + "\n" + news[news_id]["body"]
                samples.append({"broadcast": merged_br, "news": news_sample})
            cleaned_data.extend(samples)
        return cleaned_data

    # generate pos and neg samples to calculate metrics
    def _test_full_mode(self, data):
        cleaned_data = []
        match_ids = list(data.keys())
        for match_id in match_ids:
            samples = []  # list of dicts {'broadcast': str, 'news': str}
            broadcast = data[match_id]["broadcast"]
            merged_br = "\n".join([broadcast[key] for key in broadcast.keys()])
            for another_match_id in match_ids:
                news = data[another_match_id]["news"]
                for news_id in news.keys():
                    news_sample = news[news_id]["title"] + "\n" + news[news_id]["body"]
                    samples.append(
                        {
                            "broadcast": merged_br,
                            "news": news_sample,
                            "label": 1 if match_id == another_match_id else 0,
                        }
                    )
            cleaned_data.extend(samples)

        return cleaned_data

    def _preprocess(self, data) -> List[Dict[str, str]]:
        cleaned_data = []
        if self.mode == "train" or self.mode == "test":
            cleaned_data.extend(self._train_mode(data))
        else:
            cleaned_data.extend(self._test_full_mode(data))
        return cleaned_data

    def _preprocess_text(self, text: List[str], pad_len: int) -> Tuple[List[int], List[int]]:

        if len(text) > pad_len - 1:
            text = [self.tokenizer.cls_token] + text[: pad_len - 1]
        else:
            text = [self.tokenizer.cls_token] + text
        text_mask = [1] * len(text) + [0] * (pad_len - len(text))
        text = text + [self.tokenizer.pad_token] * (pad_len - len(text))
        assert len(text) == len(text_mask) == pad_len
        text = self.tokenizer.encode(text, add_special_tokens=False)
        return text, text_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx: int) -> Dict[str, torch.LongTensor]:
        sample = self.data[indx]
        broadcast, news = sample["broadcast"], sample["news"]
        broadcast = self.tokenizer.tokenize(broadcast)
        news = self.tokenizer.tokenize(news)
        broadcast, broadcast_mask = self._preprocess_text(broadcast, self.broadcast_pad_len)
        news, news_mask = self._preprocess_text(news, self.news_pad_len)

        broadcast = torch.LongTensor(broadcast)
        broadcast_mask = torch.LongTensor(broadcast_mask)

        news = torch.LongTensor(news)
        news_mask = torch.LongTensor(news_mask)

        returned_sample = {
            "broadcast": broadcast,
            "broadcast_mask": broadcast_mask,
            "news": news,
            "news_mask": news_mask,
        }

        if self.mode == "test_full":
            label = sample["label"]
            label = torch.LongTensor([label])
            returned_sample["label"] = label

        return returned_sample
