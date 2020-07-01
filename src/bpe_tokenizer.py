import youtokentome as yttm 
from typing import List


class YttmTokenizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = yttm.BPE(model_path)
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_id = 2
        self.eos_id = 3
        self.pad_id = 0
        self.unk_id = 1

    def tokenize(self, sentence: str, add_bos=False, add_eos=False, dropout_prob=0, lower=False) -> List[str]:
        if lower is True:
            sentence = sentence.lower()
        tokenized = self.model.encode([sentence], 
                                      output_type=yttm.OutputType.SUBWORD,
                                      bos=add_bos, eos=add_eos,
                                      dropout_prob=dropout_prob)[0]
        return tokenized

    def encode(self, tokens: List[str]) -> List[int]:
        encoded = []
        for token in tokens:
            encoded.append(self.model.subword_to_id(token))
        return encoded
    
    def decode(self, encoded: List[int]) -> str:
        return self.model.decode(encoded, ignore_ids=[self.pad_id])
    


