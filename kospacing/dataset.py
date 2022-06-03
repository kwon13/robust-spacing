from typing import Callable, List, Tuple
from torch.utils.data import Dataset

from utils import load_slot_labels


class CorpusDataset(Dataset):
    def __init__(self, data_path: str, transform: Callable[[List, List], Tuple]):
        self.sentences = []
        self.transform = transform
        self.slot_labels = load_slot_labels()

        self._load_data(data_path)

    def _load_data(self, data_path: str):
        
        with open(data_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            self.sentences = [line.split() for line in lines if line!='\n']

    def _get_tags(self, sentence: List[str]) -> List[str]:
        """문장에 대해 띄어쓰기 tagging을 한다.
        character 단위로 분리하여 IOBES tagging을 한다.
        
        각 tags의 뜻
        I : 안을 의미하는 token
        O : pad를 의미하는 token
        B : 시작을 의미하는 token
        E : 끝을 의미하는 token
        S : 단일 문자를 의미하는 token
        
        """

        all_tags = []
        for word in sentence:
            if len(word) == 1:
                all_tags.append("S")
            elif len(word) > 1:
                for i, c in enumerate(word):
                    if i == 0:
                        all_tags.append("B")
                    elif i == len(word) - 1:
                        all_tags.append("E")
                    else:
                        all_tags.append("I")
        return all_tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = "".join(self.sentences[idx])
        tags = self._get_tags(self.sentences[idx])
        tags = [self.slot_labels.index(t) for t in tags]

        (
            input_ids,
            slot_labels,
            attention_mask,
            token_type_ids,
        ) = self.transform(sentence, tags)

        return input_ids, slot_labels, attention_mask, token_type_ids
