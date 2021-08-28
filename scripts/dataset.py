from torch.utils.data import Dataset
import torch


def convert_examples_to_features(text, tokenizer, max_len):
    tok = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        padding='max_length',
    )
    return tok


class CLRPDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        self.excerpts = self.data.excerpt.tolist()
        if not is_test:
            self.targets = self.data.target.tolist()

        self.tokenizer = tokenizer
        self.is_test = is_test
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if not self.is_test:
            excerpt = self.excerpts[item]
            label = self.targets[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer, self.max_len
            )
            return {
                'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float),
            }
        else:
            excerpt = self.excerpts[item]
            features = convert_examples_to_features(
                excerpt, self.tokenizer, self.max_len
            )
            return {
                'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long),
            }