import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from typing import Dict, Any
from app.config import GENRE_COLUMNS

class MovieRobertaDataset(Dataset):
    """
    Dataset class for RoBERTa. Handles tokenization and tensor conversion.
    """
    def __init__(self,
                 csv_file_path: str,
                 tokenizer: RobertaTokenizerFast,
                 max_len: int,
                 is_test_dataset: bool = False) -> None:
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test_dataset = is_test_dataset

        print(f"Loading data from {csv_file_path}...")
        try:
            self.data = pd.read_csv(csv_file_path, engine='python', on_bad_lines='skip', encoding_errors='ignore')
        except Exception:
            # Fallback for older pandas versions or different separators
            self.data = pd.read_csv(csv_file_path, sep=',', on_bad_lines='skip')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.data.iloc[index]
        text = str(row['plot_synopsis'])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.is_test_dataset:
            # Assuming first column is ID if it's a test dataset without headers
            # Adjust based on actual CSV structure
            item['movie_id'] = str(row[0])
        else:
            labels = row[GENRE_COLUMNS].values.astype(float)
            item['labels'] = torch.tensor(labels, dtype=torch.float32)

        return item
