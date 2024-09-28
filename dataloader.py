import tiktoken
import torch
import os


class BookDataLoader:

    def __init__(
        self,
        dir_with_books: str,
        train_fraction: float,
        tokenizer: tiktoken.core.Encoding,
        window_size_default : int,
        batch_size_default : int,
    ) -> None:
        self.dir_with_books =  dir_with_books
        self.train_fraction = train_fraction
        self.tokenizer = tokenizer
        self.window_size_default = window_size_default
        self.batch_size_default = batch_size_default
        self.make_train_val_sets()


    def make_train_val_sets(self):
        self.train = list()
        self.val = list()
        for book in os.listdir(self.dir_with_books):
            with open(os.path.join(self.dir_with_books, book), "r") as f:
                text = f.read()
            text_encoded = self.tokenizer.encode(text)
            threshold = int(self.train_fraction * len(text_encoded))
            self.train += text_encoded[:threshold]
            self.val += text_encoded[threshold:]


    def get_batch(
        self,
        batch_size: int = None,
        window_size: int = None,
        from_val: bool = True,
    ) -> tuple[torch.tensor, torch.tensor]:
        batch_size = self.batch_size_default if batch_size is None else batch_size
        window_size = self.window_size_default if window_size is None else window_size
        data = self.val if from_val else self.train
        idx_sequence_start = torch.randint(len(data) - window_size, (batch_size,))
        x = torch.tensor([data[i: i+window_size] for i in idx_sequence_start])
        y = torch.tensor([data[i+1: i+1+window_size] for i in idx_sequence_start])
        return x, y
