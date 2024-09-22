import tiktoken
import torch
import os

# TODO: handle migrating data to different device


class BookData:

    def __init__(
        self,
        dir_with_books: str,
        train_fraction: float,
        tokenizer: tiktoken.core.Encoding,
    ) -> None:
        self.dir_with_books =  dir_with_books
        self.train_fraction = train_fraction
        self.tokenizer = tokenizer
        self.make_train_val()


    def make_train_val(self):
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
        batch_size: int,
        window_size: int,
        from_train: bool = True
    ) -> tuple[torch.tensor, torch.tensor]:
        data = self.train if from_train else self.val
        idx_sequence_start = torch.randint(len(data) - window_size, size=(batch_size,))
        x = torch.tensor([data[i: i+window_size] for i in idx_sequence_start])
        y = torch.tensor([data[i+1: i+1+window_size] for i in idx_sequence_start])
        return x, y
