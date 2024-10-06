import os
import click
from dataloader import process_book_text
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


DIR_WITH_BOOKS = "data/"


def load_text(dir_with_books: str) -> str:
    text = ""
    for book in os.listdir(dir_with_books):
        with open(os.path.join(dir_with_books, book), "r") as f:
            text += process_book_text(f.read(), num_lines_to_skip=200)
    return text


def create_tokenizer(
    corpus: str,
    vocab_size: int,
) -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=100,
    )
    step_size = 10000
    training_corpus_generator = (
        corpus[i : i + step_size] for i in range(0, len(corpus), step_size)
    )
    tokenizer.train_from_iterator(training_corpus_generator, trainer=trainer)
    tokenizer.save(f"tokenizer_{vocab_size}.json")


@click.command()
@click.option(
    "vocab_size",
    "-vs",
    required=True,
    type=int,
    help="The vocabulary size of the tokenizer"
)
def main(vocab_size):
    corpus = load_text(DIR_WITH_BOOKS)
    create_tokenizer(corpus, vocab_size)


if __name__ == "__main__":
    main()
