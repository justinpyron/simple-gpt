# simple-gpt
From-scratch GPT-style generative language model.

The model uses a decoder-only transformer architecture inspired by _[Attention Is All You Need](https://arxiv.org/abs/1706.03762)_.

# Project Organization
```
├── README.md                   <- Overview
├── app.py                      <- Streamlit web app frontend
├── simple_gpt.py               <- Architecture of underlying transformer model
├── saved_models                <- Weights of trained models
│   └── model_streamlit_app.pt  <- Model used in the app
├── create_tokenizer.py         <- Trains a tokenizer
├── dataloader.py               <- Class for iterating through data
├── trainer.py                  <- Utils for training the model
├── pyproject.toml              <- Poetry config specifying Python environment dependencies
├── poetry.lock                 <- Locked dependencies to ensure consistent installs
```

# Installation
This project uses [Poetry](https://python-poetry.org/docs/) to manage its Python environment.

1. Install Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies
```
poetry install
```

# Usage
A Streamlit web app is the frontend for interacting with the model.

The app can be accessed at https://simple-gpt.streamlit.app

Alternatively, the app can be run locally with
```
poetry run streamlit run app.py
```
