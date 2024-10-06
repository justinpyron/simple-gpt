import streamlit as st
import torch
from simple_gpt import SimpleGPT
from tokenizers import Tokenizer


st.set_page_config(page_title="Text Generator", layout="centered")


def highlight_text(
    text_original: str,
    text_full: str,
) -> str:
    text_generated = text_full[len(text_original):]
    highlighted_text_1 = f'<span style="background-color:#f9e79f">{text_original}</span>'
    highlighted_text_2 = f'<span style="background-color:#8e44ad">{text_generated}</span>'
    return f'{highlighted_text_1} {highlighted_text_2}'


# Load model and tokenizer
tokenizer = Tokenizer.from_file("tokenizer_500.json")
model = SimpleGPT(tokenizer.get_vocab_size())
model.load_state_dict(torch.load("saved_models/model_streamlit_app.pt", weights_only=True))


st.title("Simple Generative Text Model")
with st.expander("How it works"):
    st.write("TODO: Brief sentence explaining how it works")
    # It was trained on a corpus of 40 books from project Gutenberg
    # I hope to improve performance by training on a larger dataset


user_input = st.text_area("Enter some text", "")
text = user_input
num_tokens_to_generate = st.slider(
    "Number of tokens to generate",
    min_value=10,
    max_value=100,
    step=10,
    value=10,
)

if st.button("Generate"):
    if text:
        model_input = torch.tensor(tokenizer.encode(text).ids).unsqueeze(dim=0)
        text = tokenizer.decode(model.generate(model_input, new_tokens=num_tokens_to_generate, device="cpu")[0].tolist())
        highlighted_output = highlight_text(user_input, text)
        st.markdown(f'<p style="font-size: 18px;">{highlighted_output}</p>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text before generating.")
if st.button("Clear seed"):
    st.success("Seed cleared!")


# TODO: Add `text` to st.cache. You want to persist this value until you hit a "Reset" button.
# TODO: Improve highlighting of the text. How to deal with light vs dark mode? Hard to have a formatting that works well for both.
