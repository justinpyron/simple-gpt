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
    highlighted_text_1 = f'<span style="background-color:#f9e79f; color:black">{text_original}</span>'
    highlighted_text_2 = f'<span style="background-color:#8e44ad; color:white">{text_generated}</span>'
    return f'{highlighted_text_1} {highlighted_text_2}'


# Load model and tokenizer
tokenizer = Tokenizer.from_file("tokenizer_500.json")
model = SimpleGPT(tokenizer.get_vocab_size())
model.load_state_dict(torch.load("saved_models/model_streamlit_app.pt", weights_only=True))


st.title("Simple Generative Text Model")
with st.expander("How it works"):
    st.markdown(
        "This app demos a simple generative text model. The model has a "
        "[simple transformer architecture](https://github.com/justinpyron/simple-gpt/blob/main/simple_gpt.py). "
        "It was trained on a corpus of 40 books from [Project Gutenberg](https://www.gutenberg.org/)."
        "\n\nAt the moment, it produces mostly correct English words. "
        "However, generated words are mostly random with little connection to the previous context. "
        "In addition, the grammar is questionable, and the influence of the training corpus "
        "is quite evident. For example, you'll probably be able to notice that the training "
        "set included Tolstoy and Dostoevsky."
        "\n\nThe plan is to improve the shortcomings above by training on a larger and more diverse dataset."
    )

if "text" not in st.session_state:
    st.session_state.text = ""
user_input = st.text_area("Enter some text", "")
if len(st.session_state.text) == 0:
    st.session_state.text = user_input

num_tokens_to_generate = st.slider(
    "Number of tokens to generate",
    min_value=10,
    max_value=50,
    step=10,
    value=30,
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate", type="primary", use_container_width=True):
        if len(st.session_state.text) > 0:
            model_input = torch.tensor(tokenizer.encode(st.session_state.text).ids).unsqueeze(dim=0)
            generated_tokens = model.generate(
                model_input,
                n_new_tokens=num_tokens_to_generate,
                temperature=0.1,
                device="cpu"
            )[0].tolist()
            st.session_state.text = tokenizer.decode(generated_tokens)
        else:
            st.warning("Enter some text first")
with col2:
    if st.button("Reset", use_container_width=True):
        st.session_state.text = user_input

if len(st.session_state.text) > 0:
    highlighted_output = highlight_text(user_input, st.session_state.text)
    st.markdown(f'<p style="font-size: 18px;">{highlighted_output}</p>', unsafe_allow_html=True)
