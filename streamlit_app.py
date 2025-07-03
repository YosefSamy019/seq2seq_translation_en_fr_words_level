import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import re
import numpy as np

INPUT_SEQUENCE_LEN = 35  # the longest input str can not exceed this numbers of words
OUTPUT_SEQUENCE_LEN = 35  # the longest output str can not exceed this numbers of words

MODEL_ALL_PATH = r'model_all.keras'

X_TOKENIZER_PATH = r'x_tokenizer.pkl'
Y_TOKENIZER_PATH = r'y_tokenizer.pkl'

START_TOKEN = 'START_TOKEN'
END_TOKEN = 'END_TOKEN'
X_TOKENIZER_OOV_TOKEN = '<OOV_X>'
Y_TOKENIZER_OOV_TOKEN = '<OOV_Y>'

UI_CONTAINER_HEIGHT = 400

SPECIAL_CHARS = list("@#$\t\n!%^&*()_+-=[]{}/|;:'\",.<>?`~")


def pkl_load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_resources():
    model = load_model(MODEL_ALL_PATH)
    x_tokenizer = pkl_load_obj(X_TOKENIZER_PATH)
    y_tokenizer = pkl_load_obj(Y_TOKENIZER_PATH)
    return model, x_tokenizer, y_tokenizer


def main():
    st.set_page_config(
        page_title="Sequence-to-Sequence Translation Model with Attention",
        page_icon="🌐",
        layout="wide",
    )

    st.title("🌐 Sequence-to-Sequence Translation Model with Attention")

    st.write(
        'This project builds and trains a sequence-to-sequence (Seq2Seq) model with attention using Keras and TensorFlow. '
        'The model is designed for machine translation (English to French) using a custom dataset.')

    main_cols = st.columns([0.48, 0.04, 0.48])

    with st.progress():
        load_resources()

    with main_cols[0]:
        with st.container(border=True):
            input_text = st.text_area('Enter English Text', height=UI_CONTAINER_HEIGHT)

    with main_cols[2]:
        with st.container(border=True):
            st.text_area('Output French Text', height=UI_CONTAINER_HEIGHT, value=st.session_state.get('OUT'))

    buttons_col = st.columns(5)
    if buttons_col[0].button('Translate', type='primary', use_container_width=True):
        st.session_state['OUT'] = predict_long_sentence(input_text)
        st.rerun()

    if buttons_col[1].button('Clear', type='secondary', use_container_width=True):
        st.session_state['OUT'] = None
        st.rerun()


def predict_long_sentence(sentence):
    segments = []
    output_segments = []

    temp_i = 0
    current_i = 0

    if str(sentence).strip() == '':
        return 'Empty sentence'

    while current_i < len(sentence):
        if sentence[current_i] in SPECIAL_CHARS:
            segments.append(sentence[temp_i:current_i + 1])
            temp_i = current_i + 1
        current_i += 1
    segments.append(sentence[temp_i:current_i])

    segments = list(filter(lambda x: len(x) > 0, segments))

    for segment in segments:
        punc_mark_i = '.'
        if segment[-1] in SPECIAL_CHARS:
            punc_mark_i = segment[-1]
            segment = segment[:-1]

        out_seg_i = predict_pure_text(segment)

        output_segments.append(out_seg_i + punc_mark_i)

    return "".join(output_segments)


def predict_pure_text(input_str):
    model, x_tokenizer, y_tokenizer = load_resources()

    def clean_str(text: str):
        text = str(text).lower().strip()

        text = re.sub(re.compile(r"[^A-Za-z0-9]"), ' ', text)
        text = re.sub(re.compile(r"\s+"), ' ', text)

        return text.strip()

    input_str = clean_str(input_str)
    input_seq = x_tokenizer.texts_to_sequences([input_str])
    input_seq = pad_sequences(input_seq,
                              maxlen=INPUT_SEQUENCE_LEN,
                              padding='post',
                              truncating='post',
                              value=0)

    output_seq = [START_TOKEN]
    counter = 0

    reverse_word_index = {value: key for key, value in y_tokenizer.word_index.items()}

    while output_seq[-1] != END_TOKEN and len(output_seq) < OUTPUT_SEQUENCE_LEN:
        target = ' '.join(output_seq)
        target_int = y_tokenizer.texts_to_sequences([target])
        target_int = pad_sequences(target_int,
                                   maxlen=OUTPUT_SEQUENCE_LEN,
                                   padding='post',
                                   truncating='post',
                                   value=0)

        output_decoder_str_hat = model.predict([input_seq, target_int], verbose=0)
        output_decoder_str_hat = np.argmax(output_decoder_str_hat, axis=-1)
        output_decoder_str_hat = np.array(output_decoder_str_hat.flatten())

        output_seq.append(reverse_word_index.get(output_decoder_str_hat[counter], Y_TOKENIZER_OOV_TOKEN))

        counter += 1

    if len(output_seq) > 0 and output_seq[0] == START_TOKEN:
        del output_seq[0]

    if len(output_seq) > 0 and output_seq[-1] == END_TOKEN:
        del output_seq[-1]

    while len(output_seq) > 1 and output_seq[-1] == Y_TOKENIZER_OOV_TOKEN:
        del output_seq[-1]

    return " ".join(output_seq)


if __name__ == "__main__":
    main()
