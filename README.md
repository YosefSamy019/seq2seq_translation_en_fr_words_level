---

# 🌐 Sequence-to-Sequence Translation Model with Attention

This project builds, trains, and serves a **Sequence-to-Sequence (Seq2Seq)** translation model (English ➡ French) using TensorFlow/Keras, attention mechanism, and a Streamlit web app for interactive translation.

---

## 📌 Overview

* **Model type:** Seq2Seq with attention (encoder-decoder architecture)
* **Input:** English sentences (max 35 tokens)
* **Output:** French sentences (max 35 tokens)
* **Libraries:** TensorFlow, Keras, Streamlit, Numpy, Pandas, Seaborn
* **Deployment:** Streamlit app for real-time translation

---

## 🚀 Workflow

### 1️⃣ **Importing Libraries**

Loads essential Python and deep learning libraries for data handling, visualization, and model building.

### 2️⃣ **Constants & Config**

Defines model configuration:

* Input/output sequence lengths
* Embedding dimensions
* Context vector length (LSTM units)
* Tokenizer vocab size, special tokens
* Paths for saving models, tokenizers, and outputs

### 3️⃣ **Dataset Preparation**

* Loads English-French sentence pairs from `fra.txt`
* Cleans text (removes special characters, lowercases)
* Adds `START_TOKEN` and `END_TOKEN`
* Visualizes sentence length distributions

### 4️⃣ **Tokenization**

* Creates and fits tokenizers
* Converts text to integer sequences
* Applies padding

### 5️⃣ **Model Architecture**

* **Encoder:** Embedding → BiLSTM → LSTM
* **Decoder:** Embedding → stacked LSTMs with residuals + normalization
* **Attention:** dot-product attention to align input and output sequences
* **Training model:** combines encoder, decoder, and attention

### 6️⃣ **Training**

* Compiles with `rmsprop` + `sparse_categorical_crossentropy`
* Trains with checkpointing and early stopping

### 7️⃣ **Inference**

* Defines separate inference encoder, decoder, attention models
* Implements step-by-step token generation

### 8️⃣ **Streamlit App**

* Accepts English input
* Segments + cleans text
* Predicts French translation interactively

---

## 💻 Demo

👉 **Try the live demo:**
[🌐 Streamlit Seq2Seq Translator (English ➡ French)](https://seq2seqtranslationenfrwordslevel-9snqxcwdzqz39iw3u63aud.streamlit.app/)


---

## 🖼️ Images

### 📌 App Run

![APP](run_image.png)

### 📌 Training Model (Expanded Nested Layers)

![Training Model Nested](training_model_show_nested.png)

---

## ⚙️ How to Run Locally

1️⃣ **Train model**

```python
# Run provided notebook code
```

2️⃣ **Save model/tokenizers**

```python
# Already done in notebook using pickle and model.save()
```

3️⃣ **Run Streamlit app**

```bash
streamlit run app.py
```

---

## 📈 Outputs

* Saved models: `model_all.keras`, `model_encoder.keras`, `model_decoder.keras`, `model_attention.keras`
* Tokenizers: `x_tokenizer.pkl`, `y_tokenizer.pkl`
* Token JSONs: `x_tokenizer_tokens_allowed.json`, `y_tokenizer_tokens_allowed.json`
* Visualizations: model architecture PNGs, loss/accuracy plots

---

## 📝 Notes

⚠ The inference could be further improved with beam search or other decoding strategies for better translation quality.

---
