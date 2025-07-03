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

The code starts by importing:

* Data handling: `numpy`, `pandas`
* Plotting: `matplotlib`, `seaborn`
* Text processing: `re`, `json`, `pickle`
* Neural networks: Keras layers (`LSTM`, `Embedding`, `Dense`, `Bidirectional`, etc.), models, and utils

### 2️⃣ **Constants & Config**

Defines constants such as:

* Max input/output lengths
* Embedding dimensions
* Context vector length (LSTM hidden units)
* Tokenizer configs (vocabulary size, OOV tokens)
* Paths for saving models, tokenizers, and outputs

### 3️⃣ **Dataset Preparation**

* Loads English-French sentence pairs from `fra.txt`
* Cleans text (lowercase, removes special characters, trims spaces)
* Visualizes sentence lengths
* Adds start (`START_TOKEN`) and end (`END_TOKEN`) tokens to target sentences

### 4️⃣ **Tokenization**

* Tokenizers for input (English) and output (French) text
* Converts text to sequences of integers
* Pads sequences to fixed lengths

### 5️⃣ **Model Architecture**

* **Encoder:**

  * Embedding + Bidirectional LSTM + LSTM (with state outputs)

* **Decoder:**

  * Embedding + two LSTM layers (with layer normalization and residual connection)

* **Attention:**

  * Computes attention weights (Dot product)
  * Combines context vectors with decoder outputs
  * Dense layer predicts next token

* Final model combines encoder, decoder, and attention for training

### 6️⃣ **Training**

* Compiled with `rmsprop` optimizer and `sparse_categorical_crossentropy` loss
* Trains on padded sequences
* Includes callbacks: early stopping and model checkpoint

### 7️⃣ **Inference Models**

* Separates encoder, decoder, and attention into independent models for step-by-step decoding during inference
* Defines a decoding loop that generates tokens until `END_TOKEN` is produced

---

## 🌐 Streamlit Web App

### 📂 Files

* `model_all.keras`: Saved combined model for inference
* `x_tokenizer.pkl`, `y_tokenizer.pkl`: Tokenizers for input and output text

### 🖥 App Features

* Input: English text via a multi-line text area
* Output: Translated French text (auto-populated)
* Buttons:

  * **Translate** — performs translation
  * **Clear** — resets output

### 🧠 Translation Logic

* Splits input by special characters into segments
* Each segment is cleaned, tokenized, and padded
* Feeds into the model iteratively, appending predicted tokens
* Stops on `END_TOKEN` or max length

---

## ⚙️ How to Run

1️⃣ **Train the model** (in Colab or local Jupyter):

```python
# Run your notebook code (provided earlier)
```

2️⃣ **Save trained model and tokenizers**:

```python
# This is done in the notebook already using pickle and model.save()
```

3️⃣ **Run Streamlit app**:

```bash
streamlit run app.py
```

*(where `app.py` is your Streamlit script)*

---

## 📈 Outputs

* Model training loss/accuracy plots
* Trained model files: `.keras`, `.pkl`, `.json`
* Streamlit app interface for real-time translation

---

## 📝 Notes

* The inference decoder uses the training model directly (for simplicity), but ideally, you can further refactor it to use the separated inference encoder/decoder.
* Attention scores help the decoder focus on relevant input tokens at each decoding step.