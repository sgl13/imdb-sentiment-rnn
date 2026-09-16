# 🎬 IMDB Sentiment Analysis using RNN

A deep learning project that classifies IMDB movie reviews as **Positive** or **Negative** using a **Recurrent Neural Network (Simple RNN / LSTM)** built with TensorFlow/Keras, with word embeddings and an interactive app for real-time predictions.

---

## 📌 Overview

This project walks through the complete workflow of building a sentiment classifier on the IMDB movie reviews dataset:

- Loading and preprocessing the IMDB dataset (tokenization, padding)
- Building word embeddings to represent review text numerically
- Training a **Recurrent Neural Network (RNN)** for binary sentiment classification
- Evaluating model performance on unseen test data
- Deploying the trained model through a simple web app for live predictions

---

## 📂 Project Structure

```
imdb-sentiment-rnn/
│
├── embedding.ipynb          # Exploring word embeddings on sample text
├── app.py                   # Web app (Streamlit) for real-time sentiment prediction
├── prediction.py            # Loads the trained model and runs a sample prediction
├── main.py                  # PyCharm scaffold file (not part of the pipeline)
├── simple_rnn_imdb.h5       # Trained RNN model weights (Keras 2 / HDF5 format)
├── simple_rnn_imdb.keras    # Trained RNN model weights (Keras 3 native format, used by app.py)
├── requirements.txt         # Python dependencies
├── Screenshots/             # App screenshots used in this README
├── .gitignore
└── README.md                # Project documentation
```

> Note: the notebook that actually trains the model (referred to below as the "training notebook") isn't checked into the repo yet — only `embedding.ipynb` (an embeddings walkthrough) and the already-trained model files are present. Add it back in, or update this section once it's committed.

---

## 🧠 Model Architecture

| Layer            | Config                          | Output Shape     | Params        |
|-------------------|----------------------------------|-------------------|---------------|
| Embedding         | vocab size 10,000 → 128-dim vectors | (None, 100, 128) | 1,280,000 |
| SimpleRNN         | 128 units, tanh activation       | (None, 128)       | 32,896        |
| Dense (Sigmoid)   | 1 unit, outputs probability of "Positive" | (None, 1) | 129           |

**Total params:** 1,313,027 (~5.01 MB) — verified directly from `simple_rnn_imdb.h5`.

**Dataset:** IMDB Movie Reviews (via `keras.datasets.imdb`), containing 50,000 labeled reviews (25,000 train / 25,000 test).

**Preprocessing steps** (matching what `app.py` uses for inference):
1. Keep only the top 10,000 most frequent words in the vocabulary
2. Convert reviews into integer sequences
3. Pad/truncate sequences to a fixed length of 100 tokens (pre-padding)
4. Feed sequences into the Embedding → SimpleRNN → Dense pipeline

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sgl13/imdb-sentiment-rnn.git
   cd imdb-sentiment-rnn
   ```

2. **Create and activate a virtual environment** *(recommended)*
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Train the model
```bash
jupyter notebook
```
- `embedding.ipynb` → understand how embeddings represent text
- *(training notebook not yet in this repo — see note in Project Structure above)*

### Test a prediction from the command line
```bash
python prediction.py
```
Loads `simple_rnn_imdb.h5` and prints the sentiment/score for a sample review.

### Run the web app
```bash
streamlit run main.py
```
Then open the local URL shown in your terminal, type in a movie review, and get an instant **Positive / Negative** prediction with a confidence score.

**Example:**

| Input Review | Sentiment | Prediction Score |
|---|---|---|
| "This movie was fantastic! The acting was great and the plot was thrilling." | Positive | 0.7989 |

---

## 🖼️ Screenshots

**Dashboard view**

![Dashboard view](https://raw.githubusercontent.com/sgl13/imdb-sentiment-rnn/55662b88224255e3fa4d1303d5fdcf5770d2ef43/Screenshots/Dashboard%20view%20IMDB%20Movie%20Review%20Sentiment_Analysis%202026-09-15%20at%206.01.41%E2%80%AFPM.png)

**Prediction — Example 1**

![Prediction example 1](https://raw.githubusercontent.com/sgl13/imdb-sentiment-rnn/55662b88224255e3fa4d1303d5fdcf5770d2ef43/Screenshots/IMDB%20Movie%20Review%20Sentiment%20Analysis_prediction%202026-09-15%20at%206.04.07%E2%80%AFPM.png)

**Prediction — Example 2**

![Prediction example 2](https://raw.githubusercontent.com/sgl13/imdb-sentiment-rnn/55662b88224255e3fa4d1303d5fdcf5770d2ef43/Screenshots/IMDB%20Movie%20Review%20Sentiment%20Analysis_prediction%20%202026-09-15%20at%206.02.24%E2%80%AFPM.png)

---

## 📊 Results

| Metric              | Score        |
|----------------------|-------------|
| Training Accuracy (best epoch, ~5/10) | 88.95% |
| Validation Accuracy (best epoch, ~5/10) | 81.14% |
| Validation Loss (best epoch, ~5/10) | 0.4547 |
| Loss Function         | Binary Crossentropy |
| Optimizer             | Adam        |

**Training history:**

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 61.08% | 0.6402 | 76.88% | 0.4897 |
| 2 | 72.51% | 0.5412 | 61.94% | 0.6443 |
| 3 | 76.75% | 0.4849 | 78.38% | 0.4841 |
| 4 | 83.69% | 0.3718 | **81.94%** | 0.4596 |
| 5 | 88.95% | 0.2774 | 81.14% | **0.4547** |
| 6 | 78.25% | 0.4762 | 74.66% | 0.5528 |
| 7 | 85.43% | 0.3394 | 78.50% | 0.5138 |
| 8 | 87.33% | 0.3065 | 78.58% | 0.5256 |

> ⚠️ Training stopped at epoch 8 of 10 — almost certainly an `EarlyStopping` callback monitoring `val_loss`, since validation loss peaked (best) at epoch 5 (0.4547) and didn't improve for the following epochs. Best validation accuracy (81.94%) came at epoch 4, closely tracking the standalone test accuracy reported earlier (81.22%).

---

## 🛠️ Tech Stack

- **Python 3.x**
- **TensorFlow / Keras** – model building & training
- **NumPy / Pandas** – data handling
- **Streamlit** – interactive web app for predictions

---

## 🔮 Future Improvements

- Replace SimpleRNN with **LSTM / GRU / Bidirectional LSTM** for better long-range context
- Use **pre-trained embeddings** (GloVe / Word2Vec) instead of learned embeddings
- Add an **attention layer** to visualize which words influenced the prediction
- Deploy the app publicly (Streamlit Community Cloud / HuggingFace Spaces)

---

## 📄 License

This project is open source. Add your preferred license (e.g. MIT) here.

---

## 🙌 Acknowledgements

- IMDB Dataset — available directly via `tensorflow.keras.datasets.imdb`
- Inspired by the structure of [ANN-Classification-Churn](https://github.com/sgl13/ANN-Classification-Churn)

---

## 👨‍💻 Author

**Shivakumar G L**  
*IMDB Movie Review Sentiment Analysis using RNN*  
*"Where reviews become predictions."*  
Python | Deep Learning | NLP | Streamlit

[GitHub](https://github.com/sgl13) · [LinkedIn](https://linkedin.com/in/shivakumar-g-l-5a6450259)