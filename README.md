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
├── simplernn.ipynb          # Building & training the Simple RNN model
├── prediction.ipynb         # Loading the trained model and testing predictions
├── main.py                  # Web app (Streamlit) for real-time sentiment prediction
├── simple_rnn_imdb.h5       # Trained RNN model weights
├── requirements.txt         # Python dependencies
├── runtime.txt              # Runtime/environment version
├── .gitignore
└── README.md                # Project documentation
```

> Note: File names may vary slightly — adjust this section to match your repo exactly.

---

## 🧠 Model Architecture

| Layer            | Description                                         |
|-------------------|------------------------------------------------------|
| Embedding Layer   | Converts word indices into dense vector representations |
| SimpleRNN Layer   | Captures sequential/contextual patterns in the review text |
| Dense (Sigmoid)   | Outputs a probability between 0 (negative) and 1 (positive) |

**Dataset:** IMDB Movie Reviews (via `keras.datasets.imdb`), containing 50,000 labeled reviews (25,000 train / 25,000 test).

**Preprocessing steps:**
1. Keep only the top *N* most frequent words in the vocabulary
2. Convert reviews into integer sequences
3. Pad/truncate sequences to a fixed length
4. Feed sequences into the Embedding → RNN → Dense pipeline

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
Open and run the notebooks in order:
```bash
jupyter notebook
```
- `embedding.ipynb` → understand how embeddings represent text
- `simplernn.ipynb` → train and save the RNN model
- `prediction.ipynb` → test predictions on custom text

### Run the web app
```bash
streamlit run main.py
```
Then open the local URL shown in your terminal, type in a movie review, and get an instant **Positive / Negative** prediction with a confidence score.

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
| Training Accuracy    | ~ *fill in* |
| Test Accuracy        | ~ *fill in* |
| Loss Function         | Binary Crossentropy |
| Optimizer             | Adam        |

*(Replace with your actual metrics/plots — consider adding accuracy/loss curves here.)*

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