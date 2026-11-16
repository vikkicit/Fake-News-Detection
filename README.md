# 📰 TRUTH MATRIX — AI-POWERED FAKE NEWS DETECTION

\<p align="center">   \<img src="[https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)" />   \<img src="[https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge\&logo=scikitlearn)" />   \<img src="[https://img.shields.io/badge/Frontend-Next.js-black?style=for-the-badge&logo=next.js](https://img.shields.io/badge/Frontend-Next.js-black?style=for-the-badge\&logo=next.js)" />   \<img src="[https://img.shields.io/badge/3D-Three.js-white?style=for-the-badge&logo=threedotjs](https://img.shields.io/badge/3D-Three.js-white?style=for-the-badge\&logo=threedotjs)" /> \</p>

### An Intelligent News Classification System Powered by Machine Learning & NLP

## 🌐 Overview

Truth Matrix is an AI-powered Fake News Detection application that analyzes news-style articles using Natural Language Processing and Machine Learning.

It combines a Python-based classification engine with a cinematic 3D web interface to provide an interactive news analysis experience.

## ✨ Features

- 🧠 Machine Learning-based news classification
- 🔍 NLP text preprocessing
- 📊 TF-IDF feature extraction
- 🤖 Logistic Regression classifier
- 🌐 Cinematic 3D interface
- 📈 Prediction confidence and evaluation metrics
- ⚡ Streamlit application

## 🛠️ Technology Stack

| Category         | Technologies                      |
| ---------------- | --------------------------------- |
| Programming      | Python, TypeScript, JavaScript    |
| Machine Learning | Scikit-learn, Logistic Regression |
| NLP              | TF-IDF                            |
| Frontend         | Next.js, React, Tailwind CSS      |
| 3D Graphics      | Three.js, React Three Fiber       |
| Animation        | GSAP, Framer Motion               |
| Application      | Streamlit                         |

## 🧠 ML Architecture

```text
NEWS ARTICLE INPUT
        |
        v
TEXT PREPROCESSING
        |
        v
TF-IDF VECTORIZATION
        |
        v
LOGISTIC REGRESSION
        |
        v
PREDICTION ENGINE
        |
        v
CLASSIFICATION OUTPUT
```

## 📊 Model Performance

| Metric          |   Result |
| --------------- | -------: |
| Accuracy        | 99.9574% |
| ROC-AUC         |   0.9998 |
| True Negatives  |    3,521 |
| False Positives |        1 |
| False Negatives |        2 |
| True Positives  |    3,520 |

*These results are from the existing test dataset and do not establish real-world fact-checking accuracy.*

## 📁 Project Structure

```text
Fake-News-Detection/
├── dataset/
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   └── predict.py
├── models/
├── reports/
├── tests/
├── web/
│   └── src/
│       ├── app/
│       └── components/
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/vikkicit/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create Environment

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

Open `http://localhost:8501`

### 5. Run Frontend

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`

## ⚠️ Dataset Limitation

The current `Fake.csv` and `True.csv` files represent two training classes. The `True.csv` file contains AI-generated synthetic fictional news-style articles rather than independently verified genuine news.

The model learns patterns from these dataset classes. A TRUE-LABELED prediction does not prove that an article is factually correct.

## 🔮 Future Enhancements

- Trusted news source integration
- External fact-checking APIs
- Source credibility analysis
- Explainable AI
- Multilingual classification
- Improved real-time backend integration

## 👨‍💻 Developer

**Vicky**

GitHub: [@vikkicit](https://github.com/vikkicit)

Project: [Truth Matrix — Fake News Detection](https://github.com/vikkicit/Fake-News-Detection)

---

### 🚀 TRUTH MATRIX

*Transforming News Analysis Through Artificial Intelligence.*
