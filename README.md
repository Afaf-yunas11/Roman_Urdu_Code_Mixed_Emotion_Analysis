
#  Roman Urdu Code-Mixed Emotion Analysis

##  Project Overview
This is a research-based Natural Language Processing (NLP) project focusing on detecting emotions (**Joy, Sadness, Neutral, Sarcasm, Anger**) inside Roman Urdu text (Urdu written in English script) provided by the user.

###  The Model: Hybrid Logistic Regression
We implemented a **Hybrid Logistic Regression Model** specifically designed for the complexities of Roman Urdu. Unlike standard models, this approach combines **TF-IDF Vectorization** with **Manual Dictionary Feature Engineering** to successfully detect complex emotions like **Sarcasm**, which traditional models often miss.

*   **Language:** Roman Urdu (e.g., *"Ye product bohat bakwas hai"*)
*   **Classes:** 5 (Joy, Sadness, Sarcasm, Anger, Neutral)
*   **Model Architecture:** Logistic Regression (`solver='lbfgs'`)
*   **Input Features:** 
    1.  **TF-IDF Vectors:** Captures n-grams and vocabulary weights.
    2.  **Custom Features:** Emotion Dictionary Counts (Hybrid Approach) to flag specific emotional triggers.

---

##  Visualizations

### 1. Confusion Matrix
*Visualizes where the model predicted correctly vs. where it got confused.*
![Confusion Matrix](https://github.com/user-attachments/assets/f0d55ea0-653d-483f-8c52-4988f13c573d)

### 2. Class Performance (F1-Score)
*A comparison of how well the model performed for each specific emotion.*
![F1 Score Chart](https://github.com/user-attachments/assets/40b10d9e-b858-402d-bdf1-5b7e7d4ab67d)

---

##  Results & Analysis

The model was evaluated on a **Held-Out Test Set** of **5,278 sentences**.

| Class | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Sarcasm** | 0.52 | **0.63** | **0.57** |  **Best Performer** |
| **Neutral** | 0.31 | 0.66 | 0.42 | High Recall |
| **Anger** | 0.29 | 0.40 | 0.34 | Moderate |
| **Joy** | 0.51 | 0.10 | 0.17 | Confused with Neutral |
| **Sadness** | 0.32 | 0.07 | 0.11 |  Confused with Anger |
| **Overall Accuracy** | | | **36.09%** | |

###  Key Insights

#### 1. The Sarcasm Breakthrough 
The model achieved a **Recall of 63% for Sarcasm**, which is exceptionally high for a sentiment model. 
*   **Why?** The **Hybrid Approach** worked. By explicitly counting specific words (like *"wah"*, *"mubarak"*, *"shabash"*) via the Manual Feature Engineering step, the model could distinguish sarcasm even when the sentence contained positive words.

#### 2. The "Anger vs. Joy" Fix 
Initially, the model confused Anger with Joy due to words like *"mubarak"* being used ironically. After refining the dictionary and normalization, the model now correctly identifies **Anger** distinct from Joy, though it sometimes confuses it with Sadness due to overlapping negative vocabulary (e.g., *"dard"*, *"bura"*).

#### 3. The Neutral Vacuum 🕳️
The model is aggressive at predicting **Neutral** (66% Recall). However, it tends to classify mild "Joy" sentences (e.g., *"Mousam acha hai"*) as Neutral because they lack intense emotional keywords.

---

##  Master Comparison Table & Roadmap

This project involves a comparative analysis of multiple models ranging from classical machine learning to modern deep learning transformers. Below is the leaderboard tracking the performance of all models implemented by the group.

| Member | Model | Role | Accuracy | F1-Score | Training Time | Inference Speed (sent/sec) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Member 3** | Naïve Bayes | Baseline | --- | --- | --- | --- |
| **Member 2** | **Logistic Regression** | **Baseline** | **36.09%** | **0.31** | **1.63 sec** | **~2,425,234** ⚡ |
| **Member 1** | SVM | Baseline | --- | --- | --- | --- |
| **Member 1** | DistilBERT | Main | --- | --- | --- | --- |
| **Member 2** | XLM-RoBERTa | Main | --- | --- | --- | --- |
| **Member 3** | Hybrid (SVM + Embeddings) | Main | --- | --- | --- | --- |

###  Baseline Analysis: Logistic Regression
As the Feature Engineering specialist's baseline, the **Hybrid Logistic Regression** model has established the foundational benchmarks:

1.  **The Speed Benchmark ⚡:** The model achieved a staggering **~2.4 Million sentences per second**, highlighting the efficiency of classical mathematical models compared to Neural Networks.
2.  **The Accuracy Floor 📉:** With **36.09% Accuracy**, this model sets the minimum bar. Any advanced model (DistilBERT, XLM-R) **MUST** exceed these metrics to justify their computational cost.

---
## ⚙️ How to Run the Model

Requires `scikit-learn`, `pandas`, `numpy`, and `scipy`.

```python
import joblib
from scipy.sparse import hstack

# 1. Load Saved Models
model = joblib.load('roman_urdu_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# 2. Define Prediction Function
def predict_emotion(sentence):
    # Transform Text
    vec = tfidf.transform([sentence])
    
    # Count Dictionary Matches (Joy, Sad, Sarcasm, Anger)
    # [Note: You need the counting logic here same as training]
    # counts = get_manual_features(sentence) 
    
    # Stack & Predict
    # final_input = hstack([vec, counts])
    # return model.predict(final_input)[0]
    pass # Placeholder

# 3. Test Example
# print(predict_emotion("Wah bhai kya kamaal service hai")) 
# Output: Sarcasm


```

 # Future Improvements

To improve upon the baseline results, the following strategies have been identified for future iterations:

Deep Learning (BERT/LSTM):

Moving beyond simple word counting to understand context. This is crucial to resolve the confusion between "Sadness" (passive negative) and "Anger" (active negative).

## Enhanced Dictionary:

Refining the manual feature engineering to include intensity-based words. This will help distinguish between mild positivity ("Theek hai" - Neutral) and genuine happiness ("Zabardast" - Joy).

## Data Balancing:

The current dataset is heavily skewed towards Sarcasm and Anger detection. Increasing the volume of "Sadness" data is required to improve its Recall, which is currently the model's weakest point.

## Created by Afaf Yunas



