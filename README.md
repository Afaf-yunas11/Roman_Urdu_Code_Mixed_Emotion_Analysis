# Roman_Urdu_Code_Mixed_Emotion_Analysis
A research based natural language project focusing on detecting emotions(joy,sadness,neutral,sarcasm,anger)inside a text given by user

#  Logistic Regression (Using td-idf vector as input)

**Hybrid Logistic Regression Model**  is used to detect emotions in **Roman Urdu** text (Urdu written in English script). Unlike standard models, this approach combines **TF-IDF Vectorization** with **Manual Dictionary Feature Engineering** to successfully detect complex emotions like **Sarcasm**, which traditional models often miss.

##  Model Overview
*   **Language:** Roman Urdu (e.g., *"Ye product bohat bakwas hai"*)
*   **Classes:** 5 (Joy, Sadness, Sarcasm, Anger, Neutral)
*   **Model Architecture:** Logistic Regression (`solver='lbfgs'`)
*   **Input Features:** 
    1.  TF-IDF Vectors (N-grams, Stop-word removal)
    2.  **Custom Feature:** Emotion Dictionary Counts (Hybrid Approach)

---

##  Visualizations

### 1. Confusion Matrix
*Visualizes where the model predicted correctly vs. where it got confused.*

<img width="2373" height="2115" alt="confusion_matrix (1)" src="https://github.com/user-attachments/assets/f0d55ea0-653d-483f-8c52-4988f13c573d" />
our_pie_chart.png)


### 2. Class Performance (F1-Score)
*A comparison of how well the model performed for each specific emotion.*

![Pie Chart](path_to_y<img width="2539" height="1349" alt="F1-score-barchart" src="https://github.com/user-attachments/assets/40b10d9e-b858-402d-bdf1-5b7e7d4ab67d" />



### 3. Dataset Distribution
*The balance of data across the 5 emotions in the test set.*



---

## 📈 Results & Analysis

The model was evaluated on a **Held-Out Test Set** of **5,278 sentences**.

| Class | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Sarcasm** | 0.52 | **0.63** | **0.57** |  **Best Performer** |
| **Neutral** | 0.31 | 0.66 | 0.42 |  High Recall |
| **Anger** | 0.29 | 0.40 | 0.34 |  Moderate |
| **Joy** | 0.51 | 0.10 | 0.17 | ❌ Confused with Neutral |
| **Sadness** | 0.32 | 0.07 | 0.11 | ❌ Confused with Anger |
| **Overall Accuracy** | | | **36.09%** | |

###  Key Insights

#### 1. The Sarcasm Breakthrough 🚀
The model achieved a **Recall of 63% for Sarcasm**, which is exceptionally high for a sentiment model. 
*   **Why?** The **Hybrid Approach** worked. By explicitly counting specific words (like *"wah"*, *"mubarak"*, *"shabash"*) via the Manual Feature Engineering step, the model could distinguish sarcasm even when the sentence contained positive words.

#### 2. The "Anger vs. Joy" Fix 🛠️
Initially, the model confused Anger with Joy due to words like *"mubarak"* being used ironically. After refining the dictionary and normalization, the model now correctly identifies **Anger** distinct from Joy, though it sometimes confuses it with Sadness due to overlapping negative vocabulary (e.g., *"dard"*, *"bura"*).

#### 3. The Neutral Vacuum 🕳️
The model is aggressive at predicting **Neutral** (66% Recall). However, it tends to classify mild "Joy" sentences (e.g., *"Mousam acha hai"*) as Neutral because they lack intense emotional keywords.

---

##  How It Works (The Hybrid Architecture)

The model does not rely on text alone. It uses a **Stacked Input Matrix**:

1.  **Text Preprocessing:**
    *   Normalization (e.g., `bohat` / `bht` / `bhut` → `bohat`)
    *   Custom Roman Urdu Stop-word removal (removing `hai`, `tha`, `ka` but keeping `nahi`, `kyun`).
2.  **Feature Stacking:**
    *   **Vector A:** TF-IDF Score (Statistical weight of words).
    *   **Vector B:** Manual Counts (How many "Joy", "Sad", "Sarcasm", "Anger" dictionary words appear?).
3.  **Classification:**
    *   These vectors are glued together (`hstack`) and fed into Logistic Regression.

---

##  How to Run the Model

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
    # [Code for counting matches goes here...]
    
    # Stack & Predict
    final_input = hstack([vec, manual_features])
    return model.predict(final_input)[0]

# 3. Test
print(predict_emotion("Wah bhai kya kamaal service hai")) 
# Output: Sarcasm
Future Improvements

Deep Learning (BERT/LSTM): To better understand context and resolve the "Sadness vs. Anger" confusion.

Enhanced Dictionary: Adding intensity-based words to distinguish between "Happy" (Joy) and "Okay" (Neutral).

Data Balancing: Increasing the volume of "Sadness" data to improve its recall.

Created by Afaf Yunas


