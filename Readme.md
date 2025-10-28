# Social Media Impact Predictor & ML Model Comparison

**A Streamlit app for predicting and comparing machine learning models on social media and mental health datasets.**

---

## 🚀 Features

- **Manual Prediction:** Enter individual user data through a structured UI and get prediction from a pretrained model.
- **Batch Model Training & Comparison:** Upload a CSV dataset, select the target column, and compare popular classifiers side-by-side.
- **Supports 9 Classifiers:**
  - RandomForestClassifier
  - GradientBoostingClassifier
  - ExtraTreesClassifier
  - AdaBoostClassifier
  - LogisticRegression
  - DecisionTreeClassifier
  - KNeighborsClassifier
  - SVC (Support Vector Machine)
  - BernoulliNB (Naive Bayes)
- **Visual Diagnostics:**
  - Accuracy score, cross-validation metrics
  - Classification report (precision, recall, F1-score)
  - Confusion matrix heatmaps
  - Learning curves for overfitting/underfitting analysis
- **Data Preview and Validation**
- **Export Best Model and Result Table as CSV**

---

## 📦 Installation

1. **Clone the repo:**
    ```
    git clone https://github.com/yourusername/teen-addiction-model-app.git
    cd teen-addiction-model-app
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

    Required main libraries:
    - streamlit
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
    - joblib

3. **Run the app:**
    ```
    streamlit run app.py
    ```

---

## 📑 Usage

### Manual Input Mode

1. Choose **Manual Input** from sidebar.
2. Fill user data in the tabs.
3. Click **Generate Prediction**.
4. See the result and summary metrics.

### Model Training & Comparison Mode

1. Select **Model Training & Comparison** from sidebar.
2. Upload a CSV with features and a target column.
3. Select the label column and pick one or more models to compare.
4. Click **Train Models**.
5. Analyze model table, confusion matrices, and learning curves.
6. Download best model and results as needed.

**CSV file must contain:**  
- Feature columns: `Age, Daily_Usage_Hours, Sleep_Hours, Academic_Performance, Social_Interactions, Exercise_Hours, Anxiety_Level, Depression_Level, Self_Esteem, Parental_Control, Screen_Time_Before_Bed, Phone_Checks_Per_Day, Apps_Used_Daily, Time_on_Social_Media, Time_on_Gaming, Time_on_Education, Family_Communication, Weekend_Usage_Hours`
- One target column (class/label you want to predict)

---

## 🖼 Screenshots

| Main Page                | Model Comparison Table      | Confusion Matrix & Learning Curve |
|------------------------- |---------------------------|-----------------------------------|
| ![main](screenshots/main.png) | ![table](screenshots/model_table.png) | ![plots](screenshots/cm_lc.png) |

---

## 💡 Notes

- For **manual prediction**, pretrained models (“gradient_boosting.pkl” & “scaler.pkl”) must be present.
- For **batch/model training**, all classifiers run from scratch and results are compared visually.
- Works best on datasets where the target label is categorical (classification task).

---

## 📚 License & Credits

Open-source for educational/research use.  
Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/).

---
