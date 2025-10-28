import streamlit as st
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Import all classifiers
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="Social Media Impact Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# Load models with compatibility handling
@st.cache_resource
def load_models():
    try:
        import joblib
        model = joblib.load('gradient_boosting.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, None
    except Exception as e:
        error_msg = str(e)
        if "MT19937" in error_msg or "BitGenerator" in error_msg:
            return None, None, "numpy_version"
        return None, None, str(e)


model, scaler, error = load_models()

# Display error message if models failed to load
if error:
    if error == "numpy_version":
        st.error("""
        ### 🔧 NumPy Version Incompatibility Detected

        Your pickle files were created with a different NumPy version. Please recreate them with your current environment.
        """)
        st.info(f"""
        **Current Environment:**
        - Python: {sys.version.split()[0]}
        - NumPy: {np.__version__}
        """)
    else:
        st.error(f"Error loading models: {error}")

    st.stop()

# Title and description
st.markdown("<h1>📱 Social Media Impact Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; font-size: 1.1rem; color: #555;'>
    Analyze the impact of social media usage on mental health and well-being using machine learning
    </p>
    """, unsafe_allow_html=True)

# Sidebar for input method selection
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/machine-learning.png", width=80)
    st.title("Input Method")
    input_method = st.selectbox(
        "Choose input method:",
        ["Manual Input", "Model Training & Comparison"],
        help="Select how you want to use the application"
    )

    st.markdown("---")
    st.markdown("### About")
    st.info("""
        This app predicts mental health outcomes based on social media usage patterns.

        **Features:**
        - Real-time predictions
        - Model training & comparison
        - Performance visualization
    """)

# Feature definitions
features = ["Age", "Daily_Usage_Hours", "Sleep_Hours", "Academic_Performance", "Social_Interactions",
            "Exercise_Hours", "Anxiety_Level", "Depression_Level", "Self_Esteem",
            "Parental_Control", "Screen_Time_Before_Bed", "Phone_Checks_Per_Day",
            "Apps_Used_Daily", "Time_on_Social_Media", "Time_on_Gaming",
            "Time_on_Education", "Family_Communication", "Weekend_Usage_Hours"]

feature_descriptions = {
    "Age": "Age of the user (years)",
    "Daily_Usage_Hours": "Average daily social media usage (hours)",
    "Sleep_Hours": "Average sleep per night (hours)",
    "Academic_Performance": "Academic score (0-100)",
    "Social_Interactions": "Daily social interactions count",
    "Exercise_Hours": "Weekly exercise hours",
    "Anxiety_Level": "Self-reported anxiety level (1-10)",
    "Depression_Level": "Self-reported depression level (1-10)",
    "Self_Esteem": "Self-esteem score (1-10)",
    "Parental_Control": "Level of parental monitoring (1-10)",
    "Screen_Time_Before_Bed": "Screen time before sleep (hours)",
    "Phone_Checks_Per_Day": "Phone check frequency per day",
    "Apps_Used_Daily": "Number of apps used daily",
    "Time_on_Social_Media": "Daily social media time (hours)",
    "Time_on_Gaming": "Daily gaming time (hours)",
    "Time_on_Education": "Daily educational app time (hours)",
    "Family_Communication": "Family communication quality (1-10)",
    "Weekend_Usage_Hours": "Weekend social media usage (hours)"
}

# Default values for input
default_values = {
    "Age": 16.0,
    "Daily_Usage_Hours": 5.0,
    "Sleep_Hours": 7.0,
    "Academic_Performance": 75.0,
    "Social_Interactions": 10.0,
    "Exercise_Hours": 3.0,
    "Anxiety_Level": 5.0,
    "Depression_Level": 5.0,
    "Self_Esteem": 6.0,
    "Parental_Control": 5.0,
    "Screen_Time_Before_Bed": 2.0,
    "Phone_Checks_Per_Day": 50.0,
    "Apps_Used_Daily": 10.0,
    "Time_on_Social_Media": 3.0,
    "Time_on_Gaming": 2.0,
    "Time_on_Education": 1.0,
    "Family_Communication": 6.0,
    "Weekend_Usage_Hours": 7.0
}


# Function to plot learning curve
def plot_learning_curve(estimator, X, y, title, ax):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Accuracy Score')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


# Function to plot confusion matrix
def plot_confusion_matrix(cm, ax, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


if input_method == "Manual Input":
    st.markdown("## 📝 Enter User Information")

    # Create tabs for organized input
    tab1, tab2, tab3 = st.tabs(["📱 Usage Patterns", "😊 Well-being", "👨‍👩‍👧 Social & Family"])

    input_data = {}

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            input_data["Age"] = st.number_input(
                "Age",
                min_value=10.0, max_value=25.0,
                value=default_values["Age"],
                help=feature_descriptions["Age"]
            )
            input_data["Daily_Usage_Hours"] = st.number_input(
                "Daily Usage Hours",
                min_value=0.0, max_value=24.0,
                value=default_values["Daily_Usage_Hours"],
                help=feature_descriptions["Daily_Usage_Hours"]
            )
            input_data["Weekend_Usage_Hours"] = st.number_input(
                "Weekend Usage Hours",
                min_value=0.0, max_value=24.0,
                value=default_values["Weekend_Usage_Hours"],
                help=feature_descriptions["Weekend_Usage_Hours"]
            )
            input_data["Time_on_Social_Media"] = st.number_input(
                "Time on Social Media",
                min_value=0.0, max_value=24.0,
                value=default_values["Time_on_Social_Media"],
                help=feature_descriptions["Time_on_Social_Media"]
            )
            input_data["Time_on_Gaming"] = st.number_input(
                "Time on Gaming",
                min_value=0.0, max_value=24.0,
                value=default_values["Time_on_Gaming"],
                help=feature_descriptions["Time_on_Gaming"]
            )

        with col2:
            input_data["Time_on_Education"] = st.number_input(
                "Time on Education",
                min_value=0.0, max_value=24.0,
                value=default_values["Time_on_Education"],
                help=feature_descriptions["Time_on_Education"]
            )
            input_data["Screen_Time_Before_Bed"] = st.number_input(
                "Screen Time Before Bed",
                min_value=0.0, max_value=24.0,
                value=default_values["Screen_Time_Before_Bed"],
                help=feature_descriptions["Screen_Time_Before_Bed"]
            )
            input_data["Phone_Checks_Per_Day"] = st.number_input(
                "Phone Checks Per Day",
                min_value=0.0, max_value=500.0,
                value=default_values["Phone_Checks_Per_Day"],
                help=feature_descriptions["Phone_Checks_Per_Day"]
            )
            input_data["Apps_Used_Daily"] = st.number_input(
                "Apps Used Daily",
                min_value=0.0, max_value=100.0,
                value=default_values["Apps_Used_Daily"],
                help=feature_descriptions["Apps_Used_Daily"]
            )
            input_data["Sleep_Hours"] = st.number_input(
                "Sleep Hours",
                min_value=0.0, max_value=24.0,
                value=default_values["Sleep_Hours"],
                help=feature_descriptions["Sleep_Hours"]
            )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            input_data["Exercise_Hours"] = st.number_input(
                "Exercise Hours (Weekly)",
                min_value=0.0, max_value=50.0,
                value=default_values["Exercise_Hours"],
                help=feature_descriptions["Exercise_Hours"]
            )
            input_data["Anxiety_Level"] = st.slider(
                "Anxiety Level",
                min_value=1, max_value=10,
                value=int(default_values["Anxiety_Level"]),
                help=feature_descriptions["Anxiety_Level"]
            )
            input_data["Depression_Level"] = st.slider(
                "Depression Level",
                min_value=1, max_value=10,
                value=int(default_values["Depression_Level"]),
                help=feature_descriptions["Depression_Level"]
            )

        with col2:
            input_data["Self_Esteem"] = st.slider(
                "Self Esteem",
                min_value=1, max_value=10,
                value=int(default_values["Self_Esteem"]),
                help=feature_descriptions["Self_Esteem"]
            )
            input_data["Academic_Performance"] = st.slider(
                "Academic Performance",
                min_value=0, max_value=100,
                value=int(default_values["Academic_Performance"]),
                help=feature_descriptions["Academic_Performance"]
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            input_data["Social_Interactions"] = st.number_input(
                "Social Interactions (Daily)",
                min_value=0.0, max_value=100.0,
                value=default_values["Social_Interactions"],
                help=feature_descriptions["Social_Interactions"]
            )
            input_data["Family_Communication"] = st.slider(
                "Family Communication Quality",
                min_value=1, max_value=10,
                value=int(default_values["Family_Communication"]),
                help=feature_descriptions["Family_Communication"]
            )

        with col2:
            input_data["Parental_Control"] = st.slider(
                "Parental Control Level",
                min_value=1, max_value=10,
                value=int(default_values["Parental_Control"]),
                help=feature_descriptions["Parental_Control"]
            )

    # Predict button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Generate Prediction", use_container_width=True)

    if predict_button:
        # Create dataframe with correct feature order
        input_df = pd.DataFrame([input_data])[features]

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        # Display prediction
        st.markdown(f"""
            <div class='prediction-box'>
                <h2 style='color: white; margin-bottom: 1rem;'>Prediction Result</h2>
                <h1 style='font-size: 3rem; color: white; margin: 0;'>{prediction[0]:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)

        # Display input summary
        st.markdown("## 📊 Input Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Age", f"{input_data['Age']:.0f} years")
            st.metric("Daily Usage", f"{input_data['Daily_Usage_Hours']:.1f} hrs")
            st.metric("Sleep Hours", f"{input_data['Sleep_Hours']:.1f} hrs")
            st.metric("Exercise", f"{input_data['Exercise_Hours']:.1f} hrs/week")

        with col2:
            st.metric("Anxiety Level", f"{input_data['Anxiety_Level']}/10")
            st.metric("Depression Level", f"{input_data['Depression_Level']}/10")
            st.metric("Self Esteem", f"{input_data['Self_Esteem']}/10")

        with col3:
            st.metric("Academic Performance", f"{input_data['Academic_Performance']}%")
            st.metric("Phone Checks", f"{input_data['Phone_Checks_Per_Day']:.0f}/day")
            st.metric("Family Communication", f"{input_data['Family_Communication']}/10")

else:  # Model Training & Comparison
    st.markdown("## 🤖 Model Training & Comparison")

    st.info(f"""
        Upload a CSV file with features and target column. The CSV should contain these feature columns:
        {', '.join(features)}

        Plus a **target column** with the class labels you want to predict.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file for training", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

            # Show data preview
            with st.expander("📋 View Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)

            # Target column selection
            st.markdown("### 🎯 Select Target Column")
            target_col = st.selectbox(
                "Choose the target/label column:",
                options=[col for col in df.columns if col not in features],
                help="Select the column that contains the class labels you want to predict"
            )

            if target_col:
                # Check if required features are present
                missing_features = [f for f in features if f not in df.columns]

                if missing_features:
                    st.warning(f"⚠️ Missing features: {', '.join(missing_features)}")
                    st.info("The model will train with available features only.")
                    available_features = [f for f in features if f in df.columns]
                else:
                    available_features = features

                # Model selection
                st.markdown("### 🔧 Select Models to Train")

                model_options = {
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Extra Trees": ExtraTreesClassifier(random_state=42),
                    "AdaBoost": AdaBoostClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Support Vector Machine": SVC(random_state=42, probability=True),
                    "Bernoulli Naive Bayes": BernoulliNB()
                }

                selected_models = st.multiselect(
                    "Choose models to compare:",
                    options=list(model_options.keys()),
                    default=["Random Forest", "Gradient Boosting", "Logistic Regression"],
                    help="Select multiple models to train and compare their performance"
                )

                # Test size selection
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
                with col2:
                    random_state = st.number_input("Random State", 0, 100, 42)

                # Train button
                if st.button("🚀 Train Models", type="primary", use_container_width=True):
                    if not selected_models:
                        st.error("Please select at least one model to train!")
                    else:
                        # Prepare data
                        X = df[available_features]
                        y = df[target_col]

                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, stratify=y
                        )

                        # Scale features
                        scaler_new = StandardScaler()
                        X_train_scaled = scaler_new.fit_transform(X_train)
                        X_test_scaled = scaler_new.transform(X_test)

                        st.markdown("---")
                        st.markdown("## 📈 Model Performance Comparison")

                        results = []

                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, model_name in enumerate(selected_models):
                            status_text.text(f"Training {model_name}...")

                            # Get model
                            clf = model_options[model_name]

                            # Train model
                            clf.fit(X_train_scaled, y_train)

                            # Predictions
                            y_pred = clf.predict(X_test_scaled)

                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
                            cv_mean = cv_scores.mean()
                            cv_std = cv_scores.std()

                            results.append({
                                'Model': model_name,
                                'Accuracy': accuracy,
                                'CV Mean': cv_mean,
                                'CV Std': cv_std,
                                'y_pred': y_pred,
                                'clf': clf
                            })

                            # Update progress
                            progress_bar.progress((idx + 1) / len(selected_models))

                        status_text.text("✅ Training completed!")

                        # Results DataFrame
                        results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'CV Mean', 'CV Std']]
                        results_df = results_df.sort_values('Accuracy', ascending=False)

                        # Display results table
                        st.markdown("### 🏆 Model Accuracy Comparison")

                        # Format the dataframe
                        results_display = results_df.copy()
                        results_display['Accuracy'] = results_display['Accuracy'].apply(lambda x: f"{x:.4f}")
                        results_display['CV Mean'] = results_display['CV Mean'].apply(lambda x: f"{x:.4f}")
                        results_display['CV Std'] = results_display['CV Std'].apply(lambda x: f"{x:.4f}")

                        st.dataframe(
                            results_display.style.background_gradient(subset=['Accuracy'], cmap='Greens'),
                            use_container_width=True
                        )

                        # Best model
                        best_model = results_df.iloc[0]['Model']
                        best_accuracy = results_df.iloc[0]['Accuracy']

                        st.success(f"🥇 **Best Model:** {best_model} with accuracy of **{best_accuracy:.4f}**")

                        # Detailed results for each model
                        st.markdown("---")
                        st.markdown("## 📊 Detailed Model Analysis")

                        for result in results:
                            with st.expander(f"🔍 {result['Model']} - Detailed Analysis"):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("#### Performance Metrics")
                                    st.metric("Test Accuracy", f"{result['Accuracy']:.4f}")
                                    st.metric("CV Mean Score", f"{result['CV Mean']:.4f}")
                                    st.metric("CV Std Dev", f"{result['CV Std']:.4f}")

                                    # Classification report
                                    st.markdown("#### Classification Report")
                                    report = classification_report(y_test, result['y_pred'], output_dict=True)
                                    report_df = pd.DataFrame(report).transpose()
                                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

                                with col2:
                                    # Confusion Matrix
                                    st.markdown("#### Confusion Matrix")
                                    fig, ax = plt.subplots(figsize=(6, 5))
                                    cm = confusion_matrix(y_test, result['y_pred'])
                                    plot_confusion_matrix(cm, ax, f"{result['Model']}")
                                    st.pyplot(fig)
                                    plt.close()

                                # Learning Curve
                                st.markdown("#### Learning Curve")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                plot_learning_curve(
                                    result['clf'], X_train_scaled, y_train,
                                    f"Learning Curve - {result['Model']}", ax
                                )
                                st.pyplot(fig)
                                plt.close()

                        # Download best model
                        st.markdown("---")
                        st.markdown("### 💾 Save Best Model")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📥 Download Best Model", use_container_width=True):
                                import joblib

                                best_clf = [r for r in results if r['Model'] == best_model][0]['clf']
                                joblib.dump(best_clf, 'best_model.pkl')
                                joblib.dump(scaler_new, 'scaler_new.pkl')
                                st.success(f"✅ {best_model} saved as 'best_model.pkl'")

                        with col2:
                            if st.button("📊 Download Results CSV", use_container_width=True):
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Click to Download",
                                    data=csv,
                                    file_name="model_comparison_results.csv",
                                    mime="text/csv"
                                )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            import traceback

            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>Built with Streamlit | Powered by Gradient Boosting</p>
    </div>
    """, unsafe_allow_html=True)
