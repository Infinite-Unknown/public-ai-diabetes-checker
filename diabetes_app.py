import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="centered")
st.title("🧠 AI Diabetes Risk Assessment")

# Set default mode in session state
if "mode" not in st.session_state:
    st.session_state["mode"] = "Use Prediction"

# Sidebar mode selection linked to session state
mode = st.sidebar.radio("Choose Mode", ["Use Prediction", "Train a Model", "Compare Models"], index=["Use Prediction", "Train a Model", "Compare Models"].index(st.session_state["mode"]))

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model function
def save_model(model, feature_names, model_name):
    with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, f"{model_name}_features.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

# Load all saved models
def list_models():
    return sorted([f[:-4] for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") and not f.endswith("_features.pkl")])

# ------------------------ TRAIN MODE ------------------------
if mode == "Train a Model":
    st.header("🔧 Train a Model with Your CSV")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    model_choice = st.selectbox("Select a model", ["Random Forest", "Logistic Regression", "Decision Tree"])
    model_name = st.text_input("Enter a name for your model")

    if uploaded_file and model_name:
        try:
            df = pd.read_csv(uploaded_file)

            st.write("📊 Preview of uploaded data:")
            st.dataframe(df.head())

            # Replace 0s with NaNs in selected columns
            cols_with_zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)

            # Show missing value heatmap
            st.subheader("🧼 Missing Values (0s Replaced as NaN)")
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.heatmap(df[cols_with_zero_as_missing].isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
            st.pyplot(fig)

            # Fill missing values with median
            df.fillna(df.median(numeric_only=True), inplace=True)

            # Show correlation matrix
            st.subheader("📊 Correlation Matrix")
            corr_fig, corr_ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=corr_ax)
            st.pyplot(corr_fig)

            # Ensure Outcome column exists
            if "Outcome" not in df.columns:
                st.error("❌ Your CSV must include an 'Outcome' column.")
            else:
                if st.button("Train Model"):
                    X = df.drop(columns=["Outcome"])
                    y = df["Outcome"]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    if model_choice == "Random Forest":
                        model = RandomForestClassifier()
                    elif model_choice == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    else:
                        model = DecisionTreeClassifier()

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    save_model(model, X.columns.tolist(), model_name)

                    st.success(f"✅ Model '{model_name}' trained and saved.")
                    st.info(f"📈 Accuracy on test set: **{accuracy:.2%}**")

                    # Confusion Matrix
                    st.subheader("📌 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    cm_fig, cm_ax = plt.subplots()
                    disp.plot(ax=cm_ax, cmap="Blues")
                    st.pyplot(cm_fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ------------------------ PREDICT MODE ------------------------
if mode == "Use Prediction":
    st.header("📈 Predict Diabetes Risk")

    available_models = list_models()
    if not available_models:
        st.warning("⚠️ No models found. Please train one first.")
        if st.button("➡️ Go to Train a Model"):
            st.session_state["mode"] = "Train a Model"
            st.rerun()
        st.stop()

    selected_model = st.selectbox("Select a model to use", available_models)

    try:
        with open(os.path.join(MODEL_DIR, f"{selected_model}.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, f"{selected_model}_features.pkl"), "rb") as f:
            feature_names = pickle.load(f)

        st.subheader("Enter Medical Data")

        user_input = []
        placeholders = {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 85,
            "BMI": 28.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 33
        }

        units = {
            "Glucose": "(mg/dL)",
            "BloodPressure": "(mm Hg)",
            "SkinThickness": "(mm)",
            "Insulin": "(mu U/ml)",
            "BMI": "(kg/m²)",
            "DiabetesPedigreeFunction": "(score)",
            "Age": "(years)",
            "Pregnancies": "(months)"
        }

        for feature in feature_names:
            label = feature  # reverted back to show "Pregnancies"
            unit = units.get(feature, "")
            default = placeholders.get(feature, 0)
            if feature in ["Pregnancies", "Age"]:
                val = st.number_input(f"{label} {unit}", value=int(default), step=1, format="%d")
            else:
                val = st.number_input(f"{label} {unit}", value=float(default), format="%.2f")
            user_input.append(val)

        if st.button("Predict"):
            try:
                input_array = np.array(user_input).reshape(1, -1)
                result = model.predict(input_array)
                if result[0] == 1:
                    st.success("🩺 You are likely to have diabetes.")
                else:
                    st.success("✅ You are not likely to have diabetes.")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
# ------------------------ COMPARE MODE ------------------------
if mode == "Compare Models":
    st.header("📊 Compare Saved Models")

    available_models = list_models()
    if not available_models:
        st.warning("⚠️ No models available. Please train at least one model first.")
    else:
        selected_models = st.multiselect("Select models to compare", available_models, default=available_models)

        st.markdown("### 🧠 Select Primary Comparison Metric")
        metric_choice = st.selectbox(
            "Choose how to determine best model",
            [
                "F1 Score (Good for imbalanced classes)",
                "Accuracy (Simple overall correctness)",
                "AUC (Good for ranking/patient risk separation)"
            ]
        )

        test_file = st.file_uploader("(Optional) Upload test data with 'Outcome' column", type=["csv"])

        if test_file:
            try:
                df = pd.read_csv(test_file)
                cols_with_zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
                df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)
                df.fillna(df.median(numeric_only=True), inplace=True)

                if "Outcome" not in df.columns:
                    st.error("❌ Test data must include 'Outcome' column.")
                else:
                    X = df.drop(columns=["Outcome"])
                    y = df["Outcome"]

                    rows = []
                    auc_curves = []

                    for model_name in selected_models:
                        try:
                            with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "rb") as f:
                                model = pickle.load(f)
                            y_pred = model.predict(X)

                            # Probabilities for AUC/ROC
                            if hasattr(model, "predict_proba"):
                                y_proba = model.predict_proba(X)[:, 1]
                                fpr, tpr, _ = roc_curve(y, y_proba)
                                auc_score = auc(fpr, tpr)
                                auc_curves.append((model_name, fpr, tpr, auc_score))
                            else:
                                auc_score = None

                            rows.append({
                                "Model": model_name,
                                "Accuracy": float(accuracy_score(y, y_pred)),
                                "Precision": float(precision_score(y, y_pred, zero_division=0)),
                                "Recall": float(recall_score(y, y_pred, zero_division=0)),
                                "F1 Score": float(f1_score(y, y_pred, zero_division=0)),
                                "AUC": auc_score
                            })

                            with st.expander(f"Confusion Matrix: {model_name}"):
                                cm = confusion_matrix(y, y_pred)
                                fig, ax = plt.subplots()
                                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                                disp.plot(ax=ax, cmap="Blues")
                                st.pyplot(fig)

                        except Exception as ex:
                            st.error(f"❌ Error scoring {model_name}: {ex}")

                    if rows:
                        result_df = pd.DataFrame(rows)
                        st.subheader("📊 Model Comparison Table")
                        st.dataframe(result_df.style.format({
                            "Accuracy": "{:.2%}",
                            "Precision": "{:.2%}",
                            "Recall": "{:.2%}",
                            "F1 Score": "{:.2%}",
                            "AUC": "{:.2f}"
                        }))

                        metric_column = "F1 Score" if "F1" in metric_choice else "Accuracy" if "Accuracy" in metric_choice else "AUC"
                        sorted_df = result_df.sort_values(by=metric_column, ascending=False)

                        st.subheader(f"🏆 Best Model by {metric_column}")
                        top_val = sorted_df.iloc[0][metric_column]
                        formatted_val = f"{top_val:.2f}" if metric_column == "AUC" else f"{top_val:.2%}"
                        st.success(f"Top Model: **{sorted_df.iloc[0]['Model']}** with {metric_column}: **{formatted_val}**")

                        st.bar_chart(result_df.set_index("Model")[metric_column])

                        if auc_curves:
                            st.subheader("📈 ROC Curves")
                            fig, ax = plt.subplots()
                            for name, fpr, tpr, auc_score in auc_curves:
                                ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
                            ax.plot([0, 1], [0, 1], "k--")
                            ax.set_xlabel("False Positive Rate")
                            ax.set_ylabel("True Positive Rate")
                            ax.set_title("ROC Curve")
                            ax.legend()
                            st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Failed to process test file: {e}")
        else:
            st.info("📌 Upload a CSV to evaluate models on real test data.")
