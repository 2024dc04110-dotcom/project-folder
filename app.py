import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt

# Import models
from model.logistic_regression import run_model as logistic
from model.decision_tree import run_model as decision_tree
from model.knn import run_model as knn
from model.naive_bayes import run_model as naive_bayes
from model.random_forest import run_model as random_forest
from model.xgboost import run_model as xgboost

MODEL_MAP = {
    "Logistic Regression": logistic,
    "Decision Tree": decision_tree,
    "KNN": knn,
    "Naive Bayes": naive_bayes,
    "Random Forest": random_forest,
    "XGBoost": xgboost
}

st.title("📊 ML Classification Models")

# ------------------------------------------
# Dataset Source Selection
# ------------------------------------------
st.subheader("📂 Choose Data Source")

col1, col2 = st.columns(2)

with col1:
    upload_btn = st.button("📤 Upload Dataset", key="upload_btn", use_container_width=True)

with col2:
    github_btn = st.button("📥 Data Set from GitHub", key="github_btn", use_container_width=True)

# Initialize session state
if "data_source" not in st.session_state:
    st.session_state.data_source = None

if upload_btn:
    st.session_state.data_source = "upload"

if github_btn:
    st.session_state.data_source = "github"

# ------------------------------------------
# Load Data Based on Selection
# ------------------------------------------
df = None

if st.session_state.data_source == "upload":
    st.subheader("📤 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file (test-sized dataset only)",
        type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif st.session_state.data_source == "github":
    st.subheader("📥 DataSet from GitHub...")
    try:
        # Hardcoded GitHub path
        github_url = "https://github.com/2024dc04110-dotcom/project-folder/blob/dc26db55c524212700498e9b79c5ff40f268f3fb/data/heartdisease.csv?raw=true"
        df = pd.read_csv(github_url)
        st.success("Dataset loaded successfully from GitHub!")
    except Exception as e:
        st.error(f"Error loading dataset from GitHub: {e}")
        st.info("Please check if the GitHub URL is correct and the file is accessible.")

# ------------------------------------------
# Continue ONLY if data is loaded
# ------------------------------------------
if df is not None:
    st.write("Dataset Preview", df.head())

    if st.session_state.data_source == "github":
        target_col = "HeartDisease"
    else:
        target_col = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0})

    # ------------------------------------------
    # Train-Test Split
    # ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------
    # Feature Processing
    # ------------------------------------------
    num_cols = X_train.select_dtypes(include=['int64','float64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    model_name = st.selectbox("Select Model", MODEL_MAP.keys())

    if st.button("Run Model"):
        metrics, cm = MODEL_MAP[model_name](
            X_train, X_test, y_train, y_test
        )

        st.subheader("📌 Evaluation Metrics")
        for k, v in metrics.items():
            st.write(f"**{k}:** {v}")

        st.subheader("📉 Confusion Matrix")
        fig, ax = plt.subplots()

        class_names = ["Yes", "No"]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"Predicted {c}" for c in class_names],
            yticklabels=[f"Actual {c}" for c in class_names],
            ax=ax
        )
        st.pyplot(fig)
