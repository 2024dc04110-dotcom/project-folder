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

MODEL_MAP = {
    "Logistic Regression": logistic,
    "Decision Tree": decision_tree,
}

st.title("📊 ML Assignment-2 Classification App")

uploaded_file = st.file_uploader("Upload CSV (Test Data)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

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
