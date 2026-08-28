import time
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="AI Diagnostic Algorithm Evaluator", page_icon="🧬", layout="wide")
st.title("🧬 AI Diagnostic Algorithm Evaluator")
st.caption("Week 3 • Algorithm Selection and Evaluation for Diagnostic Applications")
st.warning("Educational prototype only. Not a clinical diagnostic tool.")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

test_size = st.sidebar.slider("Test-set size", 0.20, 0.40, 0.25, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(seed), stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "Neural Network (MLP)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1000,
                                early_stopping=True, random_state=42))
    ])
}

rows = []
for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    rows.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "Training time (s)": elapsed
    })

results = pd.DataFrame(rows)
st.subheader("Dataset")
a,b,c = st.columns(3)
a.metric("Samples", X.shape[0])
b.metric("Features", X.shape[1])
c.metric("Classes", len(data.target_names))
st.write("Breast Cancer Wisconsin (Diagnostic) dataset: 569 samples, 30 numeric features, 2 classes.")

st.subheader("Algorithm comparison")
st.dataframe(results.style.format({
    "Accuracy":"{:.3f}", "Precision":"{:.3f}", "Recall":"{:.3f}",
    "F1":"{:.3f}", "ROC-AUC":"{:.3f}", "Training time (s)":"{:.4f}"
}), use_container_width=True)

st.subheader("Metric chart")
st.bar_chart(results.set_index("Algorithm")[["Accuracy","Precision","Recall","F1","ROC-AUC"]])

best = results.sort_values(["ROC-AUC","F1"], ascending=False).iloc[0]
st.success(f"Prototype recommendation: {best['Algorithm']} ranked highest by ROC-AUC in this run.")

st.subheader("Interpretation")
st.markdown("""
- **Decision Tree:** transparent if/then rules and easy visualization.
- **SVM:** strong option for structured numeric data; kernel training can scale poorly on very large datasets.
- **Neural Network:** flexible nonlinear model but generally less transparent and more tuning-intensive.
- Healthcare evaluation should consider recall, precision, calibration, fairness and clinical consequences, not accuracy alone.
""")
st.info("For real clinical deployment, independent external validation, privacy/security controls, regulatory review and qualified human oversight are required.")
