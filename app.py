
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(page_title="AI Diagnostic Algorithm Evaluator", page_icon="🩺", layout="wide")

st.title("🩺 AI Diagnostic Algorithm Evaluator")
st.caption("Week 3 educational prototype • AI assists; professionals decide")

st.warning(
    "Educational prototype only. This application is NOT a clinical diagnostic system "
    "and must not be used to diagnose or treat patients."
)

@st.cache_data
def load_data():
    d = load_breast_cancer()
    return d.data, d.target, d.feature_names, d.target_names

X, y, feature_names, target_names = load_data()

@st.cache_data
def run_experiment():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42
        ),
        "SVM": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(
                C=1.0, kernel="rbf", probability=True,
                class_weight="balanced", random_state=42
            ))
        ]),
        "Neural Network": Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=1e-3,
                max_iter=1000,
                early_stopping=True,
                random_state=42
            ))
        ])
    }

    rows, fitted = [], {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, pred)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, prob),
            "Train time (s)": train_time,
            "False negatives": int(cm[1,0]),
            "False positives": int(cm[0,1]),
        })
        fitted[name] = (model, cm)

    return pd.DataFrame(rows), fitted, y_test

results, fitted, y_test = run_experiment()

st.subheader("Experimental results")
st.dataframe(
    results.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1": "{:.2%}",
        "ROC-AUC": "{:.3f}",
        "Train time (s)": "{:.4f}",
    }),
    use_container_width=True
)

best = results.sort_values(["F1", "Recall"], ascending=False).iloc[0]
st.success(
    f"Prototype recommendation: **{best['Model']}** based on the strongest combined "
    f"F1-score and recall in this fixed held-out experiment."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Metric comparison")
    chart = results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]]
    st.bar_chart(chart)
with col2:
    st.subheader("False-negative comparison")
    st.bar_chart(results.set_index("Model")[["False negatives"]])

st.subheader("Confusion matrix")
choice = st.selectbox("Select model", list(fitted.keys()))
model, cm = fitted[choice]
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[str(x).title() for x in target_names]
)
disp.plot(ax=ax, values_format="d")
st.pyplot(fig, clear_figure=True)

st.subheader("Dataset")
st.write(
    f"Samples: **{len(X)}** • Features: **{X.shape[1]}** • "
    f"Classes: **{', '.join(target_names)}**"
)

with st.expander("Why SVM is recommended in this prototype"):
    st.write(
        "On the fixed 80/20 stratified split (random_state=42), SVM achieved the highest "
        "accuracy and F1-score among the three tested configurations, while maintaining "
        "strong recall and ROC-AUC. Decision Tree remains the most transparent baseline, "
        "and the Neural Network remains a useful nonlinear benchmark."
    )

with st.expander("Validation required before any real-world use"):
    st.markdown(
        "- External validation on representative data\n"
        "- Repeated cross-validation and uncertainty estimates\n"
        "- Threshold and calibration analysis\n"
        "- Subgroup/fairness assessment where appropriate\n"
        "- Data-quality and robustness testing\n"
        "- Human-factors and clinical workflow evaluation\n"
        "- Security, auditability and rollback controls\n"
        "- Appropriate regulatory/governance review"
    )
