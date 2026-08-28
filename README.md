# AI Diagnostic Algorithm Evaluator – Week 3

## Objective
Compare Decision Tree, Support Vector Machine (SVM), and Neural Network (MLP) models for a healthcare diagnostic classification task.

## Dataset
Scikit-learn Breast Cancer Wisconsin (Diagnostic) dataset:
- 569 samples
- 30 numerical features
- 2 classes

## Method
- Stratified 80/20 train-test split
- `random_state=42`
- StandardScaler before SVM and MLP
- Decision Tree: max_depth=4, balanced class weights
- SVM: RBF kernel, C=1.0, balanced class weights
- MLP: hidden layers (32,16), early stopping

## Prototype result
In the fixed held-out experiment, SVM gave the strongest overall balance of accuracy, precision, recall and F1-score.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – interactive Streamlit evaluator
- `requirements.txt` – dependencies
- `Week3_AI_Diagnostic_Algorithm_Evaluator_Enhanced.docx` – full report
- `evaluation_results.csv` – recorded benchmark results

## Important limitation
This is an educational prototype, not a clinical diagnostic system. Public-dataset benchmark results do not establish clinical effectiveness or safety.
