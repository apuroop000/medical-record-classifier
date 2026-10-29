import os
import ast
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def train_and_compare_classifiers():
    print("🔄 Loading medical document dataset...")
    dataset_path = "data/medical_documents_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}.")
        return

    df = pd.read_csv(dataset_path)
    
    # Clean data: drop nulls or empty texts
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != ""]

    # Map / Filter to the strict 5 target categories
    # Target Classes: 
    # 1. Laboratory / Blood Test Report
    # 2. X-Ray Report
    # 3. ECG Report
    # 4. Prescription
    # 5. Discharge Summary
    
    def map_to_target_classes(row):
        lbl = str(row['label']).strip().lower()
        txt = str(row['text']).strip().lower()
        
        if lbl == 'prescription':
            return 'Prescription'
        elif lbl == 'discharge_summary':
            return 'Discharge Summary'
        elif lbl == 'lab_report' or 'blood' in txt or 'pathology' in txt:
            return 'Laboratory / Blood Test Report'
        elif lbl == 'scan_report' and ('x-ray' in txt or 'xray' in txt or 'radiology' in txt):
            return 'X-Ray Report'
        elif 'ecg' in txt or 'electrocardiogram' in txt or ('scan_report' in txt and 'cardio' in txt):
            return 'ECG Report'
        else:
            return None  # Exclude unrelated categories

    print("⚙️ Filtering and mapping dataset to the 5 target classes...")
    df['target_class'] = df.apply(map_to_target_classes, axis=1)
    df_filtered = df.dropna(subset=['target_class']).copy()

    print(f"📊 Filtered dataset shape: {df_filtered.shape[0]} samples (down from {len(df)})")
    print(df_filtered['target_class'].value_counts())

    if len(df_filtered) < 50:
        print("❌ Error: Too few samples remaining after filtering. Check mapping logic.")
        return

    X = df_filtered['text']
    y_raw = df_filtered['target_class']

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Train / Test split (80% train, 20% separate test set for accurate comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define TF-IDF Vectorizer (No filename reliance as requested)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\n--- Training Model 1: Logistic Regression ---")
    logreg = LogisticRegression(max_iter=1000, C=2.0, class_weight='balanced', n_jobs=-1)
    logreg.fit(X_train_vec, y_train)
    y_pred_lr = logreg.predict(X_test_vec)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {acc_lr * 100:.2f}%")

    print("\n--- Training Model 2: XGBoost Classifier ---")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    xgb.fit(X_train_vec, y_train)
    y_pred_xgb = xgb.predict(X_test_vec)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost Accuracy: {acc_xgb * 100:.2f}%")

    # Select the model that performs better on the separate test set
    if acc_lr >= acc_xgb:
        best_model = logreg
        best_name = "Logistic Regression"
    else:
        best_model = xgb
        best_name = "XGBoost"

    print(f"\n🏆 Selected Best Performing Model: {best_name}")

    # Create full pipeline bundle for deployment
    model_bundle = {
        "vectorizer": vectorizer,
        "model": best_model,
        "label_encoder": le,
        "model_type": best_name
    }

    os.makedirs("classifier/saved_model", exist_ok=True)
    model_path = "classifier/saved_model/model.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"✅ Successfully saved optimized 5-class model bundle to {model_path}!")

    # Print actual evaluation metrics on test set
    print("\n📋 Classification Report on Separate Test Set:")
    y_pred_best = best_model.predict(X_test_vec)
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

if __name__ == "__main__":
    train_and_compare_classifiers()
