import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_PATH = "classifier/saved_model/model.pkl"

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained model not found. Run train.py first.")
        return
        
    model_bundle = joblib.load(MODEL_PATH)
    type_model = model_bundle["type_model"]
    dept_model = model_bundle["dept_model"]
    
    # Comprehensive test dataset simulating unseen incoming medical records
    test_data = [
        ("Complete blood count hemoglobin and platelet levels test result", "Laboratory Report", "Laboratory"),
        ("Chest radiograph showing bilateral lung fields clear", "X-Ray", "Radiology"),
        ("Brain MRI scan finding normal intracranial structures", "MRI Scan", "Radiology"),
        ("Electrocardiogram rhythm strip normal sinus rhythm", "ECG", "Cardiology"),
        ("Prescription for amoxicillin 500mg tablets", "Prescription", "Pharmacy"),
        ("Discharge summary patient stable post recovery", "Discharge Summary", "Medical Records")
    ]
    
    X_test = [item[0] for item in test_data]
    y_type_true = [item[1] for item in test_data]
    y_dept_true = [item[2] for item in test_data]
    
    # Predictions
    y_type_pred = type_model.predict(X_test)
    y_dept_pred = dept_model.predict(X_test)
    
    print("==========================================")
    print("       AI MODEL PERFORMANCE METRICS       ")
    print("==========================================")
    
    print(f"\n--- Document Type Classifier Accuracy: {accuracy_score(y_type_true, y_type_pred) * 100:.1f}% ---")
    print(classification_report(y_type_true, y_type_pred, zero_division=0))
    
    print(f"\n--- Department Router Accuracy: {accuracy_score(y_dept_true, y_dept_pred) * 100:.1f}% ---")
    print(classification_report(y_dept_true, y_dept_pred, zero_division=0))
    
    print("\nConfusion Matrix (Department Routing):")
    print(confusion_matrix(y_dept_true, y_dept_pred))

if __name__ == "__main__":
    evaluate_model()
