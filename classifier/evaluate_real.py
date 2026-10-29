import os
import joblib
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

def evaluate_real_world_samples():
    model_path = "classifier/saved_model/model.pkl"
    if not os.path.exists(model_path):
        print("❌ Error: Trained model bundle not found.")
        return

    model_bundle = joblib.load(model_path)
    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]
    le = model_bundle["label_encoder"]

    test_base_dir = "data/real_test"
    if not os.path.exists(test_base_dir):
        print(f"⚠️ Directory '{test_base_dir}' not found.")
        return

    print("🔍 Evaluating Unseen Real-World Test Documents with OCR...\n")
    total, correct = 0, 0

    for category_folder in os.listdir(test_base_dir):
        folder_path = os.path.join(test_base_dir, category_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                continue

            file_path = os.path.join(folder_path, filename)
            ext = os.path.splitext(filename)[1].lower()

            # Robust Text Extraction (PyMuPDF for PDFs, Tesseract OCR for Images)
            text = ""
            try:
                if ext == ".pdf":
                    doc = fitz.open(file_path)
                    for page in doc:
                        text += page.get_text() + "\n"
                else:
                    # Perform OCR on image screenshots
                    image = Image.open(file_path)
                    text = pytesseract.image_to_string(image)
            except Exception as e:
                text = filename

            # Fallback if OCR finds nothing
            if not text.strip():
                text = filename

            # Model Inference
            X_vec = vectorizer.transform([text])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_vec)
                pred_idx = probs.argmax()
                confidence = float(probs[0, pred_idx])
            else:
                pred_idx = model.predict(X_vec)[0]
                confidence = 1.0

            predicted_type = le.inverse_transform([pred_idx])[0]
            
            total += 1
            is_match = category_folder.lower() in predicted_type.lower()
            if is_match:
                correct += 1

            print(f"📄 File: {filename}")
            print(f"   * Actual Category Folder: {category_folder}")
            print(f"   * Predicted Type: {predicted_type}")
            print(f"   * Confidence: {confidence * 100:.1f}%")
            print(f"   * Result: {'✅ Correct' if is_match else '❌ Mismatch'}\n")

    if total > 0:
        print(f"📊 Summary: {correct}/{total} correct ({correct/total*100:.2f}% accuracy on real test set)")
    else:
        print(f"ℹ️ No test files found in {test_base_dir}/.")

if __name__ == "__main__":
    evaluate_real_world_samples()
