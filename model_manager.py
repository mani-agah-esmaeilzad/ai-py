import os
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from google import generativeai as genai
import joblib

# پیکربندی logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fix_encoding(text):
    if not isinstance(text, str):
        return text
    try:
        text.encode('utf-8')
        return text
    except UnicodeEncodeError:
        try:
            return text.encode('latin1').decode('utf-8')
        except Exception:
            return text
    except Exception:
        return text

def extract_text_from_json(data):
    texts = []
    if isinstance(data, dict):
        for key, value in data.items():
            texts.extend(extract_text_from_json(value))
    elif isinstance(data, list):
        for item in data:
            texts.extend(extract_text_from_json(item))
    elif isinstance(data, str):
        texts.append(fix_encoding(data))
    return texts

class ModelManager:
    def __init__(self, site_name: str):
        self.site_name = site_name
        # ساخت پوشه مخصوص سایت (اگر وجود ندارد)
        self.model_dir = os.path.join("saved_models", site_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.vector_file = os.path.join(self.model_dir, "vectorizer.pkl")
        self.xgb_model_file = os.path.join(self.model_dir, "xgb_model.pkl")
        self.trained_docs_file = os.path.join(self.model_dir, "trained_documents.pkl")
        
        self.vectorizer = None
        self.xgb_model = None
        self.trained_documents = []
        self.gemini_model = None
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", None)
        self.load_model()
    
    def load_model(self):
        """بارگذاری مدل‌های ذخیره شده برای سایت در صورت وجود."""
        if os.path.exists(self.vector_file) and os.path.exists(self.xgb_model_file) and os.path.exists(self.trained_docs_file):
            try:
                self.vectorizer = joblib.load(self.vector_file)
                self.xgb_model = joblib.load(self.xgb_model_file)
                self.trained_documents = joblib.load(self.trained_docs_file)
                logging.info("Models for site '%s' loaded successfully from disk.", self.site_name)
            except Exception as e:
                logging.error("Error loading models for site '%s': %s", self.site_name, e)
    
    def save_model(self):
        """ذخیره مدل‌های vectorizer، xgb_model و داده‌های آموزشی برای سایت."""
        try:
            joblib.dump(self.vectorizer, self.vector_file)
            joblib.dump(self.xgb_model, self.xgb_model_file)
            joblib.dump(self.trained_documents, self.trained_docs_file)
            logging.info("Models for site '%s' saved successfully to disk.", self.site_name)
        except Exception as e:
            logging.error("Error saving models for site '%s': %s", self.site_name, e)
    
    def set_api_key(self, api_key: str):
        """تنظیم کلید API Gemini برای سایت."""
        self.gemini_api_key = api_key
        os.environ["GEMINI_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        self.gemini_model = None  # بازنشانی مدل Gemini
        logging.info("Gemini API key set successfully for site '%s'.", self.site_name)
    
    def train(self, data):
        """آموزش مدل با استفاده از داده‌های ورودی برای سایت."""
        texts = extract_text_from_json(data)
        if not texts:
            return (False, "No text found to extract.")
        self.trained_documents = [text for text in texts if text and text.strip()]
        if not self.trained_documents:
            return (False, "Extracted texts are empty after cleaning.")
        
        # آموزش vectorizer و مدل XGBoost
        self.vectorizer = TfidfVectorizer()
        _ = self.vectorizer.fit_transform(self.trained_documents)
        
        # ایجاد داده‌های آموزشی برای XGBoost با نمونه‌های مثبت و یک نمونه منفی
        documents_for_xgb = self.trained_documents + ["This is a completely unrelated text."]
        labels_for_xgb = np.array([1] * len(self.trained_documents) + [0])
        X_xgb = self.vectorizer.transform(documents_for_xgb)
        
        self.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.xgb_model.fit(X_xgb, labels_for_xgb)
        logging.info("Model trained on %d documents for site '%s'.", len(self.trained_documents), self.site_name)
        self.save_model()  # ذخیره مدل‌ها پس از آموزش
        return (True, f"Model successfully trained on {len(self.trained_documents)} text documents for site '{self.site_name}'.")
    
    def query_with_gemini(self, question: str):
        """ارسال پرسش به Gemini API با استفاده از داده‌های آموزشی برای سایت."""
        if not self.gemini_api_key:
            return "Gemini API key is not set. Please set it using the /set_api_key endpoint."
        if not self.gemini_model:
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        context = "\n\n".join(self.trained_documents[:10])
        prompt = f"""Based on the following information, answer the question:

Information:
---
{context}
---

Question: {question}

Answer:"""
        try:
            response = self.gemini_model.generate_content(prompt)
            if response.parts:
                return response.text
            else:
                logging.warning("Gemini response empty or filtered for site '%s': %s", self.site_name, response.prompt_feedback)
                return "Unfortunately, I was unable to generate an answer. The content may have been flagged as inappropriate."
        except Exception as e:
            logging.error("Error in Gemini API for site '%s': %s", self.site_name, e)
            return f"Error generating answer with Gemini: {e}"
    
    def query(self, question: str):
        """پاسخ به پرسش کاربر؛ در صورت مرتبط بودن، از مدل Gemini استفاده می‌شود."""
        if not self.vectorizer or not self.xgb_model or not self.trained_documents:
            return (False, "Model has not been trained yet. Please post data to /train first.", 0)
        question_encoded = fix_encoding(question)
        question_vec = self.vectorizer.transform([question_encoded])
        proba = self.xgb_model.predict_proba(question_vec)[0][1]
        threshold = 0.3
        if proba < threshold:
            return (True, "Unfortunately, no relevant information was found in the training data for your question.", proba)
        retrieved_answer = self.query_with_gemini(question_encoded)
        return (True, retrieved_answer, proba)
