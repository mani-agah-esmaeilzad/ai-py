import nest_asyncio
nest_asyncio.apply() 

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from google import generativeai as genai
from flask import Flask, request, jsonify, abort
from werkzeug.exceptions import BadRequest
import os

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  

vectorizer = None
xgb_model = None
trained_documents = []
gemini_model = None
GEMINI_API_KEY = None  
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

def query_with_gemini(question, context_docs):
    global gemini_model, GEMINI_API_KEY
    if not GEMINI_API_KEY:
        return "کلید API Gemini تنظیم نشده است. لطفاً به endpoint /set_api_key کلید را ارسال کنید."
    if not gemini_model:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    
    context = "\n\n".join(context_docs[:10])
    prompt = f"""بر اساس اطلاعات زیر به سوال پاسخ بده:

اطلاعات موجود:
---
{context}
---

سوال: {question}

پاسخ:"""
    try:
        response = gemini_model.generate_content(prompt)
        if response.parts:
            return response.text
        else:
            print("پاسخ Gemini فیلتر یا خالی بود:", response.prompt_feedback)
            return "متاسفانه نتوانستم پاسخی تولید کنم. ممکن است محتوای نامناسب شناسایی شده باشد."
    except Exception as e:
        print(f"خطا در ارتباط با Gemini API: {e}")
        return f"خطا در تولید پاسخ با Gemini: {e}"

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    global GEMINI_API_KEY, gemini_model
    if not request.is_json:
        abort(400, description="درخواست باید شامل بدنه JSON باشد.")
    try:
        data = request.get_json()
        api_key = data.get('api_key')
    except BadRequest:
        abort(400, description="JSON ارسال شده نامعتبر است.")
    if not api_key or not isinstance(api_key, str):
        abort(400, description="درخواست JSON باید شامل کلید 'api_key' با مقدار رشته‌ای باشد.")
    
    GEMINI_API_KEY = api_key
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = None
    return jsonify({"message": "کلید API با موفقیت تنظیم شد.", "status": "success"}), 200

@app.route('/train', methods=['POST'])
def train_model():
    global vectorizer, xgb_model, trained_documents
    if not request.is_json:
        abort(400, description="درخواست باید شامل بدنه JSON باشد.")
    try:
        raw_data = request.get_json()
    except BadRequest:
        abort(400, description="JSON ارسال شده نامعتبر است.")
    if not raw_data:
        abort(400, description="JSON ارسال شده خالی است.")
    
    texts = extract_text_from_json(raw_data)
    if not texts:
        return jsonify({"message": "هیچ متن قابل استخراجی در JSON ورودی یافت نشد.", "status": "no_action"}), 200
    
    trained_documents = [text for text in texts if text and text.strip()]
    if not trained_documents:
        return jsonify({"message": "متون استخراج شده پس از پاکسازی خالی بودند.", "status": "no_action"}), 200
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(trained_documents)
    
    documents_for_xgb = trained_documents + ["این یک متن کاملا نامرتبط به موضوع اصلی است"]
    labels_for_xgb = np.array([1] * len(trained_documents) + [0])
    X_xgb = vectorizer.transform(documents_for_xgb)
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_xgb, labels_for_xgb)
    
    return jsonify({
        "message": f"مدل با موفقیت روی {len(trained_documents)} سند متنی آموزش داده شد.",
        "status": "success"
    }), 200

@app.route('/query', methods=['POST'])
def handle_query():
    global vectorizer, xgb_model, trained_documents, gemini_model, GEMINI_API_KEY
    if not GEMINI_API_KEY:
        return jsonify({"error": "کلید API Gemini تنظیم نشده است. لطفاً به endpoint /set_api_key کلید را ارسال کنید."}), 400
    
    if not request.is_json:
        abort(400, description="درخواست باید شامل بدنه JSON باشد.")
    try:
        data = request.get_json()
        question = data.get('question')
    except BadRequest:
        abort(400, description="JSON ارسال شده نامعتبر است.")
    if not question or not isinstance(question, str):
        abort(400, description="درخواست JSON باید شامل کلید 'question' با مقدار رشته‌ای باشد.")
    
    if not vectorizer or not xgb_model or not trained_documents:
        return jsonify({"error": "مدل هنوز آموزش داده نشده است. لطفاً ابتدا به endpoint /train داده ارسال کنید."}), 400
    
    question_vec = vectorizer.transform([fix_encoding(question)])
    proba = xgb_model.predict_proba(question_vec)[0][1]
    threshold = 0.3
    if proba < threshold:
        return jsonify({"answer": "متاسفانه اطلاعاتی در مورد سوال شما در داده‌های آموزش دیده یافت نشد.", "score": float(proba), "status": "irrelevant"}), 200
    
    retrieved_answer = query_with_gemini(fix_encoding(question), trained_documents)
    return jsonify({"answer": retrieved_answer, "score": float(proba), "status": "relevant"}), 200

if __name__ == '__main__':
    print("اجرای سرور Flask در http://0.0.0.0:5000")
    print("برای تنظیم کلید API، یک درخواست POST با بدنه JSON حاوی {'api_key': '...'} به /set_api_key بفرستید.")
    app.run(host='0.0.0.0', port=10000, debug=True)