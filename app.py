import os
import logging
from flask import Flask, request, jsonify, abort
from werkzeug.exceptions import BadRequest
from model_manager import ModelManager
from pydantic import BaseModel, ValidationError

# پیکربندی Flask و logging
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# مدل‌های pydantic برای اعتبارسنجی ورودی
class APIKeyRequest(BaseModel):
    api_key: str

class QueryRequest(BaseModel):
    question: str

# دیکشنری برای نگهداری نمونه‌های ModelManager بر اساس نام سایت
site_managers = {}

def get_site_manager(site_name: str):
    if site_name not in site_managers:
        site_managers[site_name] = ModelManager(site_name)
    return site_managers[site_name]

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    if not request.is_json:
        abort(400, description="Request must include a JSON body.")
    try:
        data = request.get_json()
        req = APIKeyRequest(**data)
    except ValidationError as e:
        abort(400, description=f"Input validation error: {e}")
    
    # تعیین نام سایت از هدر Host یا از بدنه JSON (اختیاری)
    site_name = request.headers.get('Host') or data.get('site')
    if not site_name:
        abort(400, description="Site identifier is required.")
    
    manager = get_site_manager(site_name)
    manager.set_api_key(req.api_key)
    return jsonify({"message": f"API key set successfully for site '{site_name}'.", "status": "success"}), 200

@app.route('/train', methods=['POST'])
def train_model():
    if not request.is_json:
        abort(400, description="Request must include a JSON body.")
    try:
        raw_data = request.get_json()
    except BadRequest:
        abort(400, description="Invalid JSON provided.")
    if not raw_data:
        abort(400, description="Empty JSON body provided.")
    
    # تعیین نام سایت از هدر Host یا از بدنه JSON (اختیاری)
    site_name = request.headers.get('Host') or raw_data.get('site')
    if not site_name:
        abort(400, description="Site identifier is required.")
    
    manager = get_site_manager(site_name)
    success, message = manager.train(raw_data)
    if not success:
        return jsonify({"message": message, "status": "no_action"}), 200
    return jsonify({"message": message, "status": "success"}), 200

@app.route('/query', methods=['POST'])
def handle_query():
    if not request.is_json:
        abort(400, description="Request must include a JSON body.")
    try:
        data = request.get_json()
    except BadRequest:
        abort(400, description="Invalid JSON provided.")
    
    try:
        req = QueryRequest(**data)
    except ValidationError as e:
        abort(400, description=f"Input validation error: {e}")
    
    # تعیین نام سایت از هدر Host یا از بدنه JSON (اختیاری)
    site_name = request.headers.get('Host') or data.get('site')
    if not site_name:
        abort(400, description="Site identifier is required.")
    
    manager = get_site_manager(site_name)
    success, answer, score = manager.query(req.question)
    status = "relevant" if score >= 0.3 else "irrelevant"
    return jsonify({"answer": answer, "score": float(score), "status": status}), 200

if __name__ == '__main__':
    logging.info("Starting Flask server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=10000, debug=True)
