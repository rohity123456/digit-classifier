"""
src/api.py

A small Flask app that exposes a /predict endpoint.
Relies on src/infer.py for preprocessing and prediction.

Note: This is a simple dev server. For production use gunicorn/uvicorn and a WSGI entry point.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.inference.index import preprocess_image_pil, load_params, predict_from_image_array
import io
import base64
import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
app = Flask(__name__)

# Lazy load params at first request to allow the app to start even if checkpoint is not present
_PARAMS = None


def get_params():
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = load_params()
    return _PARAMS

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        params = get_params()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Accept form file upload
    if 'file' in request.files:
        file = request.files['file']
        img = Image.open(file.stream)
    else:
        data = request.get_json(silent=True)
        if not data or 'image_b64' not in data:
            return jsonify({'error': 'No image provided. Use multipart/form-data with "file" or JSON {"image_b64": ...}'}), 400
        try:
            b = base64.b64decode(data['image_b64'])
            img = Image.open(io.BytesIO(b))
        except Exception as e:
            return jsonify({'error': 'Invalid base64 image: ' + str(e)}), 400

    x = preprocess_image_pil(img)
    pred, probs = predict_from_image_array(x, params)
    top3_idx = list(np.argsort(probs)[-3:][::-1])
    top3 = [{'label': int(i), 'prob': float(probs[i])} for i in top3_idx]
    return jsonify({'pred': int(pred), 'probs': probs.tolist(), 'top3': top3})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))