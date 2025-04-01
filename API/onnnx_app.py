import re
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import onnxruntime
import numpy as np

app = Flask(__name__)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

model_name = "recuse/distiluse-base-multilingual-cased-v2-mean-dataV1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
onnx_session = onnxruntime.InferenceSession("distiluse-base-multilingual-cased-v2-mean-dataV1.onnx", providers=["CPUExecutionProvider"])

def get_embedding_onnx(text):
    cleaened_text = clean_text(text)
    inputs = tokenizer(cleaened_text, return_tensors="np", padding=True, truncation=True)
    
    onnx_inputs = {
        "input_ids": inputs['input_ids'].astype(np.int64),
        "attention_mask": inputs['attention_mask'].astype(np.int64)
    }
    
    onnx_outputs = onnx_session.run(None, onnx_inputs)
    embeddings = np.mean(onnx_outputs[0], axis=1)
    return embeddings.tolist()

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "Text is required"}), 400

    embedding = get_embedding_onnx(text)
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5009)
