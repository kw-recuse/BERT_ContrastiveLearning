import re
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("recuse/distiluse-base-multilingual-cased-v2-mean-dataV1")
model = AutoModel.from_pretrained("recuse/distiluse-base-multilingual-cased-v2-mean-dataV1")

# inference
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().tolist()

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    cleaned_text = clean_text(text)

    embedding = get_embedding(cleaned_text)
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)

"""
resume_str = "...."
curl -X POST http://127.0.0.1:5004/embed \
-H "Content-Type: application/json" \
-d '{"text": resume_str}'
"""