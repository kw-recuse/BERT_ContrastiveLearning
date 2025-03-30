# Run the requirements.txt first (transformers, torch, etc.)
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("recuse/distiluse-base-multilingual-cased-v2-mean-dataV1")
model = AutoModel.from_pretrained("recuse/distiluse-base-multilingual-cased-v2-mean-dataV1")

# inference
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().tolist()

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "Text is required"}), 400

    embedding = get_embedding(text)
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)