from flask import Flask, request, jsonify, render_template
import torch
import json
from backend.model.model import LSTMClassifier
from backend.model.preprocess import tokenize_text, encode_text

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 2
MAX_LEN = 200

with open("backend/model/vocab.json", "r") as f:
    vocab = json.load(f)

model_path = "backend/model/model.pth"
model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"Error": "Field 'text' is required"}), 400

    text = data["text"]
    tokenized_text = tokenize_text(text)
    encoded_text = encode_text(tokenized_text, vocab, MAX_LEN).unsqueeze(0)

    with torch.no_grad():
        prediction = model(encoded_text)
        prediction_class = torch.argmax(prediction, dim=1).item()
        result = "Fake News" if prediction_class == 0 else "Real News"
    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True)
