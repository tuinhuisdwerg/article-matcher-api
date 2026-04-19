from flask import Flask, request, jsonify
import math

app = Flask(__name__)

def parse_embedding(value):
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if not value:
            return []
        return [float(x.strip()) for x in value.split(",")]
    return []

def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Article matcher API is running"})

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()

    article_title = data.get("article_title", "")
    article_description = data.get("article_description", "")
    article_embedding = parse_embedding(data.get("article_embedding", []))
    objectives = data.get("objectives", [])

    best_match = None
    best_score = -1

    for item in objectives:
        record = item.get("Record", item)
        objective_embedding = parse_embedding(record.get("embedding", []))
        score = cosine_similarity(article_embedding, objective_embedding)

        if score > best_score:
            best_score = score
            best_match = record

    matched = best_score >= 0.70 if best_match else False

    return jsonify({
        "matched": matched,
        "article_title": article_title,
        "article_description": article_description,
        "objective_id": str(best_match.get("id")) if best_match else None,
        "learning_objective": best_match.get("leerdoel") if best_match else None,
        "similarity_score": round(best_score, 4) if best_match else 0
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
