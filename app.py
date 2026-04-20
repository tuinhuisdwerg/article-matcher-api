from flask import Flask, request, jsonify
import math
import json

app = Flask(__name__)

def parse_embedding(value):
    if isinstance(value, list):
        try:
            return [float(x) for x in value]
        except Exception:
            return []

    if isinstance(value, dict):
        data = value.get("data", [])
        if data and isinstance(data, list):
            first = data[0]
            if isinstance(first, dict) and "embedding" in first:
                try:
                    return [float(x) for x in first["embedding"]]
                except Exception:
                    return []
        if "embedding" in value and isinstance(value["embedding"], list):
            try:
                return [float(x) for x in value["embedding"]]
            except Exception:
                return []
        return []

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return [float(x) for x in parsed]

            if isinstance(parsed, dict):
                data = parsed.get("data", [])
                if data and isinstance(data, list):
                    first = data[0]
                    if isinstance(first, dict) and "embedding" in first:
                        return [float(x) for x in first["embedding"]]
                if "embedding" in parsed and isinstance(parsed["embedding"], list):
                    return [float(x) for x in parsed["embedding"]]
        except Exception:
            pass

        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]

        try:
            return [float(x.strip()) for x in value.split(",") if x.strip()]
        except Exception:
            return []

    return []

def parse_objectives(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []

    return []

def extract_record(item):
    if not isinstance(item, dict):
        return {}

    # vorm 1: {"Record": {...}}
    if isinstance(item.get("Record"), dict):
        return item["Record"]

    # vorm 2: {"data": {"Record": {...}}}
    data = item.get("data")
    if isinstance(data, dict) and isinstance(data.get("Record"), dict):
        return data["Record"]

    # vorm 3: {"data": {...record velden...}}
    if isinstance(data, dict):
        return data

    return item

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
    article_embedding_raw = data.get("article_embedding", [])
    objectives_raw = data.get("objectives", [])

    article_embedding = parse_embedding(article_embedding_raw)
    objectives = parse_objectives(objectives_raw)

    best_match = None
    best_score = -1

    first_item = objectives[0] if objectives and isinstance(objectives[0], dict) else None
    first_record = extract_record(first_item) if first_item else None
    first_record_embedding = parse_embedding(first_record.get("embedding", [])) if isinstance(first_record, dict) else []

    for item in objectives:
        record = extract_record(item)
        if not isinstance(record, dict):
            continue

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
        "similarity_score": round(best_score, 4) if best_match else 0,
        "debug": {
            "article_embedding_length": len(article_embedding),
            "objectives_count": len(objectives),
            "first_item_keys": list(first_item.keys()) if isinstance(first_item, dict) else None,
            "first_record_keys": list(first_record.keys()) if isinstance(first_record, dict) else None,
            "first_record_embedding_length": len(first_record_embedding),
            "article_embedding_raw_type": str(type(article_embedding_raw).__name__),
            "objectives_raw_type": str(type(objectives_raw).__name__)
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
