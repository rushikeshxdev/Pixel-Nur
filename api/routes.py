import os
import io
import base64
import requests as http
from flask import Blueprint, request, jsonify
from PIL import Image
from core.steganography import embed, extract

api_bp = Blueprint("api", __name__)

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")
_PIXABAY_URL = "https://pixabay.com/api/"

@api_bp.route("/embed", methods=["POST"])
def api_embed():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        image_file = request.files["image"]
        message    = request.form.get("message", "").strip()
        password   = request.form.get("password", "")
        robustness = request.form.get("robustness", "None")

        if not message:
            return jsonify({"error": "Message cannot be empty."}), 400
        if not password:
            return jsonify({"error": "Password cannot be empty."}), 400
        if robustness not in ("None", "Low", "Medium", "High"):
            robustness = "None"

        img = Image.open(image_file.stream).convert("RGB")
        stego, metrics = embed(img, message, password, robustness)

        buf = io.BytesIO()
        stego.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return jsonify({
            "image":   f"data:image/png;base64,{img_b64}",
            "metrics": metrics,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

@api_bp.route("/extract", methods=["POST"])
def api_extract():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        image_file = request.files["image"]
        password   = request.form.get("password", "")

        if not password:
            return jsonify({"error": "Password cannot be empty."}), 400

        img = Image.open(image_file.stream).convert("RGB")
        message = extract(img, password)

        return jsonify({"message": message})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

@api_bp.route("/search-images")
def search_images():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Enter a search term."}), 400

    if not PIXABAY_API_KEY:
        return jsonify({"error": "Pixabay API key not set. Add PIXABAY_API_KEY=your_key to pixelNur/.env"}), 503

    try:
        resp = http.get(
            _PIXABAY_URL,
            params={
                "key":        PIXABAY_API_KEY,
                "q":          q,
                "image_type": "photo",
                "per_page":   15,
                "safesearch": "true",
                "lang":       "en",
            },
            timeout=8,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        images = [
            {"thumb": h["previewURL"], "full": h["webformatURL"]}
            for h in hits
            if h.get("previewURL") and h.get("webformatURL")
        ]
        return jsonify({"images": images})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route("/fetch-image")
def fetch_image():
    """Proxy an external image URL to avoid CORS issues in the browser."""
    url = request.args.get("url", "")
    if not url.startswith("https://"):
        return jsonify({"error": "Invalid URL — must start with https://"}), 400
    try:
        resp = http.get(url, timeout=10)
        ct = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
        img_b64 = base64.b64encode(resp.content).decode("utf-8")
        return jsonify({"image": f"data:{ct};base64,{img_b64}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
