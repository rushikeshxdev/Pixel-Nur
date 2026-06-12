import os
import warnings
from flask import Flask, render_template
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

from api.routes import api_bp

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

app.register_blueprint(api_bp, url_prefix="/api")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    print(f"\n  PixelNur running at  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
