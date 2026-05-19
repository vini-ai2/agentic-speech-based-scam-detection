from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import json
from pathlib import Path

from agent.agent import ScamDetectionAgent


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_FOLDER = ROOT / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("audio")
        if file is None:
            return redirect(request.url)
        filename = secure_filename(file.filename) or f"upload_{uuid.uuid4().hex}.wav"
        dest = UPLOAD_FOLDER / filename
        file.save(dest)

        agent = ScamDetectionAgent(use_llm=True)
        result = agent.evaluate(str(dest))
        result["filename"] = filename

        rid = uuid.uuid4().hex
        outpath = UPLOAD_FOLDER / f"{rid}.json"
        outpath.write_text(json.dumps(result, indent=2), encoding="utf-8")

        return redirect(url_for("result", rid=rid))

    return render_template("index.html")


@app.route("/result/<rid>")
def result(rid):
    path = UPLOAD_FOLDER / f"{rid}.json"
    if not path.exists():
        return "Result not found", 404
    result = json.loads(path.read_text(encoding="utf-8"))
    return render_template(
        "result.html",
        result=result,
        audio_url=url_for("uploaded_file", filename=result.get("filename")),
        rid=rid,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route("/result_json/<rid>")
def result_json(rid):
    path = UPLOAD_FOLDER / f"{rid}.json"
    if not path.exists():
        return "Not found", 404
    return send_from_directory(str(UPLOAD_FOLDER), f"{rid}.json", as_attachment=True, mimetype="application/json")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "audio" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["audio"]
    filename = secure_filename(file.filename) or f"upload_{uuid.uuid4().hex}.wav"
    dest = UPLOAD_FOLDER / filename
    file.save(dest)

    agent = ScamDetectionAgent(use_llm=True)
    result = agent.evaluate(str(dest))
    result["filename"] = filename
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=8501)
