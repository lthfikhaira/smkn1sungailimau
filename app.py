from flask import Flask, render_template, request, jsonify
from chat import get_response
from controllers.daftar import handle_form

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("base.html")

@app.get("/about")
def about():
    return render_template("about.html")

@app.get("/pendaftaran")
def pendaftaran():
    return render_template("pendaftaran.html")

@app.get("/pendidik")
def pendidik():
    return render_template("pendidik.html")

@app.route('/handle_form', methods=['POST'])
def handle_form_route():
    return handle_form()

@app.post("/predict")
def predict():
    text = request.get_json().get("message")

    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
