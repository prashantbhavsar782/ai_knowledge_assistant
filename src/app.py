from flask import Flask, request, render_template
from rag_pipeline import ask_question  # Import your RAG function

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    response = None
    user_input = None
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            response = ask_question(user_input)  # Get AI response from RAG pipeline
    return render_template("index.html", user_input=user_input, response=response)

if __name__ == "__main__":
    app.run(debug=True)
