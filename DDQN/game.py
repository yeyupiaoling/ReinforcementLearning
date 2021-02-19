from flask import Flask, render_template

app = Flask(__name__, template_folder="dino", static_folder="dino", static_url_path="/")


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
