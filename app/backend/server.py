from flask import Flask, request

app = Flask(__name__)


@app.route("/members")
def members():
    return "Members"


@app.route("/image", methods=["POST"])
def api():
    print("Recieved image")


app.run(port=5000)

if __name__ == "__main__":
    app.run(debug=True)
