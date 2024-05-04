from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/image", methods=["POST"])
def image():
    # Get the file from the request
    file = request.files["image"]

    image = Image.open(file)

    image.show()

    return jsonify({"message": "Image uploaded successfully"}), 200


if __name__ == "__main__":
    app.run(host="100.110.148.60", port=5000, debug=True)
