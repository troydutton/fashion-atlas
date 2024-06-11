import base64
import sys
from io import BytesIO

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from inference import get_similar_garments

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def convert_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


@app.route("/image", methods=["POST"])
def image():
    try:
        # Get the file from the request
        file = request.files["image"]

        image = Image.open(file)

        results = get_similar_garments(image)

        if results is None:
            return jsonify({"error": "No detections found"}), 500

        # Convert the images to base64
        for result in results:
            result["similar_garments"] = [convert_image_to_base64(img) for img in result["similar_garments"]]
            result["similar_models"] = [convert_image_to_base64(img) for img in result["similar_models"]]

        return jsonify(results), 200
    except KeyError:
        # The client did not provide an image
        return jsonify({"error": "No image provided"}), 400
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 501


if __name__ == "__main__":
    if len(sys.argv) > 1:
        host = sys.argv[1]
        app.run(host=host, port=5000, debug=True)
    else:
        print("No host provided, please provide a host")
