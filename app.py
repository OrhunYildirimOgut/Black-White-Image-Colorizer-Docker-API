#app.py
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import base64
from main import colorize_image_eccv16, colorize_image_siggraph17
from util import load_img_from_json, image_to_base64
app = Flask(__name__)

@app.route('/colorize', methods=['POST'])
def colorize():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    image, image_format= load_img_from_json(data['image'])

    # Colorize the image
    colorized_image_eccv16 = colorize_image_eccv16(image)
    colorized_image_siggraph17 = colorize_image_siggraph17(image)

    # Convert images to base64 to return as JSON
    original_img_base64 = image_to_base64(image, image_format)
    colorized_img_eccv16_base64 = image_to_base64(colorized_image_eccv16, image_format)
    colorized_img_siggraph17_base64 = image_to_base64(colorized_image_siggraph17, image_format)


    return jsonify({
        'original_image': original_img_base64,
        'colorized_img_model1': colorized_img_eccv16_base64,
        'colorized_img_model2': colorized_img_siggraph17_base64
    })

if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)


