#main.py
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from eccv16 import eccv16
from siggraph17 import siggraph17
from util import preprocess_img, postprocess_tens

def colorize_image_eccv16(image):
    model = eccv16(pretrained=True).eval()  # Assuming you use eccv16 model
    img = np.array(image)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    out_img = postprocess_tens(tens_l_orig, model(tens_l_rs).cpu())
    return Image.fromarray((out_img * 255).astype(np.uint8))

def colorize_image_siggraph17(image):
    model = siggraph17(pretrained=True).eval()  # Assuming you use eccv16 model
    img = np.array(image)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    out_img = postprocess_tens(tens_l_orig, model(tens_l_rs).cpu())
    return Image.fromarray((out_img * 255).astype(np.uint8))


