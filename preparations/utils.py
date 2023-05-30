import base64
import cv2
import numpy


def get_code(pil_image):
    """
    Encode image and get its height and width
    Returns: code, height and width of photo
    """
    img = numpy.array(pil_image)
    img = img[:, :, ::-1].copy()
    success, encoded_image = cv2.imencode('.jpg', img)
    byte_data = encoded_image.tobytes()
    return base64.b64encode(byte_data).decode('utf-8'), img.shape[0], img.shape[1]
