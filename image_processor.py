import cv2
import os
import numpy as np
from ultralytics import YOLOWorld
from ultralytics import YOLOE


os.environ['YOLO_HOME'] = './models'


class ImageProcessor:
    """Class in charge of doing object detection, segmentation and tracking."""
    
    def __init__(self, model_name):

        classes = ["toy lego block"]
        if model_name and model_name.startswith("yoloe"):
            self.model = YOLOE(model_name)
            self.model.set_classes(classes, self.model.get_text_pe(classes))
        else:
            self.model = YOLOWorld(model_name)
            self.model.set_classes(classes)
        

    def process_image(self, image):
        # Decode JPEG image
        image = cv2.imdecode(
            np.frombuffer(image, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        results = self.model.track(image, device='mps', conf=0.01, max_det=1, persist=True)
        # Visualize the results on the frame
        image = results[0].plot(labels=False)
        # Compress image (adjust quality as needed)
        _, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image = jpeg.tobytes()
        return image


    def set_classes(self, classes):
        if self.model_name and self.model_name.startswith("yoloe"):
            self.model.set_classes(classes, self.model.get_text_pe(classes))
        else:
            self.model.set_classes(classes)