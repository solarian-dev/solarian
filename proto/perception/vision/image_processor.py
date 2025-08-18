import cv2
import torch
import numpy as np
from ultralytics import YOLO, YOLOWorld, YOLOE, FastSAM

from .depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
 


class ImageProcessor:
    """Class in charge of doing object detection, segmentation and tracking."""
    
    def __init__(self, tracking_model_name, depth_model_name):
        self.tracking_model_name = tracking_model_name
        self.depth_model_name = depth_model_name
        # Initialize outside the processing loop
        self.current_id = None
        self.clicked_point = None
        self.tracking_active = False
        self.tracked_id = None
        self.show_mask = False
        self.tracking_confidence = 0.1
        classes = ["toy lego block"]
        if tracking_model_name and "yoloe" in tracking_model_name:
            self.tracking_model = YOLOE(tracking_model_name)
            # self.model.set_classes(classes, self.model.get_text_pe(classes))
        elif "FastSAM" in tracking_model_name:
            self.tracking_model = FastSAM(tracking_model_name)
        elif "YOLOWorld" in tracking_model_name:
            self.tracking_model = YOLOWorld(tracking_model_name)
            # self.model.set_classes(classes)
        else:
            self.tracking_model = YOLO(tracking_model_name)
            # self.model.set_classes(classes)

        if depth_model_name:
            model_depth_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

            encoder = 'vits' # or 'vits', 'vitb', 'vitg'

            self.depth_model = DepthAnythingV2(**model_depth_configs[encoder])
            self.depth_model.load_state_dict(torch.load(depth_model_name, map_location='cpu'))
            self.depth_model = self.depth_model.to(DEVICE).eval()

    def track_object(self, clicked_point):
        self.clicked_point = clicked_point

    def process_image(self, image):
        # Decode JPEG image
        # frame = cv2.imdecode(
        #     np.frombuffer(image, dtype=np.uint8),
        #     cv2.IMREAD_COLOR
        # )
        frame = image

        device = 0 if DEVICE == 'cuda' else DEVICE


        # track object
        if "FastSAM" in self.tracking_model_name:
            # print("Processing FastSAM")
            
            results = None

            # Run YOLO tracking with optimized parameters
            if self.clicked_point:
                print(f"Tracking point: {self.clicked_point}")
                results = self.tracking_model.track(
                    frame,
                    device=device,
                    conf=self.tracking_confidence,  # Increased confidence threshold for more stable detections
                    # max_det=10,
                    persist=True,
                    iou=0.2,    # IOU threshold for tracking
                    verbose=False,
                    points=[self.clicked_point]
                )
                if results and len(results[0].boxes) > 0:
                    self.clicked_point = None
                    self.tracking_active = True
                    self.tracked_id = results[0].boxes[0].id.item() if hasattr(results[0].boxes[0], 'id') else None
            elif self.tracking_active:
                print("Tracking pointed object id:", self.tracked_id)
                results = self.tracking_model.track(
                    frame,
                    device=device,
                    conf=self.tracking_confidence,  # Increased confidence threshold for more stable detections
                    # max_det=10,
                    verbose=False,
                    # texts=names
                    persist=True,  # This is key for maintaining tracking
                    iou=0.2,
                    
                )

            tracked_box = None
            tracked_mask = None

            # # Visualization
            if results and len(results[0].boxes) > 0:
                # Only show the tracked object
                boxes = results[0].boxes
                masks = results[0].masks
                
                
                # Find our tracked object
                for i, box in enumerate(boxes):
                    if hasattr(box, 'id') and box.id.item() == self.tracked_id:
                        tracked_box = box.xyxy[0].tolist()
                        print("Tracked box: ", box)

                        # if self.show_mask:
                        #     tracked_mask = masks[i].data[0].cpu().numpy()
                        #     # Resize mask to match frame dimensions
                        #     tracked_mask = cv2.resize(
                        #         tracked_mask,
                        #         (frame.shape[1], frame.shape[0]),
                        #         interpolation=cv2.INTER_NEAREST
                        #     )
                        #     print("mask shape", tracked_mask.shape)
                        break
                
            #     if tracked_box is not None:
            #         # Create a copy of the frame for drawing
            #         annotated_frame = frame.copy()
            #         if self.show_mask:
            #             # Create colored overlay for the mask
            #             color_mask = np.zeros_like(annotated_frame)
            #             color_mask[tracked_mask == 1] = [0, 255, 0]  # Green color for the mask
                        
            #             # Blend the mask with the frame
            #             alpha = 0.5  # Transparency factor
            #             mask_overlay = cv2.addWeighted(annotated_frame, 1, color_mask, alpha, 0)
            #             annotated_frame = mask_overlay
                    
            #         # Draw just the tracked box
            #         x1, y1, x2, y2 = tracked_box.xyxy[0].tolist()
            #         tracked_box = [x1, y1, x2, y2]
            #         # cv2.rectangle(annotated_frame, 
            #         #             (int(x1), int(y1)), 
            #         #             (int(x2), int(y2)), 
            #         #             (0, 255, 0), 2)
                    
            #         # cv2.putText(annotated_frame, f"Tracking ID: {self.tracked_id}", 
            #         #         (int(x1), int(y1) - 10), 
            #         #         cv2.FONT_HERSHEY_SIMPLEX, 
            #         #         0.9, (0, 255, 0), 2)
            #     else:
            #         print("Lost tracked object")
            #         self.tracking_active = False
            #         self.tracked_id = None
            #         annotated_frame = frame
            # else:
            #     annotated_frame = frame


            # frame = annotated_frame

        else:
            results = self.tracking_model.track(frame, device=device, conf=0.01, persist=True)
            if results:
                # Visualize the results on the frame
                frame = results[0].plot(labels=True)
        
        # Compress image (adjust quality as needed)
        # _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        # return jpeg.tobytes(), tracked_box
        return None, tracked_box


    def set_classes(self, classes):
        if self.tracking_model_name and "yoloe" in self.tracking_model_name:
            self.tracking_model.set_classes(classes, self.tracking_model.get_text_pe(classes))
        else:
            self.tracking_model.set_classes(classes)