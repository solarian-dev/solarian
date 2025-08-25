import cv2
import torch
import numpy as np
from ultralytics import YOLO, YOLOWorld, YOLOE, FastSAM

from .depth_anything_v2.dpt import DepthAnythingV2


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class Coords:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ImageProcessor:
    """Class in charge of doing object detection, segmentation and tracking."""
    
    def __init__(self, tracking_model_name, depth_model_name, ref_x, ref_y):
        self.tracking_model_name = tracking_model_name
        self.depth_model_name = depth_model_name
        # Initialize outside the processing loop
        self.current_id = None
        self.clicked_point = None
        self.tracking_active = False
        self.tracked_id = None
        self.show_mask = False
        self.tracked_box = None
        self.tracked_mask = None
        self.tracking_confidence = 0.6
        self.tracking_iou = 0.2
        self.tracker = 'models/custom_tracker.yaml'
        self.ref_coords = Coords(ref_x, ref_y)
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
        frame = image
        device = 0 if DEVICE == 'cuda' else DEVICE

        # track object
        if "FastSAM" in self.tracking_model_name:
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
                    iou=self.tracking_iou,    # IOU threshold for tracking
                    verbose=False,
                    points=[self.clicked_point],
                    tracker=self.tracker,
                )
                # if results and len(results[0].boxes) > 0 and results[0].boxes[0].id is not None:
                #     self.clicked_point = None
                #     self.tracking_active = True
                #     self.tracked_id = results[0].boxes[0].id.item() if hasattr(results[0].boxes[0], 'id') else None
                # Filter out large boxes when clicking on a point
                if results and len(results[0].boxes) > 0:
                    # Iterate through all detected boxes and find the first one that meets our criteria
                    found_valid_object = False
                    for i, box in enumerate(results[0].boxes):
                        if not hasattr(box, 'xyxy') or not hasattr(box, 'id'):
                            continue
                            
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        frame_area = frame.shape[1] * frame.shape[0]
                        area_ratio = box_area / frame_area
                        
                        # Only track the object if it's not too large
                        if area_ratio < 0.5 and box.id is not None:
                            self.clicked_point = None
                            self.tracking_active = True
                            self.tracked_id = box.id.item()
                            print(f"Selected object with ID {self.tracked_id}, area ratio: {area_ratio:.2f}")
                            found_valid_object = True
                            break
                            
                    if not found_valid_object:
                        print(f"No suitable objects found at the clicked point (all too large or invalid)")
                else:
                    print("No objects detected at the clicked point")
            
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
                    iou=self.tracking_iou,
                    tracker=self.tracker,       
                )

            # Visualization
            if results and len(results[0].boxes) > 0:
                # Only show the tracked object
                boxes = results[0].boxes
                masks = results[0].masks
                
                # Find our tracked object
                for i, box in enumerate(boxes):
                    if hasattr(box, 'id') and box.id is not None and box.id.item() == self.tracked_id:
                        self.tracked_box = box.xyxy[0].tolist()
                        # print("Tracked box: ", box)

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

        else:
            results = self.tracking_model.track(frame, device=device, conf=0.01, persist=True)
            if results:
                # Visualize the results on the frame
                frame = results[0].plot(labels=True)
        
        # get depth
        depth_diff = -1
        if self.depth_model and self.tracking_active:
            depth = self.depth_model.infer_image(frame)
            ref_depth_value = depth[self.ref_coords.y, self.ref_coords.x]
            object_coords = Coords(int((self.tracked_box[0] + self.tracked_box[2]) / 2), int((self.tracked_box[1] + self.tracked_box[3]) / 2))
            # get depth of the tracked object
            object_depth_value = np.mean(depth[object_coords.y - 10:object_coords.y + 10, object_coords.x - 10:object_coords.x + 10])
            print("object_depth_value", object_depth_value)

            tmp_depth_diff = (ref_depth_value - object_depth_value) * 100 + 100
            if not np.isnan(tmp_depth_diff):
                depth_diff = float(tmp_depth_diff)
            
        if self.tracked_box:
            return self.tracked_box + [depth_diff]
        else:
            return None


    def set_classes(self, classes):
        if self.tracking_model_name and "yoloe" in self.tracking_model_name:
            self.tracking_model.set_classes(classes, self.tracking_model.get_text_pe(classes))
        else:
            self.tracking_model.set_classes(classes)