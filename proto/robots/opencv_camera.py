from proto.common.configs import CameraConfig, ValueName
from proto.robots.camera import Camera
import cv2
from enum import Enum
import numpy as np


BACKEND = {
    "linux": cv2.CAP_V4L2,
    "darwin": cv2.CAP_AVFOUNDATION,  # macOS
    "win32": cv2.CAP_MSMF            # or cv2.CAP_DSHOW
}[__import__("sys").platform]


class Rotation(int, Enum):
    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90

class ColorMode(str, Enum):
    RGB = "rgb"
    BGR = "bgr"

class OpenCVCamera(Camera):
    def __init__(self, cfg: CameraConfig, name: str):
        super().__init__(cfg, name)

        if cfg.port.startswith('/dev/'):
            port = cfg.port
        else:
            port = int(cfg.port)

        print(f"Opening video file: {port}, type: {type(port)}")
        self.camera = cv2.VideoCapture(port, BACKEND)
    
    def connect(self):

        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.video_format))
        width_success = self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.width))
        height_success = self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.height))
        fps_success = self.camera.set(cv2.CAP_PROP_FPS, float(self.cfg.fps))
        
        if not self.camera.isOpened():
            print(f"Error: Could not open video file with FFMPEG backend: {self.cfg.port}")
            print("Trying with default backend...")
            self.camera = cv2.VideoCapture(self.cfg.port)
        
        if not self.camera.isOpened():
            print(f"Error: Could not open video file with any backend: {self.cfg.port}")
            print("This might be due to AV1 codec support issues.")
            print("Please ensure you have the necessary codecs installed.")
            exit(1)
        
        # Get video properties
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        frame_count_total = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"AV1 Video loaded successfully:")
        print(f"  Path or port: {self.cfg.port}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {frame_count_total}")
        print(f"  Duration: {frame_count_total/fps:.2f} seconds")
        print("Note: AV1 codec warnings are normal and won't affect functionality")
        print("The script will continue processing despite these warnings.\n")
    
    def disconnect(self):
        self.camera.release()
    
    def read(self, value_name: ValueName):
        if value_name == ValueName.IMAGE:
            ret, frame = self.camera.read()
            if not ret:
                raise ValueError(f"Could not read frame")

            if isinstance(frame, bytes):
                frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)

            processed_image = frame

            # Color mode
            if self.cfg.color_mode == ColorMode.RGB.value:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            # Rotation
            if self.cfg.rotation == Rotation.ROTATE_90.value:
                processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)
            elif self.cfg.rotation == Rotation.ROTATE_180.value:
                processed_image = cv2.rotate(processed_image, cv2.ROTATE_180_CLOCKWISE)
            elif self.cfg.rotation == Rotation.ROTATE_270.value:
                processed_image = cv2.rotate(processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Resize
            if self.cfg.resize_width is not None or self.cfg.resize_height is not None:
                if processed_image.shape[0] != self.cfg.resize_height or processed_image.shape[1] != self.cfg.resize_width:
                    processed_image = cv2.resize(processed_image, (self.cfg.resize_width, self.cfg.resize_height), interpolation=cv2.INTER_AREA)
            
            # Ensure we're returning a numpy array
            if isinstance(processed_image, bytes):
                processed_image = cv2.imdecode(np.frombuffer(processed_image, dtype=np.uint8), cv2.IMREAD_COLOR)
                # Apply color mode again if needed
                if self.cfg.color_mode == ColorMode.RGB.value:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Final check to ensure we have a numpy array
            if not isinstance(processed_image, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray, got {type(processed_image)}. This is likely due to a processing error.")
            
            # print("Processedd image type: ", type(processed_image))
            return processed_image
        else:
            raise ValueError(f"Unknown value name: {value_name}")

    def is_calibrated(self):
        return True


    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying serial port is open."""
        return self.camera.isOpened()


    def needs_setup(self) -> bool:
        """Check if the component needs setup."""
        return False