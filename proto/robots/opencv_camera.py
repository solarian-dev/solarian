from proto.common.configs import CameraConfig, ValueName
from lerobot.common.cameras.opencv import OpenCVCamera as OpenCVCameraImpl, OpenCVCameraConfig
from lerobot.common.cameras.configs import Cv2Rotation, ColorMode
from proto.robots.camera import Camera


rotation_map = {
    0: Cv2Rotation.NO_ROTATION,
    90: Cv2Rotation.ROTATE_90,
    180: Cv2Rotation.ROTATE_180,
    -90: Cv2Rotation.ROTATE_270,
}

color_mode_map = {
    "rgb": ColorMode.RGB,
    "bgr": ColorMode.BGR,
}

class OpenCVCamera(Camera):
    def __init__(self, cfg: CameraConfig, name: str):
        super().__init__(cfg, name)

        self.camera = OpenCVCameraImpl(OpenCVCameraConfig(
            index_or_path=int(cfg.port),
            fps=cfg.fps,
            width=cfg.width,
            height=cfg.height,
            color_mode=color_mode_map[cfg.color_mode],
            rotation=rotation_map[cfg.rotation],
        ))
    
    def connect(self):
        self.camera.connect()
    
    def disconnect(self):
        self.camera.disconnect()
    
    def read(self, value_name: ValueName):
        if value_name == ValueName.IMAGE:
            return self.camera.read()
        else:
            raise ValueError(f"Unknown value name: {value_name}")

    def is_calibrated(self):
        return True
        