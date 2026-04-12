import threading

from solarian.perception.processor import Processor


class BBoxProcessor(Processor):
    """
    Stateful processor for bounding box tracking via YOLO models.

    Tracks a user-clicked object across video frames.  Internal tracking
    state (model weights, tracked object ID, etc.) persists across calls
    so that the same object is followed until explicitly stopped.

    Thread-safe: the relay server may call the processor from the state
    relay thread while ``on_event`` is called from the action relay thread.

    The ML model is loaded lazily on first call so that importing the
    observation schema on machines without GPU/ultralytics is free.

    Config (loaded from YAML via omegaconf)::

        tracking_model: models/FastSAM-x.pt
        confidence: 0.6
        iou: 0.2
        tracker: models/custom_tracker.yaml
        max_area_ratio: 0.5
    """

    EMPTY_BOX = (0, 0, 0, 0)

    def __init__(self, inputs, output, conf: str | None = None):
        super().__init__(inputs, output, conf)

        self._model = None
        self._device = None
        self._lock = threading.Lock()

        # Tracking state
        self.clicked_point = None
        self.tracking_active = False
        self.tracked_id = None
        self.tracked_box = self.EMPTY_BOX

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _cfg_get(self, key, default=None):
        assert self.cfg is not None, "BBoxProcessor requires a conf file"
        return self.cfg.get(key, default)

    @property
    def tracking_model_name(self):
        return self._cfg_get("tracking_model")

    @property
    def confidence(self):
        return self._cfg_get("confidence", 0.6)

    @property
    def iou(self):
        return self._cfg_get("iou", 0.2)

    @property
    def tracker(self):
        return self._cfg_get("tracker", "models/custom_tracker.yaml")

    @property
    def max_area_ratio(self):
        return self._cfg_get("max_area_ratio", 0.5)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return

        import torch
        from ultralytics import YOLO, FastSAM

        self._device = (
            0
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        if "FastSAM" in self.tracking_model_name:
            self._model = FastSAM(self.tracking_model_name)
        else:
            self._model = YOLO(self.tracking_model_name)

    # ------------------------------------------------------------------
    # External event API
    # ------------------------------------------------------------------

    def track_object(self, clicked_point):
        with self._lock:
            self.clicked_point = clicked_point

    def stop_tracking(self):
        with self._lock:
            self.clicked_point = None
            self.tracked_id = None
            self.tracking_active = False
            self.tracked_box = self.EMPTY_BOX

    def on_event(self, event_name: str, data=None):
        if event_name == "click":
            self.track_object(data)
        elif event_name == "stop_tracking":
            self.stop_tracking()

    # ------------------------------------------------------------------
    # Processor interface
    # ------------------------------------------------------------------

    def process(self, inputs):
        self._ensure_model()
        assert self._model is not None

        image = inputs[0].value if hasattr(inputs[0], "value") else inputs[0]
        if image is None:
            return self.EMPTY_BOX

        with self._lock:
            return self._process(image)

    # ------------------------------------------------------------------
    # Core tracking logic (must be called under self._lock)
    # ------------------------------------------------------------------

    def _process(self, frame):
        assert self._model is not None
        results = None

        if self.clicked_point:
            results = self._model.track(
                frame,
                device=self._device,
                conf=self.confidence,
                persist=True,
                iou=self.iou,
                verbose=False,
                points=[self.clicked_point],
                tracker=self.tracker,
            )
            boxes = results[0].boxes if results else None
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    if not hasattr(box, "xyxy") or not hasattr(box, "id"):
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_area = (x2 - x1) * (y2 - y1)
                    frame_area = frame.shape[1] * frame.shape[0]
                    if box_area / frame_area < self.max_area_ratio and box.id is not None:
                        self.clicked_point = None
                        self.tracking_active = True
                        self.tracked_id = box.id.item()
                        break

        elif self.tracking_active:
            results = self._model.track(
                frame,
                device=self._device,
                conf=self.confidence,
                persist=True,
                iou=self.iou,
                verbose=False,
                tracker=self.tracker,
            )
        else:
            return self.EMPTY_BOX

        # Extract the tracked object's box from results
        boxes = results[0].boxes if results else None
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                if (
                    hasattr(box, "id")
                    and box.id is not None
                    and box.id.item() == self.tracked_id
                ):
                    self.tracked_box = tuple(
                        int(v) for v in box.xyxy[0].tolist()
                    )
                    return self.tracked_box

        return self.tracked_box
