from typing import Any, Generator, List, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore
from .config import config


class Tracker:
    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        self.model = YOLO("resources/" + model_path)
        self.roi_points: List[List[int]] = config.roi
        self.source: str = config.camera_url

    def draw_roi(self) -> List[List[int]]:
        class RoiState:
            def __init__(self):
                self.points: List[List[int]] = []
                self.drawing: bool = False

        roi_state = RoiState()

        def mouse_callback(
            event: int, x: int, y: int, flags: int, param: Any
        ) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_state.points.append([x, y])
                roi_state.drawing = True
            elif event == cv2.EVENT_RBUTTONDOWN and len(roi_state.points) > 2:
                roi_state.drawing = False
                cv2.setMouseCallback("Draw ROI", lambda event, x, y, flags, param: None)

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError("Could not open video stream")
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Could not read frame")

        cv2.namedWindow("Draw ROI")
        cv2.setMouseCallback("Draw ROI", mouse_callback)
        print(
            "Left-click to add ROI points, right-click to close polygon (min 3 points)."
        )

        while roi_state.drawing or len(roi_state.points) < 3:
            temp_frame = frame.copy()
            if roi_state.points:
                cv2.polylines(
                    temp_frame,
                    [np.array(roi_state.points)],
                    False,
                    (0, 255, 0),
                    2,
                )
                for point in roi_state.points:
                    cv2.circle(temp_frame, tuple(point), 5, (0, 0, 255), -1)
            cv2.imshow("Draw ROI", temp_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(roi_state.points) < 3:
            raise ValueError("ROI must have at least 3 points")
        self.roi_points = roi_state.points
        return roi_state.points

    def track_and_capture(
        self,
    ) -> Generator[
        Tuple[Any, List[Tuple[int, bool, Optional[np.ndarray]]]], None, None
    ]:
        results = self.model.track(
            source=self.source,
            stream=True,
            persist=True,
            classes=[2],  # Car class (COCO ID 2)
            verbose=False,
        )
        roi_poly = (
            np.array(self.roi_points, np.int32)
            if self.roi_points and len(self.roi_points) > 2
            else None
        )

        for result in results:
            original_frame = result.orig_img
            display_frame = result.plot()
            if roi_poly is not None:
                cv2.polylines(display_frame, [roi_poly], True, (0, 255, 0), 2)

            vehicles: List[Tuple[int, bool, Optional[np.ndarray]]] = []
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id) if box.id is not None else -1
                    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                    in_roi = roi_poly is not None and any(
                        cv2.pointPolygonTest(roi_poly, corner, False) >= 0
                        for corner in corners
                    )
                    car_crop = original_frame[y1:y2, x1:x2] if in_roi else None
                    vehicles.append((track_id, in_roi, car_crop))

            yield display_frame, vehicles

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
