import json
from collections import deque

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import LOGGER as logger

from byte_tracker import BYTETracker


class Namespace:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


class Box:
    def __init__(self, x: int, y: int, X: int, Y: int):
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y

    def __repr__(self):
        return f"Box({self.x}, {self.y}, {self.X}, {self.Y})"


class MatchManager:
    def __init__(self, frame_rate=None):
        self.__images: deque[np.ndarray] = deque(maxlen=2)
        self.__bb1_boxes: deque[Box] = deque(maxlen=2)
        self.__bb2_boxes: deque[Box] = deque(maxlen=2)
        self.__boxes = (
            self.__bb1_boxes,
            self.__bb2_boxes,
        )

        self.frame_rate = frame_rate
        self.dataframe: list[dict] = []
        self.temp_row = {}

    def get_battle_summary(self):
        start = self.dataframe[0]["at_second"]
        end = self.dataframe[-1]["at_second"]

        bb0 = self.dataframe[-1]["beyblade_0_pixel_diff"]
        bb1 = self.dataframe[-1]["beyblade_1_pixel_diff"]

        if bb0 == bb1 == 0 or bb0 == bb1 != 0:
            winner = "draw"
        elif bb0 == 0:
            winner = "beyblade_1"
        elif bb1 == 0:
            winner = "beyblade_0"
        else:
            winner = "draw"

        return {
            "duration": end - start,
            "winner": winner,
        }

    def __update_dataframe(self):
        if self.temp_row and any(key.endswith("box") for key in self.temp_row):
            self.dataframe.append(self.temp_row)
            self.temp_row = {}

    def update_image(self, frame: np.ndarray, frame_index: int):
        self.__images.append(frame)

        self.temp_row["index"] = frame_index
        if self.frame_rate:
            self.temp_row["at_second"] = frame_index / self.frame_rate

    def update_box(self, beyblade_id, box: tuple[int] | list[int]):
        self.__boxes[beyblade_id].append(Box(*box))
        self.temp_row[f"beyblade_{beyblade_id}_box"] = box

    def is_battle_over(self):
        flag = False

        if len(self.__bb1_boxes) == 2 and len(self.__bb2_boxes) == 2:
            for i, bb in enumerate(self.__boxes):
                x = min(bb[0].x, bb[1].x)
                y = max(bb[0].y, bb[1].y)
                X = min(bb[0].X, bb[1].X)
                Y = max(bb[0].Y, bb[1].Y)

                diff = cv2.absdiff(
                    self.__images[0][y:Y, x:X], self.__images[1][y:Y, x:X]
                )
                diff = np.count_nonzero(diff > 100)
                self.temp_row[f"beyblade_{i}_pixel_diff"] = diff
                if diff == 0:
                    flag = True

        self.__update_dataframe()

        return flag

    def export_data(self):
        pd.DataFrame(self.dataframe).to_csv("battle_data.csv", index=False)

        with open("battle_summary.json", "w") as f:
            json.dump(
                self.get_battle_summary(),
                f,
                indent=4,
            )


def pad2square(image: np.ndarray, fill_color=(0, 0, 0)):
    height, width, _ = image.shape
    size = max(height, width)

    top = bottom = (size - height) // 2
    left = right = (size - width) // 2

    if (size - height) % 2 != 0:
        bottom += 1
    if (size - width) % 2 != 0:
        right += 1

    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=fill_color,
    )


if __name__ == "__main__":
    cap = cv2.VideoCapture("battle.mp4")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = cv2.VideoWriter(
        filename="output.mp4", fourcc=fourcc, fps=frame_rate, frameSize=(size, size)
    )

    match_manager = MatchManager(frame_rate=frame_rate)
    model = YOLO("runs/detect/train2/weights/best.pt")
    tracker = BYTETracker(
        Namespace(
            {
                "track_high_thresh": 0.5,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.6,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
            }
        ),
        frame_rate=frame_rate,
    )

    while True:
        retval, origin_frame = cap.read()
        if retval is False:
            logger.info("Video ended, no winner")
            break

        frame = pad2square(origin_frame)
        match_manager.update_image(frame=frame, frame_index=frame_index)

        # single batch
        results: Results = model.predict(
            conf=0.5,
            imgsz=320,
            source=frame,
            verbose=False,
        )[0]

        ids = tracker.update(results.boxes, img=frame)

        boxes = results.boxes.xyxy.tolist()
        for i, (box, id) in enumerate(zip(boxes, ids)):
            x, y, X, Y = map(int, box)
            frame = cv2.rectangle(frame, (x, y), (X, Y), (0, 255, 0), 2)
            frame = cv2.putText(
                frame,
                str(id),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 0),
                lineType=1,
                thickness=3,
            )
            match_manager.update_box(beyblade_id=id, box=(x, y, X, Y))

        if match_manager.is_battle_over():
            logger.info("Battle ended because one did not move")
            break

        if cv2.waitKey(1) == 113:
            break

        cv2.imshow("frame", frame)
        writer.write(frame)
        frame_index += 1

    match_manager.export_data()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
