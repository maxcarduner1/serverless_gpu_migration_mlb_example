"""MLflow pyfunc wrapper around an ultralytics YOLO checkpoint.

Input:  DataFrame with column `image_path` (a /Volumes/... path or a local file)
Output: DataFrame with column `detections` (JSON list of {cls, name, conf, xyxy})
"""

import json

import mlflow.pyfunc
import pandas as pd


class YoloDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from ultralytics import YOLO

        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        conf = (params or {}).get("conf", 0.25)
        out = []
        for path in model_input["image_path"]:
            r = self.model.predict(path, conf=conf, verbose=False)[0]
            dets = [
                {"cls": int(c), "name": r.names[int(c)], "conf": round(float(s), 4),
                 "xyxy": [round(float(v), 1) for v in box]}
                for box, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
            ]
            out.append(json.dumps(dets))
        return pd.DataFrame({"detections": out})
