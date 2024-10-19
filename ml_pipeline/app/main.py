import sys

from fastapi import FastAPI, UploadFile

sys.path.append('.')
from predict.predict import predict
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
from configs import MODEL_PATH, IMAGE_PATH

app = FastAPI()

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
main_app_predictions = Histogram(
    "main_app_predictions",
    "Histogram of predictions",
    buckets=(0.25, 0.5, 0.75, 1)
)
error_number = Counter(
    "error_number",
    "Number of errors",
)

@app.get("/")
def read_root():
    return {"Description": "This is Document Image Quality Assesment Project. Send an image to get quality score"}


@app.post("/api/diqa/predict")
def get_prediction_for_item(file: UploadFile):
    try:
        with open(IMAGE_PATH, 'wb') as f:
            f.write(file.file.read())
    except Exception:
        error_number.inc()
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    try:
        score = predict(['--model_path', MODEL_PATH, '--image_path', IMAGE_PATH], standalone_mode=False)
        main_app_predictions.observe(score)
    except Exception:
        error_number.inc()
        return {"message": "There was an error predicting"}
    return {'score': score[0]}
