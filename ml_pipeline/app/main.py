import sys

from fastapi import FastAPI, UploadFile

sys.path.append('.')
from predict.predict import predict
from configs import MODEL_PATH, IMAGE_PATH

app = FastAPI()


@app.get("/")
def read_root():
    return {"Description": "This is Document Image Quality Assesment Project. Send an image to get quality score"}


@app.post("/api/diqa/predict")
def get_prediction_for_item(file: UploadFile):
    try:
        with open(IMAGE_PATH, 'wb') as f:
            f.write(file.file.read())
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    score = predict(['--model_path', MODEL_PATH, '--image_path', IMAGE_PATH], standalone_mode=False)
    return {'score': score[0]}
