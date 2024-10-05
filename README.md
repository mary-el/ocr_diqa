# Document Image Quality Assessment pipeline with Airflow and FastAPI

DIQA is an important task in document recognition. 
I've used the [SmartDoc QA dataset](https://zenodo.org/records/5293201), consisting of smartphone captured document images with ground truth texts. 

## Features to study
* Connected Components features:
  * FS - Font Size
  * SSF - Small Speckle Factor
  * TCF - Touching Characters Factor
  * WSF - White Speckle Factor
  * SWS - Small White Speckle
  * SW_1, SW_2 - Stroke Widthe mean and std
* Morphology Based Features:
  * Erosion
  * Dilation
  * Closing
  * Opening
* Noise Removal Based Features
  * Gaussian_1..4
  * Median_1..2
* Spatial Characteristic Features
  * Foreground Percent
  * Gradients_1..3
* Statistical Features
  * Entropy_1..7
  * Mean SD
  * Grounds Mean
 
## Running in docker
* docker compose up --build

* Run airflow DAGs on http://localhost:8080/home:
  - Trigger diqa_load_data to download SmartDoc dataset and get features
  - Trigger diqa_train to train a model
  - Trigger diqa_predict to predict quality score
* Predict with FastAPI on http://127.0.0.1:8000 

 
## Used articles on DIQA 
* Image quality and readability. In Proceedings of the International Conference on Image Processing
* SEDIQA: Sound Emitting Document Image Quality Assessment in a Reading Aid for the Visually Impaired
* Character-based automated human perception quality assessment in document images
* Automated image quality assessment for camera-captured OCR
* Automatic filter selection using image quality assessment
* Learning features for predicting OCR accuracy
* Metric-based no-reference quality assessment of heterogeneous document images
* Prediction of OCR accuracy using simple image features
* QUAD: Quality assessment of documents
* Quality assessment and restoration of typewritten document images
* Sorting qualities of handwritten Chinese characters for setting up a research database.
* TextNet for text-related image quality assessment
* Correlating degradation models and image quality metrics.
* CG-DIQA: No-reference document image quality assessment based on character gradient
* Document image quality assessment using discriminative sparse representation
* Combining focus measure operators to predict ocr accuracy in mobile-captured document
* A dataset for quality assessment of camera-captured document images
* No-reference document image quality assessment based on high order image statistics.
* Quality evaluation of character image database and its application.
* Sharpness estimation for document and scene images
* Automatic Parameter Selection for Denoising Algorithms Using a No-Reference Measure of Image Content
* Unsupervised Feature Learning Framework for No-reference Image Quality Assessment

## Links
* https://github.com/anastasiia-p/airflow-ml/tree/main
* https://github.com/albincorreya/ml-training-airflow-mlflow-example/tree/main
