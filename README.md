# Document Image Quality Assessment pipeline with Airflow and FastAPI
![DIQA](https://github.com/user-attachments/assets/7b8fa088-8c0b-4590-a61c-6adfa556452e)

Document Image Quality Assessment (DIQA) evaluates the suitability of digitized documents for subsequent processing. It looks for things like blurriness, stains, and uneven lighting that can make it hard to read. This is crucial for Optical Character Recognition (OCR) because OCR software works much better with clear images. Poor quality images lead to more mistakes in OCR, wasting time and money. DIQA helps by identifying bad images beforehand, allowing for improvement (like sharpening or brightening) or flagging them for manual review, ensuring more accurate and efficient OCR results.


This research utilized the [SmartDoc QA dataset](https://zenodo.org/records/5293201), comprising smartphone-captured document images with ground truth text annotations. The pipeline was developed as a practical application of various ML engineering tools and best practices, including:

* **Containerization:** Docker
* **Workflow Orchestration:** Apache Airflow
* **API Serving:** FastAPI
* **Version Control & Reproducibility:** DVC (Data Version Control)
* **Experiment Management:** MLflow
* **Monitoring:** Prometheus
* **Visualization:** Grafana
* **Database:** PostgreSQL
* **Storage:** Amazon S3

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

* Run airflow DAGs on http://localhost:8080
  - Trigger diqa_load_data to download SmartDoc dataset and get features
  - Trigger diqa_train to train a model
  - Trigger diqa_predict to predict quality score
* Predict with FastAPI on http://127.0.0.1:8000/docs
* Run model selection [here](https://github.com/mary-el/ocr_diqa/blob/master/notebooks/model_selection.ipynb) and look at the results at http://127.0.0.1:600 
* Look at metrics with Prometheus on http://localhost:9090
* Visualize them with Grafana on http://localhost:1234
* Reproduce a DVC pipeline with *dvc repro* and push it to s3 storage with *dvc push*

## Results
Pearson's Linear correlation coefficient measures the strength of the linear relationship between two variables. If there is a strong linear relationship, the correlation coefficient is close to 1 or −1, and 0 means no linear relationship. 
|Algoritm|PLCC Metric|
|--------|-----------|
|Elastic |0.6        |
|Linear  |0.61       |
|Ridge   |0.62       |
|Tree    |0.77       |
|XGBoost |0.89       |
|SVR     |0.89       |
|CatBoost|0.90       |
 
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
