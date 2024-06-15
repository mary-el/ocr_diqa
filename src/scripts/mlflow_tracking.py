import mlflow
from mlflow.models import infer_signature
from predict_quality import train_model, split_dataset, load_split_ds, evaluate_model
from sklearn.svm import SVR


mlflow.set_tracking_uri('http://127.0.0.1:8080')
mlflow.set_experiment("Document Image Quality Assessment")

ds_file = r'data/ds_500.pkl'
ds_split_file = r'data/ds_500_split.pkl'
split_dataset(ds_file, ds_split_file)
X_train, X_test, y_train, y_test = load_split_ds(ds_split_file)
params = {'kernel': 'rbf',
          'gamma': 0.5}
model = SVR(**params)
pipe = train_model(model, X_train, y_train)
y_test, y_pred, results = evaluate_model(pipe, X_test, y_test)

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(results)
    mlflow.set_tag("Training Info", "Basic SVR model")
    signature = infer_signature(X_train, pipe.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path="diqa_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="diqa",
    )

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)
print(predictions)
