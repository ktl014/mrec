# mrec-dvc-mlflow-examples
This is a collection of DVC project examples that you can directly run with dvc and mlflow CLI commands or directly
 using
 Python.

 ## The ideal workflow of DVC + MLFlow
```
dvc init
mkdir -p /tmp/dvc-storage-mrec-v3
dvc remote add -d myremote /tmp/dvc-storage-mrec-v3
git commit .dvc/config -m "Add local remote"

dvc add dataset/raw

git add dataset/raw.dvc

dvc run -n prepare_dataset \
          -d mrec/data/make_dataset.py -d dataset/raw \
          -o dataset/processed \
          python mrec/data/make_dataset.py
```

Include logging into training script:
```
mrec/train_mrec.py

def main():
    """Train the best model"""
    experiment_name = 'train-mrec_v1.0.0'
    mlflow.set_experiment(experiment_name)
    logger.info(f'Beginning experiment {experiment_name}...')

    run_name = f'model-run-{datetime.today().strftime("%Y%m%d_%H:%M:%S")}'
    with mlflow.start_run(run_name=run_name) as run:
        params = yaml.safe_load(open('params.yaml'))['train']
        model = NuSVC(**params)
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metrics({"accuracy": compute_acc(), "precision": compute_prec(),...,}

        # Save model artifact
        model_path = os.path.join(cleaned_data_dir, 'model.joblib')
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, 'models/clean_data_model')
```

Followed by this run:
```
python mrec/train_mrec.py
```
Open this in a separate terminal to keep it active:
```
mlflow ui
```

Let's add this into our dvc data pipeline now
```
dvc run --force -n train \
		  -p train.random_state,train.degree,train.kernel,train.nu,train.gamma \
          -d mrec/train_mrec.py -d dataset/processed \
          -o models/clean_data_model \
          python mrec/train_mrec.py
```

Next lets save this model.
```
To track the changes with git, run:

        git add dvc.lock dvc.yaml
(mrec) C02D256NMD6R:mrec ktle2$ git add dvc.lock dvc.yaml
(mrec) C02D256NMD6R:mrec ktle2$ git commit -m "First model, trained with 1000 entries"
(mrec) C02D256NMD6R:mrec ktle2$ git tag -a "v1.0" -m "model v1.0, 1000 entries"
(mrec) C02D256NMD6R:mrec ktle2$ git push
(mrec) C02D256NMD6R:mrec ktle2$ dvc push
```
Now let's tune the model.
```
parameters:
degree: 3
```
```
dvc repro
```
We should be able to see a run to compare within mlflow.
Time to save this new model now.
```
(mrec) C02D256NMD6R:mrec ktle2$ git add dvc.lock
(mrec) C02D256NMD6R:mrec ktle2$ git commit -m "Second model, tuned degree parameter"
(mrec) C02D256NMD6R:mrec ktle2$ git tag -a "v2.0" -m "model v2.0, tuned degree parameter"
(mrec) C02D256NMD6R:mrec ktle2$ git push
(mrec) C02D256NMD6R:mrec ktle2$ dvc push
```
