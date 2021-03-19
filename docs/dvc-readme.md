# mrec-dvc-examples
This is a collection of DVC project examples that you can directly run with dvc CLI commands or directly using
 Python.

The goal is provide you with additional set of samples, focusing on machine learning and deep learning examples, to
 get you quickly started on DVC.

## 1. Preparation

To begin let's start by version controlling our raw dataset.
```
dvc add dataset/raw
git add dataset/raw.dvc data/.gitignore
git commit -m "Add raw data"
```
<details>
  <summary>Expand to see what happens under the hood</summary>

    dvc add moved the data to the project's cache, and linked* it back to the workspace.
    ```
    ```

</details>

## 2. Training our first model

This is a simple Scikit-Learn NuSVC that is similar to SVC but uses a paramter to control the number of suport
 vectors.

 The arguments to run this simple NuSVC network model are as follows:
 - `--random_state`: Seed value. Default is `20170428`
 - `--degree`: Degree of polynomial kernel function. Default is `2`.
 - `--kernel`: Kernel type to be used in algorithm. Default is `rbf`
 - `--nu`: An upper bound on fraction of margin errors and a lower bound of the fraction of support vectors. Default
  is `0.25`
  - `--gamma`: Kernel coefficient. Default is `scale`

How to run the model
```
(mrec) C02D256NMD6R:mrec ktle2$ python mrec/train_mrec.py
    ...
    ..
    .
[2020-12-28 09:40:53,989] [INFO] [__main__::main::126] Saved model to /Users/ktle2/personal_projects/mrec/models/clean_data_model/model.joblib
```

**How to work in DVC to version control**

DVC provides data pipelines for **reproducibility**. Here's how to automatically capture the different model version
 and have it track it in the version control.
```
dvc run -n prepare_dataset \
          -d mrec/data/make_dataset.py -d dataset/raw \
          -o dataset/processed \
          python mrec/data/make_dataset.py

dvc run --force -n train \
		  -p train.random_state,train.degree,train.kernel,train.nu,train.gamma \
          -d mrec/train_mrec.py -d dataset/processed \
          -o models/clean_data_model \
          python mrec/train_mrec.py
```
These will add stages to our `dvc.yaml` and begin to automatically track any changes to the resulting model and
 processed dataset.

 Let's save and tag this model.
```
To track the changes with git, run:

        git add dvc.lock dvc.yaml
(mrec) C02D256NMD6R:mrec ktle2$ git add dvc.lock dvc.yaml
(mrec) C02D256NMD6R:mrec ktle2$ git commit -m "First model, trained with 1000 entries"
(mrec) C02D256NMD6R:mrec ktle2$ git tag -a "v1.0" -m "model v1.0, 1000 entries"
(mrec) C02D256NMD6R:mrec ktle2$ git push
(mrec) C02D256NMD6R:mrec ktle2$ dvc push
```

## 3. Tuning our trained Model
```
train:
    random_state: 20170428
    degree: 2 -->
    kernel: rbf
    nu: 0.25
    gamma: scale
```

```
(mrec) C02D256NMD6R:mrec ktle2$ dvc repro
'dataset/raw.dvc' didn't change, skipping
Stage 'prepare_dataset' didn't change, skipping
Running stage 'train' with command:
        python mrec/train_mrec.py

Updating lock file 'dvc.lock'

To track the changes with git, run:

        git add dvc.lock
Use `dvc push` to send your updates to remote storage.
```

```
(mrec) C02D256NMD6R:mrec ktle2$ git add dvc.lock
(mrec) C02D256NMD6R:mrec ktle2$ git commit -m "Second model, tuned degree parameter"
(mrec) C02D256NMD6R:mrec ktle2$ git tag -a "v2.0" -m "model v2.0, tuned degree parameter"
(mrec) C02D256NMD6R:mrec ktle2$ git push
(mrec) C02D256NMD6R:mrec ktle2$ dvc push
```

## 4. Reverting back to our old model
There are two ways of doing this: a full workspace checkout or checkout of a specific data or model file. Let's consider the full checkout first. It's pretty straightforward:
```
git checkout v1.0
dvc checkout
```
These commands will restore the workspace to the first snapshot we made: code, data files, model, all of it. DVC optimizes this operation to avoid copying data or model files each time. So dvc checkout is quick even if you have large datasets, data files, or models.

On the other hand, if we want to keep the current code, but go back to the previous dataset version, we can do something like this:

```
git checkout v1.0 data.dvc
dvc checkout data.dvc
```
f you run git status you'll see that data.dvc is modified and currently points to the v1.0 version of the dataset, while code and model files are from the v2.0 tag.

# FAQ
When to use `dvc add` vs. `dvc run`?

`dvc add` makes sense when you need to keep track of different versions of datasets or model files that come from source projects.

When you have a script that takes some data as an input and produces other data outputs, a better way to capture them is to use `dvc run`

# How to start up your own dvc-initialized project from scratch
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

dvc run --force -n train \
		  -p train.random_state,train.degree,train.kernel,train.nu,train.gamma \
          -d mrec/train_mrec.py -d dataset/processed \
          -o models/clean_data_model \
          python mrec/train_mrec.py
```
