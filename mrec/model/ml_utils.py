""" Utilties functions for machine learning models

Current utilities consist of mlflow utilities for fetching and logging data

"""

import mlflow


def yield_artifacts(run_id :str, path=None):
    """ Yield all artifacts in the specified run

    An MLFlow client is initialized and used to list the artifacts from a specified run.
    If the item is a directory, yield is done recursively.

    Usage

    >>> tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    >>> artifacts = list(yield_artifacts(run_id))

    Args:
        run_id: MLFlow run id.
        path: Directory path of the artifact

    Returns:

    """
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id: str) -> dict:
    """ Fetch params, metrics, tags, and artifacts in the specified run

    Data specific to the run is retrieved through an initialized MLFlow client.
    Logged data is fetched via helper function `yield_artifacts()`.

    Usage

    >>> # show data logged in the parent run
    >>> from pprint import pprint
    >>> print(f"\n========== {experiment_name} run ==========")
    >>> for key, data in fetch_logged_data(run.info.run_id).items():
            print("\n---------- logged {} ----------".format(key))
            pprint(data)
    >>> print(f'Saved model to {model_path}')

    Args:
        run_id: MLFlow run id

    Returns:
        dict: Parameters, metrics, tags, & artifacts

    """
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }
