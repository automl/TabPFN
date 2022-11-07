from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import Callable

import torch

import tabpfn.scripts.tabular_baselines as tb
from tabpfn.datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from tabpfn.scripts.tabular_baselines import clf_dict
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts.tabular_metrics import (accuracy_metric, auc_metric,
                                            calculate_score, cross_entropy)

HERE = Path(__file__).parent.resolve().absolute()

METRICS = {"roc": auc_metric, "cross_entropy": cross_entropy, "acc": accuracy_metric}

PREDFINED_DATASET_PATHS = HERE / "tabpfn" / "datasets"
PREDEFINED_DATASET_COLLECTIONS = {
    "cc_valid": {
        "ids": open_cc_valid_dids,
        "path": PREDFINED_DATASET_PATHS / "cc_valid_datasets_multiclass.pickle",
    },
    "cc_test": {
        "ids": open_cc_dids,
        "path": PREDFINED_DATASET_PATHS / "cc_test_datasets_multiclass.pickle",
    },
}


@dataclass
class Dataset:
    """Small helper class just to name entries in the loaded pickled datasets."""

    name: str
    X: torch.Tensor
    y: torch.Tensor
    categorical_columns: list[int]
    attribute_names: list[str]
    # Seems to be some things about how the dataset was constructed
    info: dict
    # Only 'multiclass' is known?
    task_type: str

    @classmethod
    def fetch(
        self, identifier: str | int | list[int], only: Callable | None = None
    ) -> list[Dataset]:
        if isinstance(identifier, str) and identifier in PREDEFINED_DATASET_COLLECTIONS:
            datasets = Dataset.from_predefined(identifier)
        elif isinstance(identifier, int):
            identifier = [identifier]
            datasets = Dataset.from_openml(identifier)
        elif isinstance(identifier, list):
            datasets = Dataset.from_openml(identifier)
        else:
            raise ValueError(identifier)

        if only:
            return list(filter(only, datasets))
        else:
            return datasets

    @classmethod
    def from_pickle(self, path: Path, task_types: str) -> list[Dataset]:
        with path.open("rb") as f:
            raw = pickle.load(f)

        return [Dataset(*entry, task_type=task_types) for entry in raw]  # type: ignore

    @classmethod
    def from_predefined(self, name: str) -> list[Dataset]:
        assert name in PREDEFINED_DATASET_COLLECTIONS
        path = PREDEFINED_DATASET_COLLECTIONS[name]["path"]

        return Dataset.from_pickle(path, task_types="multiclass")

    @classmethod
    def from_openml(
        self,
        dataset_id: int | list[int],
        filter_for_nan: bool = False,
        min_samples: int = 100,
        max_samples: int = 2_000,
        num_feats: int = 100,
        return_capped: bool = False,
        shuffled: bool = True,
        multiclass: bool = True,
    ) -> list[Dataset]:
        # TODO: should be parametrized, defaults taken from ipy notebook
        if not isinstance(dataset_id, list):
            dataset_id = [dataset_id]

        datasets, _ = load_openml_list(
            dataset_id,
            filter_for_nan=filter_for_nan,
            num_feats=num_feats,
            min_samples=min_samples,
            max_samples=max_samples,
            return_capped=return_capped,
            shuffled=shuffled,
            multiclass=multiclass,
        )
        return [
            Dataset(  # type: ignore
                *entry,
                task_type="multiclass" if multiclass else "binary",
            )
            for entry in datasets
        ]


# Predefined methods with `no_tune={}` inidicating they are not tuned
METHODS = {
    # svm
    "svm": tb.svm_metric,
    "svm_default": partial(tb.svm_metric, no_tune={}),
    # gradient boosting
    "gradient_boosting": tb.gradient_boosting_metric,
    "gradient_boosting_default": partial(tb.gradient_boosting_metric, no_tune={}),
    # gp
    "gp": clf_dict["gp"],
    "gp_default": partial(
        clf_dict["gp"],
        no_tune={"params_y_scale": 0.1, "params_length_scale": 0.1},
    ),
    # lightgbm
    "lightgbm": clf_dict["lightgbm"],
    "lightgbm_default": partial(clf_dict["lightgbm"], no_tune={}),
    # xgb
    "xgb": clf_dict["xgb"],
    "xgb_default": partial(clf_dict["xgb"], no_tune={}),
    # random forest
    "random_forest": clf_dict["random_forest"],
    "rf_default": partial(clf_dict["random_forest"], no_tune={}),
    "rf_default_n_estimators_10": partial(
        clf_dict["random_forest"], no_tune={"n_estimators": 10}
    ),
    "rf_default_n_estimators_32": partial(
        clf_dict["random_forest"], no_tune={"n_estimators": 32}
    ),
    # knn
    "knn": clf_dict["knn"],
    # logistic classification
    "logistic": clf_dict["logistic"],
    # Transformers
    "transformer_cpu_N_1": partial(
        clf_dict["transformer"], device="cpu", N_ensemble_configurations=1
    ),
    "transformer_cpu_N_8": partial(
        clf_dict["transformer"], device="cpu", N_ensemble_configurations=8
    ),
    "transformer_cpu_N_32": partial(
        clf_dict["transformer"], device="cpu", N_ensemble_configurations=32
    ),
    "transformer_gpu_N_1": partial(
        clf_dict["transformer"], device="cuda", N_ensemble_configurations=1
    ),
    "transformer_gpu_N_8": partial(
        clf_dict["transformer"], device="cuda", N_ensemble_configurations=8
    ),
    "transformer_gpu_N_32": partial(
        clf_dict["transformer"], device="cuda", N_ensemble_configurations=32
    ),
}


def eval_method(
    dataset: Dataset,
    label: str,
    classifier_evaluator: Callable,
    max_time: int | None,
    metric_used: Callable,
    split: int,
    result_path: Path,
    append_metric: bool = True,
    fetch_only: bool = False,
    verbose: bool = False,
    eval_positions: list[int] = [1000],
    bptt: int = 2000,
    overwrite: bool = False,
):
    """Evaluate a given method."""
    if max_time is not None:
        label += f"_time_{max_time}"

    if append_metric:
        label += f"_{tb.get_scoring_string(metric_used, usage='')}"

    if isinstance(classifier_evaluator, partial):
        device = classifier_evaluator.keywords.get("device", "cpu")
    else:
        device = "cpu"

    return evaluate(
        datasets=[
            [
                dataset.name,
                dataset.X,
                dataset.y,
                dataset.categorical_columns,
                dataset.attribute_names,
                dataset.info,
            ]
        ],  # Expects a list of datasets which are stored as a list of properties
        model=classifier_evaluator,
        method=label,
        bptt=bptt,
        base_path=result_path,
        eval_positions=eval_positions,
        device=device,
        max_splits=1,
        overwrite=overwrite,
        save=True,
        metric_used=metric_used,
        path_interfix=dataset.task_type,
        fetch_only=fetch_only,
        split_number=split,
        verbose=verbose,
        max_time=max_time,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=Path,
        help="Where the results path is",
        default=HERE,
    )
    parser.add_argument("--gpu", action="store_true", help="GPU's available?")
    parser.add_argument(
        "--times",
        nargs="+",
        type=int,
        default=[30],
        help="Times to evaluate (seconds)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[0],
        help="The splits to evaluate",
    )

    parser.add_argument(
        "--validation_datasets",
        nargs="+",
        type=int,
        help="The validation datasets",
    )
    parser.add_argument(
        "--test_datasets",
        nargs="+",
        type=int,
        help="The test datasets",
    )

    parser.add_argument(
        "--optimization_metrics",
        type=str,
        choices=METRICS,
        help="Metrics to optimize for (if possible)",
        default=["roc"],
    )
    parser.add_argument(
        "--result_metrics",
        type=str,
        nargs="+",
        choices=METRICS,
        help="Metrics to calculate for results",
        default=["roc", "cross_entropy", "acc"],
    )

    parser.add_argument(
        "--methods",
        choices=METHODS.keys(),
        nargs="+",
        type=str,
        help="The methods to evaluate",
        default=["svm_default"],
    )
    parser.add_argument(
        "--fetch_only",
        action="store_true",
        help="Whether to only fetch results and not run anything",
    )

    # Transformer args
    parser.add_argument(
        "--bptt",
        type=int,
        help="Transformer sequence length",
        default=2000,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite results if they already exist",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.validation_datasets:
        args.validation_datasets = "cc_valid"

    if not args.test_datasets:
        args.test_datasets = "cc_test"

    # We need to create some directories for this to work
    (args.result_path / "results" / "tabular" / "multiclass").mkdir(
        parents=True, exist_ok=True
    )

    # We ignore the flags datasets
    filter_f = lambda d: d.name != "flags"  # noqa: ignore

    valid_datasets = Dataset.fetch(args.validation_datasets, only=filter_f)
    test_datasets = Dataset.fetch(args.test_datasets, only=filter_f)
    all_datasets = valid_datasets + test_datasets

    results = {}
    for dataset, method, metric, time, split in product(
        all_datasets,
        args.methods,
        args.optimization_metrics,
        args.times,
        args.splits,
    ):
        metric_f = METRICS[metric]
        metric_name = tb.get_scoring_string(metric_f, usage="")
        r = eval_method(
            dataset=dataset,
            label=method,
            result_path=args.result_path,
            classifier_evaluator=METHODS[method],
            eval_positions=[1000],  # It's a constant basically
            fetch_only=args.fetch_only,
            verbose=args.verbose,
            max_time=time,
            metric_used=metric_f,
            split=split,
            overwrite=args.overwrite,
        )
        results.update(r)

    # This will update the results in place
    for metric in args.result_metrics:
        metric_f = METRICS[metric]
        calculate_score(
            metric=metric_f,
            name=metric,
            global_results=results,
            ds=all_datasets,
            eval_positions=[1_000],
        )

    # We also do some other little bits
