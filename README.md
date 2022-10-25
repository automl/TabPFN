# TabPFN

The TabPFN is a neural network that learned to do tabular data prediction.
This is the original CUDA-supporting pytorch impelementation.

We created a [Colab](https://colab.research.google.com/drive/194mCs6SEPEW6C0rcP7xWzcEtt1RBc8jJ), that lets you play with our scikit-learn interface.

We also created two demos. One to experiment with the TabPFNs predictions (https://huggingface.co/spaces/TabPFN/TabPFNPrediction) and one to check cross-
validation ROC AUC scores on new datasets (https://huggingface.co/spaces/TabPFN/TabPFNEvaluation). Both of them run on a weak CPU, thus it can require a little bit of time.
Both demos are based on a scikit-learn interface that makes using the TabPFN as easy as a scikit-learn SVM.

## Installation

```bash
pip install tabpfn
```

If you want to evaluate our baselines, too, please install with
```bash
pip install tabpfn[baselines]
```
To run the autogluon and autosklearn baseline please create a separate environment and install autosklearn / autogluon==0.4.0, installation in the same environment as our other baselines is not possible.

## Getting started

A simple usage of our sklearn interface is:
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
# When N_ensemble_configurations > #features * #classes, no further averaging is applied.

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

# By setting normalize_with_test to True, input normalization is applied across train + test set (weak transductive setting). [default = False]
classifier.fit(X_train, y_train, normalize_with_test=False)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))
```

### Our Paper
Read our [paper](https://arxiv.org/abs/2207.01848) for more information about the setup (or contact us ☺️).
If you use our method, please cite us using
```
@misc{tabpfn,
  doi = {10.48550/ARXIV.2207.01848},
  url = {https://arxiv.org/abs/2207.01848},
  author = {Hollmann, Noah and Müller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
