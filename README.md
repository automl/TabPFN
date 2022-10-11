# TabPFN

The TabPFN is a neural network that learned to do tabular data prediction.
This is the original CUDA-supporting pytorch impelementation.

We created a [Colab](https://colab.research.google.com/drive/194mCs6SEPEW6C0rcP7xWzcEtt1RBc8jJ#scrollTo=N8mEqwDWHD5S), that lets you play with our scikit-learn interface.

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
To run the autogluon baseline please create a separate environment and install autogluon==0.4.0, installation in the same environment as our other baselines is not possible.

## Getting started

A simple usage of our sklearn interface is:
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = TabPFNClassifier(device='cpu')
classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))
```


