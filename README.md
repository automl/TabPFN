# TabPFN

We created a Colab notebook, that lets you interact with our scikit-learn interface at [https://colab.research.google.com/drive/1J0l1AtMVH1KQ7IRbgJje5hMhKHczH7-?usp=sharing](https://colab.research.google.com/drive/1J0l1AtMV_H1KQ7IRbgJje5hMhKHczH7-?usp=sharing)

We also created two demos. One to experiment with the TabPFNs predictions (https://huggingface.co/spaces/TabPFN/TabPFNPrediction) and one to check cross-
validation ROC AUC scores on new datasets (https://huggingface.co/spaces/TabPFN/TabPFNEvaluation). Both of them run on a weak CPU, thus it can require a little bit of time.
Both demos are based on a scikit-learn interface that makes using the TabPFN as easy as a scikit-learn SVM.

## Installation
```
conda create -n TabPFN python=3.7
$environment_path$/pip install -r requirements.txt
```

To run the autogluon baseline please create a separate environment and install autogluon==0.4.0, installation in the same environment as our other baselines is not possible.


