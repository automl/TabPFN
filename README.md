# TabPFN

## Installation
```
conda create -n TabPFN python=3.7
$environment_path$/pip install -r requirements.txt
```

To run the autogluon baseline please create a separate environment and install autogluon==0.4.0, installation in the same environment as our other baselines is not possible.

## Usage
TrainingTuningAndPrediction: Train a TabPFN, Prior Tune and predict using a pretrained model.

TabularEvaluationVisualization: Run Baselines and load Baseline and TabPFN Results for comparison and plotting.

PrepareDatasets: Notebook used to inspect Datasets (Not needed to run baselines / TabPFN).

SytheticGPAblation: Ablation experiments for Gaussian Process fitting with differentiable Hyper Parameters.


