import unittest
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch import nn
from tabpfn import TabPFNClassifier


class TestLoadModuleOnlyInference(unittest.TestCase):
    def test_main(self):
        x, y = load_breast_cancer(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        x_train = torch.from_numpy(x_train)
        x_train.requires_grad = True

        x_test = torch.from_numpy(x_test)
        x_test.requires_grad = True

        old_loss = 1.0

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

        classifier = TabPFNClassifier(device=device,
                                      N_ensemble_configurations=3,
                                      no_preprocess_mode=True,
                                      no_grad=False)
        for i in range(10):
            classifier.fit(x_train, y_train)

            logits = classifier.predict_proba(x_test, return_logits=True)

            loss = nn.CrossEntropyLoss()(logits, torch.from_numpy(y_test).long().to(device))
            current_loss = float(loss)
            loss.backward()

            rate = 0.01

            x_test = x_test.detach() - rate * x_test.grad
            x_test.requires_grad = True

            x_train = x_train.detach() - rate * x_train.grad
            x_train.requires_grad = True

            # test if the improvement is monotone
            self.assertTrue(old_loss - current_loss > 0)

            old_loss = current_loss
