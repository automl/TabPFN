import unittest

import time
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch import nn

from tabpfn import TabPFNClassifier


class TestLoadModuleOnlyInference(unittest.TestCase):
    def test_main(self):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        X_train = torch.from_numpy(X_train)
        X_train.requires_grad = True

        X_test = torch.from_numpy(X_test)
        X_test.requires_grad = True

        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        old_loss = 1.0

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

        classifier = TabPFNClassifier(device=device, N_ensemble_configurations=3, no_preprocess_mode=True, no_grad=False)
        start_time = time.time()
        for i in range(10):
            classifier.fit(X_train, y_train)

            logits = classifier.predict_proba(X_test, return_logits=True)

            loss = nn.CrossEntropyLoss()(logits, y_test.long().to(device))
            current_loss = float(loss)
            loss.backward()

            rate = 0.01

            X_test = X_test.detach() - rate * X_test.grad
            X_test.requires_grad = True

            X_train = X_train.detach() - rate * X_train.grad
            X_train.requires_grad = True

            # test if the improvement is monotone
            self.assertTrue(old_loss - current_loss > 0)

            old_loss = current_loss
        print(time.time() - start_time)
        # 10.217917203903198s
