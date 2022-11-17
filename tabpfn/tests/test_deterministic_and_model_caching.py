import unittest

from tabpfn.datasets import load_openml_list, open_cc_dids
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier


class TestLoadModuleOnlyInference(unittest.TestCase):
    def test_main(self):
        test_datasets, cc_test_datasets_multiclass_df = load_openml_list(open_cc_dids[:1], multiclass=True,
                                                                         shuffled=True,
                                                                         filter_for_nan=False,
                                                                         max_samples=10000,
                                                                         num_feats=100,
                                                                         return_capped=True)


        for dataset in test_datasets: # only test first dataset, can be removed to test all
            xs, ys = dataset[1].clone(), dataset[2].clone()
            eval_position = xs.shape[0] // 2
            train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
            test_xs, test_ys = xs[eval_position:], ys[eval_position:]

            classifier = TabPFNClassifier(device='cpu')
            classifier.fit(train_xs, train_ys)
            pred1 = classifier.predict_proba(test_xs)
            self.assertEqual(classifier.models_in_memory, TabPFNClassifier.models_in_memory)
            self.assertEqual(len(classifier.models_in_memory), 1)

            classifier = TabPFNClassifier(device='cpu')
            classifier.fit(train_xs, train_ys)
            pred2 = classifier.predict_proba(test_xs)

            number_of_predictions, number_of_classes = pred1.shape

            for number in range(number_of_predictions):
                for class_nr in range(number_of_classes):
                    # checks that every class probability has difference of at most
                    self.assertTrue(pred1[number][class_nr] ==
                                    pred2[number][class_nr])
