import unittest

from scripts import tabular_metrics
from datasets import load_openml_list, open_cc_dids
from scripts.transformer_prediction_interface import TabPFNClassifier


class TestLoadModuleOnlyInference(unittest.TestCase):
    def test_main(self):
        test_datasets, cc_test_datasets_multiclass_df = load_openml_list(open_cc_dids, multiclass=True,
                                                                         shuffled=True,
                                                                         filter_for_nan=False,
                                                                         max_samples=10000,
                                                                         num_feats=100,
                                                                         return_capped=True)

        classifier_with_only_inference = TabPFNClassifier(device='cpu', only_inference=True,
                                                          N_ensemble_configurations=4)
        classifier_normal = TabPFNClassifier(device='cpu', only_inference=False, N_ensemble_configurations=4)

        overall_difference_in_auc_mean = 0
        overall_difference_in_cross_entropy_mean = 0

        for dataset in test_datasets:
            xs, ys = dataset[1].clone(), dataset[2].clone()
            eval_position = xs.shape[0] // 2
            train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
            test_xs, test_ys = xs[eval_position:], ys[eval_position:]

            classifier_with_only_inference.fit(train_xs, train_ys)
            classifier_normal.fit(train_xs, train_ys)

            prediction_with_only_inference = classifier_with_only_inference.predict_proba(test_xs)
            prediction_normal = classifier_normal.predict_proba(test_xs)

            self.assertTrue(prediction_normal.shape == prediction_with_only_inference.shape)
            number_of_predictions, number_of_classes = prediction_normal.shape

            allowed_difference_of_probabilities = 1  # until now comparisons are not possible
            for number in range(number_of_predictions):
                for class_nr in range(number_of_classes):
                    # checks that every class probability has difference of at most
                    self.assertTrue(abs(prediction_with_only_inference[number][class_nr]
                                        - prediction_normal[number][class_nr])
                                    < allowed_difference_of_probabilities)

            allowed_difference_in_auc_score = 0.02
            difference_in_auc_score = (tabular_metrics.auc_metric(test_ys, prediction_with_only_inference)
                                       - tabular_metrics.auc_metric(test_ys, prediction_normal))
            overall_difference_in_auc_mean += difference_in_auc_score
            self.assertTrue(abs(difference_in_auc_score) < allowed_difference_in_auc_score)

            allowed_difference_in_cross_entropy_score = 0.1
            difference_in_cross_entropy_score = (tabular_metrics.cross_entropy(test_ys, prediction_with_only_inference)
                                                 - tabular_metrics.cross_entropy(test_ys, prediction_normal))
            overall_difference_in_cross_entropy_mean += difference_in_cross_entropy_score
            self.assertTrue(abs(difference_in_cross_entropy_score) < allowed_difference_in_cross_entropy_score)

        print("overall_difference_in_auc_mean_over_all_datasets:", overall_difference_in_auc_mean)
        print("overall_difference_in_cross_entropy_mean_over_all_datasets :", overall_difference_in_cross_entropy_mean)

        allowed_average_auc_difference_over_all_datasets = 0.001
        allowed_average_cross_entropy_difference_over_all_datasets = 0.002

        self.assertTrue(len(test_datasets) >= 1)
        overall_difference_in_auc_mean /= len(test_datasets)
        overall_difference_in_cross_entropy_mean /= len(test_datasets)
        self.assertTrue(abs(overall_difference_in_auc_mean) < allowed_average_auc_difference_over_all_datasets)
        self.assertTrue(abs(overall_difference_in_cross_entropy_mean)
                        < allowed_average_cross_entropy_difference_over_all_datasets)
