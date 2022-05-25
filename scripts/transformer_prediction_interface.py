import torch
import random

from torch.utils.checkpoint import checkpoint

from utils import normalize_data, to_ranking_low_mem, remove_outliers
from priors.utils import normalize_by_used_features_f
from utils import NOP
import numpy as np

from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler

def transformer_predict(model, eval_xs, eval_ys, eval_position,
                        device='cpu',
                        max_features=100,
                        style=None,
                        inference_mode=False,
                        num_classes=2,
                        extend_features=True,
                        normalize_to_ranking=False,
                        softmax_temperature=0.0,
                        multiclass_decoder='permutation',
                        preprocess_transform='mix',
                        categorical_feats=[],
                        feature_shift_decoder=True,
                        N_ensemble_configurations=10,
                        average_logits=True,
                        normalize_with_sqrt=False, **kwargs):
    """

    :param model:
    :param eval_xs:
    :param eval_ys:
    :param eval_position:
    :param rescale_features:
    :param device:
    :param max_features:
    :param style:
    :param inference_mode:
    :param num_classes:
    :param extend_features:
    :param normalize_to_ranking:
    :param softmax_temperature:
    :param multiclass_decoder:
    :param preprocess_transform:
    :param categorical_feats:
    :param feature_shift_decoder:
    :param N_ensemble_configurations:
    :param average_logits:
    :param normalize_with_sqrt:
    :param metric_used:
    :return:
    """
    num_classes = len(torch.unique(eval_ys))

    def predict(eval_xs, eval_ys, used_style, softmax_temperature, return_logits):
        # Initialize results array size S, B, Classes

        inference_mode_call = torch.inference_mode() if inference_mode else NOP()
        with inference_mode_call:
            output = model(
                    (used_style.repeat(eval_xs.shape[1], 1) if used_style is not None else None, eval_xs, eval_ys.float()),
                    single_eval_pos=eval_position)[:, :, 0:num_classes]

            output = output[:, :, 0:num_classes] / torch.exp(softmax_temperature)
            if not return_logits:
                output = torch.nn.functional.softmax(output, dim=-1)
            #else:
            #    output[:, :, 1] = model((style.repeat(eval_xs.shape[1], 1) if style is not None else None, eval_xs, eval_ys.float()),
            #               single_eval_pos=eval_position)

            #    output[:, :, 1] = torch.sigmoid(output[:, :, 1]).squeeze(-1)
            #    output[:, :, 0] = 1 - output[:, :, 1]

        #print('RESULTS', eval_ys.shape, torch.unique(eval_ys, return_counts=True), output.mean(axis=0))

        return output

    def preprocess_input(eval_xs, preprocess_transform):
        import warnings

        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim - TODO")
        if preprocess_transform != 'none':
            if preprocess_transform == 'power' or preprocess_transform == 'power_all':
                pt = PowerTransformer(standardize=True)
            elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
                pt = QuantileTransformer(output_distribution='normal')
            elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
                pt = RobustScaler(unit_variance=True)

        # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
        eval_xs = normalize_data(eval_xs)

        # Removing empty features
        eval_xs = eval_xs[:, 0, :].cpu().numpy()
        sel = [len(np.unique(eval_xs[0:eval_ys.shape[0], col])) > 1 for col in range(eval_xs.shape[1])]
        eval_xs = np.array(eval_xs[:, sel])

        warnings.simplefilter('error')
        if preprocess_transform != 'none':
            feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else set(
                range(eval_xs.shape[1])) - set(categorical_feats)
            for col in feats:
                try:
                    pt.fit(eval_xs[0:eval_ys.shape[0], col:col + 1])
                    trans = pt.transform(eval_xs[:, col:col + 1])
                    # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                    eval_xs[:, col:col + 1] = trans
                except:
                    pass
        warnings.simplefilter('default')

        eval_xs = torch.tensor(eval_xs).float().unsqueeze(1).to(device)

        # eval_xs = normalize_data(eval_xs)

        # TODO: Cautian there is information leakage when to_ranking is used, we should not use it
        eval_xs = remove_outliers(eval_xs) if not normalize_to_ranking else normalize_data(to_ranking_low_mem(eval_xs))

        # Rescale X
        eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                               normalize_with_sqrt=normalize_with_sqrt)
        return eval_xs.detach()

    eval_xs, eval_ys = eval_xs.to(device), eval_ys.to(device)
    eval_ys = eval_ys[:eval_position]

    model.to(device)
    style = style.to(device)

    model.eval()

    import itertools
    style = style.unsqueeze(0) if len(style.shape) == 1 else style
    num_styles = style.shape[0]
    styles_configurations = range(0, num_styles)
    preprocess_transform_configurations = [preprocess_transform if i % 2 == 0 else 'none' for i in range(0, num_styles)]
    if preprocess_transform == 'mix':
        def get_preprocess(i):
            if i == 0:
                return 'power_all'
            if i == 1:
                return 'robust_all'
            if i == 2:
                return 'none'
        preprocess_transform_configurations = [get_preprocess(i) for i in range(0, num_styles)]
    styles_configurations = zip(styles_configurations, preprocess_transform_configurations)

    feature_shift_configurations = range(0, eval_xs.shape[2]) if feature_shift_decoder else [0]
    class_shift_configurations = range(0, len(torch.unique(eval_ys))) if multiclass_decoder == 'permutation' else [0]

    ensemble_configurations = list(itertools.product(styles_configurations, feature_shift_configurations, class_shift_configurations))
    random.shuffle(ensemble_configurations)
    ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]

    output = None

    eval_xs_transformed = {}
    for ensemble_configuration in ensemble_configurations:
        (styles_configuration, preprocess_transform_configuration), feature_shift_configuration, class_shift_configuration = ensemble_configuration

        style_ = style[styles_configuration:styles_configuration+1, :]
        softmax_temperature_ = softmax_temperature[styles_configuration]

        eval_xs_, eval_ys_ = eval_xs.clone(), eval_ys.clone()

        if preprocess_transform_configuration in eval_xs_transformed:
            eval_xs_ = eval_xs_transformed['preprocess_transform_configuration'].clone()
        else:
            eval_xs_ = preprocess_input(eval_xs_, preprocess_transform=preprocess_transform_configuration)
            eval_xs_transformed['preprocess_transform_configuration'] = eval_xs_

        eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()

        eval_xs_ = torch.cat([eval_xs_[..., feature_shift_configuration:],eval_xs_[..., :feature_shift_configuration]],dim=-1)

        # Extend X
        if extend_features:
            eval_xs_ = torch.cat(
                [eval_xs_,
                 torch.zeros((eval_xs_.shape[0], eval_xs_.shape[1], max_features - eval_xs_.shape[2])).to(device)], -1)

        #preprocess_transform_ = preprocess_transform if styles_configuration % 2 == 0 else 'none'
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")
            output_ = checkpoint(predict, eval_xs_, eval_ys_, style_, softmax_temperature_, True)
            output_ = torch.cat([output_[..., class_shift_configuration:],output_[..., :class_shift_configuration]],dim=-1)

        #output_ = predict(eval_xs, eval_ys, style_, preprocess_transform_)
        if not average_logits:
            output_ = torch.nn.functional.softmax(output_, dim=-1)
        output = output_ if output is None else output + output_

    output = output / len(ensemble_configurations)
    if average_logits:
        output = torch.nn.functional.softmax(output, dim=-1)

    output = torch.transpose(output, 0, 1)

    return output

def get_params_from_config(c):
    return {'max_features': c['num_features']
        , 'rescale_features': c["normalize_by_used_features"]
        , 'normalize_to_ranking': c["normalize_to_ranking"]
        , 'normalize_with_sqrt': c.get("normalize_with_sqrt", False)
            }