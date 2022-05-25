import tqdm
import numpy as np

def baseline_predict(metric_function, eval_xs, eval_ys, categorical_feats, metric_used=None, eval_pos=2, max_time=300, **kwargs):
    """
    Baseline prediction interface.
    :param metric_function:
    :param eval_xs:
    :param eval_ys:
    :param categorical_feats:
    :param metric_used:
    :param eval_pos:
    :param max_time: Scheduled maximum time
    :param kwargs:
    :return: list [np.array(metrics), np.array(outputs), best_configs] or [None, None, None] if failed
    """

    metrics = []
    outputs = []
    best_configs = []
    eval_splits = list(zip(eval_xs.transpose(0, 1), eval_ys.transpose(0, 1)))
    for eval_x, eval_y in tqdm.tqdm(eval_splits, desc='Calculating splits'+str(metric_function)+' '+str(eval_pos)):
        try:
            metric, output, best_config = metric_function(eval_x[:eval_pos],
                                                          eval_y[:eval_pos],
                                                          eval_x[eval_pos:],
                                                          eval_y[eval_pos:],
                                                          categorical_feats,
                                                          metric_used=metric_used
                                                          , max_time=max_time)
            metrics += [metric]
            outputs += [output]
            best_configs += [best_config]
            return np.array(metrics), np.array(outputs), best_configs
        except Exception as e:
            print(f'There was an exception in {metric_function}')
            print(e)
            return None, None, None