def normalize_data(eval_xs):
    mean = eval_xs.mean(0)
    std = eval_xs.std(0) + .000001
    eval_xs = (eval_xs - mean) / std

    return eval_xs


