import pandas as pd
import torch
import numpy as np
import openml


def get_openml_classification(did, max_samples, multiclass=True, shuffled=True):
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    if not multiclass:
        X = X[y < 2]
        y = y[y < 2]

    if multiclass and not shuffled:
        raise NotImplementedError("This combination of multiclass and shuffling isn't implemented")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        print('Not a NP Array, skipping')
        return None, None, None, None

    if not shuffled:
        sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
        pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
        X, y = X[sort][-pos * 2:], y[sort][-pos * 2:]
        y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
        X = torch.tensor(X).reshape(2, -1, X.shape[1]).transpose(0, 1).reshape(-1, X.shape[1]).flip([0]).float()
    else:
        order = np.arange(y.shape[0])
        np.random.seed(13)
        np.random.shuffle(order)
        X, y = torch.tensor(X[order]), torch.tensor(y[order])
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]

    return X, y, list(np.where(categorical_indicator)[0]), attribute_names

def load_openml_list(dids, filter_for_nan=False
                     , num_feats=100
                     , min_samples = 100
                     , max_samples=400
                     , multiclass=True
                     , max_num_classes=10
                     , shuffled=True
                     , return_capped = False):
    datasets = []
    openml_list = openml.datasets.list_datasets(dids)
    print(f'Number of datasets: {len(openml_list)}')

    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    if filter_for_nan:
        datalist = datalist[datalist['NumberOfInstancesWithMissingValues'] == 0]
        print(f'Number of datasets after Nan and feature number filtering: {len(datalist)}')

    for ds in datalist.index:
        modifications = {'samples_capped': False, 'classes_capped': False, 'feats_capped': False}
        entry = datalist.loc[ds]

        print('Loading', entry['name'], entry.did, '..')

        if entry['NumberOfClasses'] == 0.0:
            raise Exception("Regression not supported")
            #X, y, categorical_feats, attribute_names = get_openml_regression(int(entry.did), max_samples)
        else:
            X, y, categorical_feats, attribute_names = get_openml_classification(int(entry.did), max_samples
                                                                , multiclass=multiclass, shuffled=shuffled)
        if X is None:
            continue

        if X.shape[1] > num_feats:
            if return_capped:
                X = X[:, 0:num_feats]
                categorical_feats = [c for c in categorical_feats if c < num_feats]
                modifications['feats_capped'] = True
            else:
                print('Too many features')
                continue
        if X.shape[0] == max_samples:
            modifications['samples_capped'] = True

        if X.shape[0] < min_samples:
            print(f'Too few samples left')
            continue

        if len(np.unique(y)) > max_num_classes:
            if return_capped:
                X = X[y < np.unique(y)[10]]
                y = y[y < np.unique(y)[10]]
                modifications['classes_capped'] = True
            else:
                print(f'Too many classes')
                continue

        datasets += [[entry['name'], X, y, categorical_feats, attribute_names, modifications]]

    return datasets, datalist


# Classification
valid_dids_classification = [13, 59, 4, 15, 40710, 43, 1498]
test_dids_classification = [973, 1596, 40981, 1468, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668, 41146, 41169, 41027, 23517, 41165, 41161, 41159, 41138, 1590, 41166, 1464, 41168, 41150, 1489, 41142, 3, 12, 31, 54, 1067]
valid_large_classification = [  943, 23512,    49,   838,  1131,   767,  1142,   748,  1112,
        1541,   384,   912,  1503,   796,    20,    30,   903,  4541,
         961,   805,  1000,  4135,  1442,   816,  1130,   906,  1511,
         184,   181,   137,  1452,  1481,   949,   449,    50,   913,
        1071,   831,   843,     9,   896,  1532,   311,    39,   451,
         463,   382,   778,   474,   737,  1162,  1538,   820,   188,
         452,  1156,    37,   957,   911,  1508,  1054,   745,  1220,
         763,   900,    25,   387,    38,   757,  1507,   396,  4153,
         806,   779,   746,  1037,   871,   717,  1480,  1010,  1016,
         981,  1547,  1002,  1126,  1459,   846,   837,  1042,   273,
        1524,   375,  1018,  1531,  1458,  6332,  1546,  1129,   679,
         389]

open_cc_dids = [11,
 14,
 15,
 16,
 18,
 22,
 23,
 29,
 31,
 37,
 50,
 54,
 188,
 458,
 469,
 1049,
 1050,
 1063,
 1068,
 1510,
 1494,
 1480,
 1462,
 1464,
 6332,
 23381,
 40966,
 40982,
 40994,
 40975]
# Filtered by N_samples < 2000, N feats < 100, N classes < 10

open_cc_valid_dids = [13,25,35,40,41,43,48,49,51,53,55,56,59,61,187,285,329,333,334,335,336,337,338,377,446,450,451,452,460,463,464,466,470,475,481,679,694,717,721,724,733,738,745,747,748,750,753,756,757,764,765,767,774,778,786,788,795,796,798,801,802,810,811,814,820,825,826,827,831,839,840,841,844,852,853,854,860,880,886,895,900,906,907,908,909,915,925,930,931,934,939,940,941,949,966,968,984,987,996,1048,1054,1071,1073,1100,1115,1412,1442,1443,1444,1446,1447,1448,1451,1453,1488,1490,1495,1498,1499,1506,1508,1511,1512,1520,1523,4153,23499,40496,40646,40663,40669,40680,40682,40686,40690,40693,40705,40706,40710,40711,40981,41430,41538,41919,41976,42172,42261,42544,42585,42638]
