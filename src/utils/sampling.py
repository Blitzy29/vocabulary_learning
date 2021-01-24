import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_sampling(sampling, model, dataset):

    if sampling == 'SMOTE':
        dataset = apply_smote_sampling(model, dataset)

    return dataset


def apply_smote_sampling(model, dataset):

    os = SMOTE(
        random_state=0
    )

    n_dataset = len(dataset)
    prop_dataset = np.mean(dataset[model.vardict["target"]])
    X_train = dataset[model.vardict["preprocessed"]]
    y_train = dataset[[model.vardict["target"]]]

    X_train_os, y_train_os = os.fit_sample(X_train, y_train)
    dataset = pd.concat([X_train_os, y_train_os], axis=1)

    # we can check the numbers of our data
    print("SMOTE - datapoints - {:5d} -> {:5d}".format(
        n_dataset, len(dataset)
    ))
    print("SMOTE - proportion - {:0.3f} -> {:0.3f}".format(
        prop_dataset, np.mean(dataset[model.vardict["target"]])
    ))

    return dataset
