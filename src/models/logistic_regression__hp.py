import dill
from itertools import product
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from src.models.logistic_regression import ModelLogisticRegression

pd.set_option("display.max_columns", None)


# Dataset
path_dataset_train = "data/raw/20210119/dataset_train.pkl"

overall_hyperparameters_df = pd.DataFrame()

list_sampling = ["None"]
list_dimension_reduction = ["None"]
list_feature_selection = [True]

for i_sampling, i_dimension_reduction, i_feature_selection in product(
    list_sampling, list_dimension_reduction, list_feature_selection
):
    print(
        "Sampling: {} - Dimension Reduction: {} - Feature Selection: {}".format(
            i_sampling, i_dimension_reduction, i_feature_selection
        )
    )

    if (i_dimension_reduction == "PCA") & (i_feature_selection):
        print('Passing this step')
        continue

    # i_sampling = None  # "SMOTE"
    # i_dimension_reduction = None
    # i_feature_selection = False

    with open(path_dataset_train, "rb") as input_file:
        dataset_train = dill.load(input_file)

    model = ModelLogisticRegression()
    model.version = "hyperparameter_tuning__logistic_regression__20210131"
    model.sampling = i_sampling

    ##### 1st part preprocessing - fix

    dataset_train = model.preprocessing_training_numerical(dataset_train)
    dataset_train = model.preprocessing_training_diff_time(dataset_train)
    dataset_train = model.preprocessing_training_boolean(dataset_train)
    dataset_train = model.preprocessing_training_categorical(dataset_train)

    model.vardict["preprocessed"] = (
        model.vardict["numerical"]
        + model.vardict["diff_time"]
        + model.vardict["dummy_boolean"]
        + model.vardict["dummy_categorical"]
    )

    if i_sampling == "SMOTE":
        print('Apply SMOTE')
        dataset_train = model.apply_sampling(dataset_train)

    X_train = dataset_train[model.vardict["preprocessed"]]
    y_train = dataset_train[[model.vardict["target"]]]

    ##### additions
    scaler = StandardScaler()
    dimension_reduction = PCA()
    feature_selection = RFE(estimator=LogisticRegression(max_iter=10000))

    ##### model
    model_logistic_regression = LogisticRegression(max_iter=10000)

    ##### pipeline
    pipeline_steps = []
    pipeline_steps.append(("std_slc", scaler))
    if i_dimension_reduction == "PCA":
        pipeline_steps.append(("pca", dimension_reduction))
    if i_feature_selection:
        pipeline_steps.append(("feat_select", feature_selection))
    pipeline_steps.append(("logistic_Reg", model_logistic_regression))

    pipe = Pipeline(
        steps=pipeline_steps,
    )

    ##### grid
    param_grid = {
        "logistic_Reg__penalty": ["l1"],
        "logistic_Reg__C": np.logspace(0, 4, 16),
        "logistic_Reg__solver": ["liblinear"],
        "logistic_Reg__fit_intercept": [True],
    }

    if i_dimension_reduction == "PCA":
        param_grid["pca__n_components"] = list(range(1, X_train.shape[1] + 1, 5))
    if i_feature_selection:
        param_grid["feat_select__n_features_to_select"] = list(range(8, 17, 1))

    print(param_grid)

    ##### Create grid search object
    clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=7)

    ##### Fit on data
    best_clf = clf.fit(X_train, y_train.values.ravel())

    ##### specify results
    hyperparameters_df = pd.DataFrame.from_dict(clf.cv_results_)

    hyperparameters_df["scaler"] = True

    hyperparameters_df["sampling"] = i_sampling
    hyperparameters_df["dimension_reduction"] = i_dimension_reduction
    hyperparameters_df["feature_selection"] = i_feature_selection

    if not (i_dimension_reduction == "PCA"):
        hyperparameters_df["param_pca__n_components"] = None
    if not (i_feature_selection):
        hyperparameters_df["param_feat_select__n_features_to_select"] = None

    ###### save results
    overall_hyperparameters_df = overall_hyperparameters_df.append(hyperparameters_df)
    overall_hyperparameters_df.to_csv(
        "data/interim/overall_hyperparameters_31_step3.csv", index=False
    )

overall_hyperparameters_df.sort_values("mean_test_score", ascending=False, inplace=True)
# by default, the scoring function for Logistic Regression is accuracy

overall_hyperparameters_df.to_csv(
    "data/interim/overall_hyperparameters_31_step3.csv", index=False
)

print('----------------------------')
print('----------- DONE -----------')
print('----------------------------')
