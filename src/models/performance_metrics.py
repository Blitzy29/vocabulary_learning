import dill
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn import metrics


def show_results(
    predictions,
    model_name,
    show_plot=True,
    save_plot=True,
    save_folder="data/processed",
):

    binary_classification_results = (
        get_binary_classification_results(
            predictions, model_name, save_folder
        )
    )

    regression_results = get_regression_results(
        predictions, model_name, save_folder
    )

    plot_roc_auc_curve(
        predictions, model_name, show_plot, save_plot, save_folder
    )

    plot_precision_recall_curve(
        predictions,
        binary_classification_results,
        model_name,
        show_plot,
        save_plot,
        save_folder,
    )

    plot_predictions(
        predictions, model_name, show_plot, save_plot, save_folder
    )


def get_binary_classification_results(dataset, model_name="model", save_folder="data"):

    binary_classification_results = dict()

    total_population = len(
        dataset[(dataset["y_true"].isin([0, 1])) & (dataset["y_pred"].isin([0, 1]))]
    )
    binary_classification_results["total_population"] = total_population

    total_positive = len(
        dataset[dataset["y_true"] == 1]
    )
    binary_classification_results["total_positive"] = total_positive

    total_negative = len(
        dataset[dataset["y_true"] == 0]
    )
    binary_classification_results["total_negative"] = total_negative

    random_precision = total_positive / total_population
    binary_classification_results["random_precision"] = random_precision

    true_positive = len(dataset[(dataset["y_true"] == 1) & (dataset["y_pred"] == 1)])
    binary_classification_results["true_positive"] = true_positive

    false_negative = len(dataset[(dataset["y_true"] == 1) & (dataset["y_pred"] == 0)])
    binary_classification_results["false_negative"] = false_negative

    false_positive = len(dataset[(dataset["y_true"] == 0) & (dataset["y_pred"] == 1)])
    binary_classification_results["false_positive"] = false_positive

    true_negative = len(dataset[(dataset["y_true"] == 0) & (dataset["y_pred"] == 0)])
    binary_classification_results["true_negative"] = true_negative

    recall = true_positive / (true_positive + false_negative)
    binary_classification_results["recall"] = recall
    miss_rate = false_negative / (true_positive + false_negative)
    binary_classification_results["miss_rate"] = miss_rate

    fall_out = false_positive / (false_positive + true_negative)
    binary_classification_results["fall_out"] = fall_out
    specificity = true_negative / (false_positive + true_negative)
    binary_classification_results["specificity"] = specificity

    precision = true_positive / (true_positive + false_positive)
    binary_classification_results["precision"] = precision
    false_discovery_rate = false_positive / (true_positive + false_positive)
    binary_classification_results["false_discovery_rate"] = false_discovery_rate

    false_omission_rate = false_negative / (false_negative + true_negative)
    binary_classification_results["false_omission_rate"] = false_omission_rate
    negative_predictive_value = true_negative / (false_negative + true_negative)
    binary_classification_results["negative_predictive_value"] = negative_predictive_value

    accuracy = (true_positive + true_negative) / total_population
    binary_classification_results["accuracy"] = accuracy

    prevalence = (true_positive + false_negative) / total_population
    binary_classification_results["prevalence"] = prevalence

    positive_likelihood_ratio = recall / fall_out
    binary_classification_results["positive_likelihood_ratio"] = positive_likelihood_ratio
    negative_likelihood_ratio = miss_rate / specificity
    binary_classification_results["negative_likelihood_ratio"] = negative_likelihood_ratio

    diagnostic_odds_ratio = positive_likelihood_ratio / negative_likelihood_ratio
    binary_classification_results["diagnostic_odds_ratio"] = diagnostic_odds_ratio

    f1_score = 2 * precision * recall / (precision + recall)
    binary_classification_results["f1_score"] = f1_score

    logit_roc_auc = metrics.roc_auc_score(dataset["y_true"], dataset["y_pred"])
    binary_classification_results["logit_roc_auc"] = logit_roc_auc

    # Transform to table (to be saved)
    binary_classification_results_table = pd.DataFrame.from_dict(
        binary_classification_results, orient="index", columns=["value"]
    )

    binary_classification_results_table.to_csv(
        f"{save_folder}/{model_name}_binary_classification_results_table.csv"
    )

    with open(
            f"{save_folder}/{model_name}_binary_classification_results_dict.pkl", "wb"
    ) as file:
        dill.dump(binary_classification_results, file)

    return binary_classification_results


def get_regression_results(dataset, model_name="model", save_folder="data"):

    regression_results = dict()

    explained_variance_score = metrics.explained_variance_score(
        y_true=dataset["y_true"], y_pred=dataset["y_proba"]
    )
    regression_results["explained_variance_score"] = explained_variance_score

    max_error = metrics.max_error(y_true=dataset["y_true"], y_pred=dataset["y_proba"])
    regression_results["max_error"] = max_error

    mean_absolute_error = metrics.mean_absolute_error(
        y_true=dataset["y_true"], y_pred=dataset["y_proba"]
    )
    regression_results["mean_absolute_error"] = mean_absolute_error

    root_mean_squared_error = metrics.mean_squared_error(
        y_true=dataset["y_true"], y_pred=dataset["y_proba"], squared=False
    )
    regression_results["root_mean_squared_error"] = root_mean_squared_error

    r2_score = metrics.r2_score(y_true=dataset["y_true"], y_pred=dataset["y_proba"])
    regression_results["r2_score"] = r2_score

    normalised_log_loss = metrics.log_loss(dataset["y_true"], dataset["y_proba"])
    regression_results["normalised_log_loss"] = normalised_log_loss

    p = sum(dataset["y_true"]) / len(dataset)
    average_log_loss = - (p * np.log(p) + (1-p) * np.log(1-p))
    normalised_cross_entropy = normalised_log_loss / average_log_loss
    regression_results["normalised_cross_entropy"] = normalised_cross_entropy

    brier_score = metrics.brier_score_loss(dataset["y_true"], dataset["y_proba"])
    regression_results["brier_score"] = brier_score

    # Transform to table (to be saved)
    regression_results_table = pd.DataFrame.from_dict(
        regression_results, orient="index", columns=["value"]
    )

    regression_results_table.to_csv(
        f"{save_folder}/{model_name}_regression_results_table.csv"
    )

    with open(
            f"{save_folder}/{model_name}_regression_results_dict.pkl", "wb"
    ) as file:
        dill.dump(regression_results, file)

    return regression_results


def add_precision_recall_curve(fig, dataset, model_name="model"):

    precision, recall, thresholds = metrics.precision_recall_curve(
        dataset["y_true"], dataset["y_proba"]
    )

    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=model_name))

    return fig


def plot_precision_recall_curve(dataset, binary_classification_results, model_name="model",
                                show_plot=True, save_plot=True, save_folder="data"):

    # Create traces
    fig = go.Figure()

    fig = add_precision_recall_curve(fig, dataset, model_name="model")

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[
                binary_classification_results["random_precision"],
                binary_classification_results["random_precision"],
            ],
            mode="lines",
            name="Random precision",
            line=dict(color="black", dash="dash"),
        )
    )

    fig = add_square(fig, x0=0, x1=1, y0=0, y1=1)

    fig.update_layout(
        title="Precision-Recall curve",
        legend={"itemsizing": "constant"},
    )

    fig.update_xaxes(title_text="Recall", range=[-0.05, 1.05])
    fig.update_yaxes(title_text="Precision", range=[-0.05, 1.05])

    if show_plot:
        fig.show()

    if save_plot:
        fig.write_html(f"{save_folder}/{model_name}_PrecisionRecall.html")


def add_roc_auc_curve(fig, dataset, model_name="model"):

    fpr, tpr, thresholds = metrics.roc_curve(dataset["y_true"], dataset["y_proba"])

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model_name))

    return fig


def plot_roc_auc_curve(dataset, model_name="model", show_plot=True, save_plot=True, save_folder="data"):

    # Create traces
    fig = go.Figure()

    fig = add_roc_auc_curve(fig, dataset, model_name="model")

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="random",
            line=dict(color="black", dash="dash"),
        )
    )

    fig = add_square(fig, x0=0, x1=1, y0=0, y1=1)

    fig.update_layout(
        title="Receiver operating characteristic (ROC) curve",
        legend={"itemsizing": "constant"},
    )

    fig.update_xaxes(title_text="False Positive Rate", range=[-0.05, 1.05])
    fig.update_yaxes(title_text="True Positive Rate", range=[-0.05, 1.05])

    if show_plot:
        fig.show()

    if save_plot:
        fig.write_html(f"{save_folder}/{model_name}_ROC.html")


def add_square(fig, x0, x1, y0, y1):

    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y0],
            mode="lines",
            showlegend=False,
            line=dict(color="black", dash="dash", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y1, y1],
            mode="lines",
            showlegend=False,
            line=dict(color="black", dash="dash", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x0, x0],
            y=[y0, y1],
            mode="lines",
            showlegend=False,
            line=dict(color="black", dash="dash", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x1, x1],
            y=[y0, y1],
            mode="lines",
            showlegend=False,
            line=dict(color="black", dash="dash", width=1),
        )
    )

    return fig


def plot_predictions(dataset, model_name="model", show_plot=True, save_plot=True, save_folder="data"):

    dataset_subset = dataset[["y_true", "y_pred", "y_proba"]].copy()
    dataset_subset.sort_values("y_proba", inplace=True)
    dataset_subset["correct"] = np.where(
        dataset_subset["y_true"] == dataset_subset["y_pred"], "green", "red"
    )

    nb_false = len(dataset_subset) - sum(dataset_subset["y_true"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(dataset_subset))),
            y=dataset_subset['y_true'],
            marker=dict(color=dataset_subset["correct"]),
            mode="markers",
            name="True values",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(dataset_subset))),
            y=dataset_subset['y_proba'],
            marker=dict(color='grey'),
            mode="lines+markers",
            name="Predictions",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[nb_false - 0.05 * len(dataset_subset), nb_false + 0.05 * len(dataset_subset)],
            y=[0, 1],
            mode="lines",
            name="limit neg/pos",
            line=dict(color="black", dash="dash", width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, len(dataset_subset)],
            y=[0.5, 0.5],
            mode="lines",
            showlegend=False,
            line=dict(color="black", dash="dash", width=1),
        )
    )

    fig.update_layout(
        title="Predictions and true values",
        xaxis_title="Datapoints",
        yaxis_title="True values and predictions",
    )

    if show_plot:
        fig.show()

    if save_plot:
        fig.write_html(f"{save_folder}/{model_name}_predictions.html")
