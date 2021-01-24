import datetime
import plotly.graph_objects as go


def plot_result_hyperparameter(
    hyperparameters_df,
    hyperparameter_to_plot,
    variable_objective,
    use_log_scale=False,
    minimize_objective=True,
    folder_save=None,
):
    """
    Plot the result of the hyperparameters search.

    hyperparameters_df: DataFrame, results of the hyperparameters search with GridSearchCV, RandomizedSearchCV, optuna ...
    hyperparameter_to_plot: str, must be a column of hyperparameters_df
    variable_objective: str,  must be a column of hyperparameters_df
    use_log_scale: boolean
    minimize_objective: boolean, if true, the lower the objective the better
    """

    hyperparameters_df = hyperparameters_df.copy()
    hyperparameters_df["n_trial"] = range(len(hyperparameters_df))

    if minimize_objective:
        best_hyperparameters = hyperparameters_df.loc[
            hyperparameters_df[variable_objective].idxmin(axis=1)
        ]
    else:
        best_hyperparameters = hyperparameters_df.loc[
            hyperparameters_df[variable_objective].idxmax(axis=1)
        ]

    # Create traces
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hyperparameters_df[hyperparameter_to_plot],
            y=hyperparameters_df[variable_objective],
            mode="markers",
            marker=dict(
                color=hyperparameters_df["n_trial"],
                colorscale="Viridis",
                colorbar=dict(title="Number trial"),
                showscale=True,
            ),
            hovertemplate="<b>Trial %{marker.color:.d}</b><br><br>"
            + "Hp: %{x:.5f}<br>"
            + "Obj.: %{y:.5f}<br>"
            + "<extra></extra>",
            showlegend=False,
            name=variable_objective,
        )
    )

    if use_log_scale:
        fig.update_xaxes(type="log")

    fig.add_trace(
        go.Scatter(
            x=[
                best_hyperparameters[hyperparameter_to_plot],
                best_hyperparameters[hyperparameter_to_plot],
            ],
            y=[
                min(hyperparameters_df[variable_objective]),
                max(hyperparameters_df[variable_objective]),
            ],
            mode="lines",
            showlegend=False,
            line=dict(color="#e377c2", dash="dash"),
        )
    )

    fig.update_layout(
        title="Evolution of {} for hyperparameter {}".format(
            variable_objective, hyperparameter_to_plot
        ),
        xaxis_title=hyperparameter_to_plot,
        yaxis_title=variable_objective,
        legend={"itemsizing": "constant"},
    )

    # fig.show()

    fig.write_html("{}/{}.html".format(folder_save, hyperparameter_to_plot))


def plot_time_hyperparameter_boxplot(hyperparameters_df, folder_save):

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=hyperparameters_df["duration_hour"],
            name="duration of one trial",
            boxmean="sd",
        )
    )

    fig.update_layout(
        title="Duration of each trial",
        xaxis_title="",
        yaxis_title="duration (in hour)",
        legend={"itemsizing": "constant"},
    )

    fig.show()

    fig.write_html("{}/{}.html".format(folder_save, "duration_boxplot"))


def plot_time_hyperparameter_scatter(hyperparameters_df, folder_save):

    # Create traces
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hyperparameters_df["n_trial_all"],
            y=hyperparameters_df["duration_hour"],
            mode="markers",
            showlegend=False,
            name="duration_hour",
        )
    )

    fig.update_layout(
        title="Evolution of duration in hours",
        xaxis_title="trials",
        yaxis_title="duration (in hour)",
        legend={"itemsizing": "constant"},
    )

    fig.show()

    fig.write_html("{}/{}.html".format(folder_save, "duration_hour"))


def plot_time_hyperparameter(hyperparameters_df, folder_save):

    hyperparameters_df["datetime_start_dt"] = hyperparameters_df["datetime_start"].map(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    )
    hyperparameters_df["datetime_complete_dt"] = hyperparameters_df[
        "datetime_complete"
    ].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))

    hyperparameters_df["duration_dt"] = (
        hyperparameters_df["datetime_complete_dt"]
        - hyperparameters_df["datetime_start_dt"]
    )

    hyperparameters_df["duration_hour"] = hyperparameters_df["duration_dt"].map(
        lambda x: x.seconds / 3600
    )

    plot_time_hyperparameter_boxplot(hyperparameters_df, folder_save)

    plot_time_hyperparameter_scatter(hyperparameters_df, folder_save)
