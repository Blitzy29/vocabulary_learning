import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def show_components(dataset, component_a=1, component_b=2, model_name='model'):

    fig = go.Figure()

    dataset_0 = dataset[
        dataset['result'] == 0
    ]

    dataset_1 = dataset[
        dataset['result'] == 1
    ]

    fig.add_trace(go.Scatter(
        x=dataset_0["component_{:02d}".format(component_a)],
        y=dataset_0["component_{:02d}".format(component_b)],
        mode="markers",
        name='result == 0',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=dataset_1["component_{:02d}".format(component_a)],
        y=dataset_1["component_{:02d}".format(component_b)],
        mode="markers",
        name='result == 1',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title="Showing components {:02d} & {:02d}".format(component_a, component_b),
        xaxis_title='component {:02d}'.format(component_a),
        yaxis_title='component {:02d}'.format(component_b),
        legend={'itemsizing': 'constant'}
    )

    fig.show()

    fig.write_html("data/figures/{}_pca_{:02d}_{:02d}.html".format(model_name, component_a, component_b))


def explained_variance(X_pca):
    explained_variance_pca = np.var(X_pca, axis=0)
    explained_variance_ratio = explained_variance_pca / np.sum(explained_variance_pca)
    return explained_variance_ratio


def plot_explained_variance(explained_variance_ratio, model_name='model'):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(explained_variance_ratio))),
            y=explained_variance_ratio,
            mode="markers+lines",
        )
    )

    fig.update_layout(
        title="Explained variance per component",
        xaxis_title="Components",
        yaxis_title="Explained variance (%)",
        legend={"itemsizing": "constant"},
    )

    fig.show()

    fig.write_html("data/figures/{}_pca_explained_variance.html".format(model_name))


def plot_feature_components(feature_components, feature_names, n_components, model_name='model'):
    plt.matshow(feature_components, cmap='viridis')
    plt.yticks(
        range(n_components),
        ['{}. Comp'.format(x+1) for x in range(0, n_components)],
        fontsize=10)
    plt.colorbar()
    plt.xticks(
        range(len(feature_names)),
        feature_names,
        rotation=65, ha='left')
    plt.tight_layout()
    plt.savefig("data/figures/{}_pca_feature_components.png".format(model_name), format='png', dpi=500)
