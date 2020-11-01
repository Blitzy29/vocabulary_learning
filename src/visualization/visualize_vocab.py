import plotly.graph_objects as go


def plot_comparison_both_probabilities(
            vocab,
            model_name="model",
            show_plot=True,
            save_plot=True,
            save_folder="data"):

    x = vocab["german_proba"]
    y = vocab["english_proba"]
    text = vocab["german"] + " - " + vocab["english"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram2dContour(
            x=x, y=y, colorscale="Blues", reversescale=True, xaxis="x", yaxis="y"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            xaxis="x",
            yaxis="y",
            mode="markers",
            marker=dict(color="rgba(0,0,0,0.3)", size=3),
            text=text,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0.9, 0.9],
            y=[0, 1],
            xaxis="x",
            yaxis="y",
            showlegend=False,
            mode="lines",
            line=dict(color="white", width=0.2, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0.9, 0.9],
            xaxis="x",
            yaxis="y",
            showlegend=False,
            mode="lines",
            line=dict(color="white", width=0.2, dash="dash"),
        )
    )

    fig.add_trace(
        go.Histogram(
            y=y, xaxis="x2", marker=dict(color="rgba(0,0,0,1)"), ybins=dict(size=0.05)
        )
    )
    fig.add_trace(
        go.Histogram(
            x=x, yaxis="y2", marker=dict(color="rgba(0,0,0,1)"), xbins=dict(size=0.05)
        )
    )

    fig.update_layout(
        title="Distributions of the probabilities",
        xaxis_title="German probability",
        yaxis_title="English probability",
        autosize=False,
        xaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
        yaxis=dict(zeroline=False, domain=[0, 0.85], showgrid=False),
        xaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
        yaxis2=dict(zeroline=False, domain=[0.85, 1], showgrid=False),
        height=700,
        width=700,
        bargap=0,
        hovermode="closest",
        showlegend=False,
    )

    if show_plot:
        fig.show()

    if save_plot:
        fig.write_html(f"{save_folder}/{model_name}_comparison_both_probabilities.html")
