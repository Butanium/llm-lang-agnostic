import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

try:
    from utils import mean_no_none, ci_no_none
except:
    from .utils import mean_no_none, ci_no_none


"""
=====================
   Mean repr defs   
=====================
"""


def get_layout_kwargs(title):
    return dict(
        autosize=True,
        title_font_size=24,
        margin=dict(t=120 if title else 80, b=50, l=60, r=30, pad=10),
        height=800,
        width=1200,
        plot_bgcolor="rgba(240,240,240,0.3)",
    )


subplot_kwargs = dict(
    vertical_spacing=0.1,
    horizontal_spacing=0.1,
)


def plot_defs_comparison(result_dict, from_what, title=None, path=None, exp_id=None):
    titles = {
        "max sim": "Max Similarity with GT Defs",
        "mean sim": "Mean Similarity with GT Defs",
        "sim w mean": "Similarity with Mean Embedding of GT Defs",
        "sim w mean fst": "Sim with Mean Emb of GT Defs (w/o first)",
        "mean sim with others": "Mean Similarity with Other Words",
        "max sim with others": "Max Similarity with Other Words",
    }
    metrics = list(titles.keys())
    values = {
        titles[m]: [result_dict[word][from_what][m] for word in result_dict]
        for m in metrics
    }
    fig = make_subplots(
        rows=2, cols=3, subplot_titles=[titles[m] for m in metrics], **subplot_kwargs
    )
    for i, (name, vals) in enumerate(values.items()):
        fig.add_trace(
            go.Histogram(
                x=vals,
                name=name,
                bingroup="a",
                showlegend=False,
                histnorm="probability",
            ),
            row=(i // 3) + 1,
            col=(i % 3) + 1,
        )
        mean_val = mean_no_none(vals)
        if mean_val is not None:
            fig.add_vline(
                x=mean_val,
                line_color="black",
                line_dash="dash",
                opacity=0.5,
                row=(i // 3) + 1,
                col=(i % 3) + 1,
                name="Mean",
                showlegend=i == 0,  # Only show legend for first mean line
            )

        fig.update_xaxes(
            title_text="Similarity Score",
            row=(i) // 3 + 1,
            col=(i) % 3 + 1,
            range=[0, 1],
        )
        fig.update_yaxes(
            title_text="Probability", row=(i) // 3 + 1, col=(i) % 3 + 1, range=[0, 1]
        )
    fig.update_layout(
        title_text=f"Definition Embedding Comparison Metrics\n{title or from_what}",
        showlegend=True,
        **get_layout_kwargs(True),
    )
    fig.show()
    if path is not None:
        fig.write_image(path / f"{exp_id}_{from_what}_defs_comparison.png", scale=3)
    return fig


def plot_compare_setup(result_dict, path=None, title=None, exp_id=None, show=True):
    titles = {
        "max sim": "Max Similarity with GT Defs",
        "mean sim": "Mean Similarity with GT Defs",
        "sim w mean": "Similarity with Mean Embedding of GT Defs",
        "sim w mean fst": "Similarity with Mean Embedding of GT Defs (w/o first)",
        "mean sim with others": "Mean Similarity with Other Words",
        "max sim with others": "Max Similarity with Other Words",
    }
    title = "" if title is None else "<br>" + title
    layout_kwargs = get_layout_kwargs(title)

    metrics = list(titles.keys())
    means = {}
    for source in ["from trans", "from def", "baseline", "rnd gt"]:
        means[source] = {
            metric: mean_no_none(
                [result_dict[word][source][metric] for word in result_dict]
            )
            for metric in metrics
        }

    fig = make_subplots(
        rows=2, cols=3, subplot_titles=[titles[m] for m in metrics], **subplot_kwargs
    )
    colors = {
        "from trans": "blue",
        "from def": "red",
        "baseline": "green",
        "rnd gt": "purple",
    }
    for i, metric in enumerate(metrics):
        row = (i // 3) + 1
        col = (i % 3) + 1

        y_values = [means[source][metric] for source in means.keys()]
        cis = []
        for source in means.keys():
            values = [result_dict[word][source][metric] for word in result_dict]
            ci = ci_no_none(values)
            cis.append(ci)
        fig.add_trace(
            go.Bar(
                x=list(means.keys()),
                y=y_values,
                name=titles[metric],
                marker_color=[colors[source] for source in means.keys()],
                error_y=dict(type="data", array=cis, visible=True, color="black"),
            ),
            row=row,
            col=col,
        )
        if col == 1:
            fig.update_yaxes(
                title_text="Mean Score with 95% CI",
                title_standoff=15,
                row=row,
                col=col,
            )
        fig.update_yaxes(
            range=[0, 1], row=row, col=col, gridcolor="rgba(0,0,0,0.1)", showgrid=True
        )
        if row == 2:
            fig.update_xaxes(title_text="Source", title_standoff=15, row=row, col=col)

    fig.update_layout(
        title_text="Mean Metrics Across Definition Sources (with 95% Confidence Intervals)"
        + (f"{title}" if title else ""),
        showlegend=False,
        **layout_kwargs,
    )
    if show:
        fig.show()
    if path is not None:
        fig.write_image(path / f"{exp_id}_defs_comparison.png", scale=3)
    fig2 = make_subplots(
        rows=2, cols=3, subplot_titles=[titles[m] for m in metrics], **subplot_kwargs
    )
    colors = {
        "from trans": "blue",
        "from def": "red",
        "baseline": "green",
        "rnd gt": "purple",
    }
    for i, metric in enumerate(metrics):
        row = (i // 3) + 1
        col = (i % 3) + 1
        for source in means.keys():
            values = [result_dict[word][source][metric] for word in result_dict]
            fig2.add_trace(
                go.Histogram(
                    x=values,
                    name=source,
                    showlegend=i == 0,
                    marker_color=colors[source],
                    opacity=0.75,
                    nbinsx=20,
                    bingroup="b",
                ),
                row=row,
                col=col,
            )
        if col == 1:
            fig2.update_yaxes(title_text="Count", row=row, col=col)
        if row == 2:
            fig2.update_xaxes(title_text="Score", row=row, col=col, range=[0, 1])
    fig2.update_layout(
        title_text="Distribution of Metrics Across Definition Sources" + title,
        showlegend=True,
        barmode="group",
        **layout_kwargs,
    )
    if show:
        fig2.show()
    if path is not None:
        fig2.write_image(path / f"{exp_id}_defs_histogram.png", scale=3)
    return fig, fig2
