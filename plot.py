import matplotlib.pyplot as plt
import numpy as np
from src.utils import mean_no_none, load_dict, ci_no_none
to_en = "/dlabscratch1/cdumas/thinking-lang/results/Llama-2-7b-hf/mean_repr_def/de-it_nl-fi_zh-es_es-ru_ru-ko-en/1733920826_frisky-trout/defs_comparison.json"
to_en = load_dict(to_en)
to_zh = "/dlabscratch1/cdumas/thinking-lang/results/Llama-2-7b-hf/mean_repr_def/de-it_nl-fi_zh-es_es-ru_ru-ko-zh/1733920826_frisky-trout/defs_comparison.json"
to_zh = load_dict(to_zh)
to_fr = "/dlabscratch1/cdumas/thinking-lang/results/Llama-2-7b-hf/mean_repr_def/en-ko_en-ja_en-et_en-fi-fr/1733920826_frisky-trout/defs_comparison.json"
to_fr = load_dict(to_fr)

# Define consistent colors and method names
METHOD_COLORS = {
    'Multi-Source Translation': '#1f77b4',  # blue
    'Multi-Source Definition': 'red',   # red
    'Single-Source Translation': '#add8e6',  # light blue
    'Single-Source Definition': 'orangered',   # light red
    'Word Patching': '#9467bd',             # purple
    'Prompting': '#8c564b',                 # brown
    'Repeat Word': '#e377c2',               # pink
}

# Helper function to extract metrics for a specific dataset
def get_metrics(data_dict, metric_name):
    metrics = []
    cis = []
    for json_key in ['from trans', 'from def', 'from single trans', 'from single def', 
                     'word patch', 'prompting', 'repeat word']:
        values = []
        for concept in data_dict:
            if json_key in data_dict[concept] and metric_name in data_dict[concept][json_key]:
                values.append(data_dict[concept][json_key][metric_name])
        metrics.append(mean_no_none(values))
        cis.append(ci_no_none(values))
    return metrics, cis, sum(v is not None for v in values)
methods = ['Multi-Source Translation', 'Multi-Source Definition', 
        'Single-Source Translation', 'Single-Source Definition',
        'Word Patching', 'Prompting', 'Repeat Word']

# Data and titles
datasets = [to_en, to_zh, to_fr]
titles = ['English', 'Chinese', 'French']


def main_merged_no_xticks_with_legend(datasets=datasets, titles=titles, plot_name="similarities"):
    plt.rcParams.update({'font.size': 14 *2,
                        'axes.titlesize': 28,
                        'axes.labelsize': 14 *2,
                        'xtick.labelsize': 14 *2,
                        'ytick.labelsize': 14 *2})

    # Create figure and subplots with extra space for legend
    fig, axs = plt.subplots(1, 3, figsize=(22, 8))  # Increased figure width for legend

    # Store bars for method legend
    method_bars = []

    # Plot for each dataset
    for idx, (data, title) in enumerate(zip(datasets, titles)):
        # Get metrics
        sim_mean_fst, cis_fst, num_concepts = get_metrics(data, 'sim w mean fst')
        mean_sim_others, cis_others, num_concepts2 = get_metrics(data, 'mean sim with others')
        
        x = np.arange(len(methods))  # Create x coordinates for bars
        width = 0.35  # Width of the bars
        
        # Plot solid bars for sim_mean_fst
        bars1 = axs[idx].bar(x - width/2, sim_mean_fst, width, yerr=cis_fst, 
                            capsize=5, edgecolor='black', label='Similarity with mean\nground truth embedding')
        
        # Plot hashed bars for mean_sim_others
        bars2 = axs[idx].bar(x + width/2, mean_sim_others, width, yerr=cis_others, 
                            capsize=5, edgecolor='black', hatch='////', 
                            label='Similarity with\nother concepts')
        
        # Color the bars
        for bar, method in zip(bars1, methods):
            bar.set_color(METHOD_COLORS[method])
            if idx == 0:  # Only store bars from first subplot for legend
                method_bars.append(bar)

        for bar, method in zip(bars2, methods):
            bar.set_edgecolor(METHOD_COLORS[method])
            bar.set_facecolor('white')
        
        axs[idx].set_title(f'Target: {title} ({num_concepts} concepts)', pad=40)
        axs[idx].set_ylim(0, 1)  # Set y-axis range to 0, 1
        
        # Remove x ticks and labels
        axs[idx].set_xticks([])
        axs[idx].set_xticklabels([])

        # Remove top and right spines
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)

        # Add y-axis label only for the first subplot
        if idx == 0:
            axs[idx].set_ylabel('Mean similarity')
        else:
            axs[idx].set_yticklabels([])

    # Add metric type legend to the right of the last subplot
    handles1, labels1 = axs[-1].get_legend_handles_labels()
    legend_handles = [
        plt.Rectangle((0,0), 1, 1, facecolor='white', edgecolor='black'),  # Solid bar
        plt.Rectangle((0,0), 1, 1, facecolor='white', edgecolor='black', hatch='////')  # Hatched bar
    ]
    fig.legend(legend_handles, labels1[:2], bbox_to_anchor=(1.02, 0.7),
              loc='center left', borderaxespad=0.)

    # Add method legend to the right of the last subplot
    fig.legend(method_bars, methods, bbox_to_anchor=(1.02, 0.3), 
              loc='center left', borderaxespad=0.)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f"./plots/{plot_name}.pdf", bbox_inches='tight')
    plt.show()
