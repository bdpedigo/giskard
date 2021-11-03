import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_1samp, uniform


def subuniformity_plot(x, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(x, ax=ax, stat="density", cumulative=True, **kwargs)
    stat, pvalue = ks_1samp(x, uniform(0, 1).cdf, alternative="greater")
    ax.plot([0, 1], [0, 1], linewidth=3, linestyle=":", color="black")
    ax.text(0, 1, f"p-value: {pvalue:.3f}")
    ax.set_ylabel("Cumulative density")
    return ax, stat, pvalue
