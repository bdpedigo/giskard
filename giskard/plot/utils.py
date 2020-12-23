def soft_axis_off(ax, top=False, bottom=False, left=False, right=False):
    ax.set(xlabel="", ylabel="", xticks=[], yticks=[])
    ax.spines["top"].set_visible(top)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_visible(left)
    ax.spines["right"].set_visible(right)
