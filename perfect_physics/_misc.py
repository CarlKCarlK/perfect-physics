from pathlib import Path
from typing import Any, List

import seaborn as sns
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sympy import Integer, srepr

from ._circle import Circle


# use srepr to save to a file
def save(expr, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        f.write(srepr(expr))


def load(filename):
    from sympy import (  # noqa: F401
        Add,
        FiniteSet,
        Integer,
        Mul,
        Pow,
        Rational,
        Symbol,
        Tuple,
    )

    with open(filename, "r") as f:
        result2 = eval(f.read())
        return result2


def plot(
    circle_list: List[Circle] = [],
    wall_list: List[Any] = [],
    clock=None,
    show=True,
    xlim=(-1, 21),
    ylim=(-5, 5),
    figsize=(19.2 / 4, 10.8 / 4),
    font_scale=4,
    colors=["tab:blue"],
    draw_wall_points=False,
    draw_radius=1.0,
    label_fun=None,
    show_fun=None,
):
    sns.set()
    figure = Figure(figsize=figsize)
    axes = figure.add_subplot()
    axes.set_aspect(1)
    axes.set_xlim(float(xlim[0]), float(xlim[1]))
    axes.set_ylim(float(ylim[0]), float(ylim[1]))

    if clock is not None:
        if isinstance(clock, str):
            title = f"clock={clock}"
        elif isinstance(clock, float):
            title = f"clock={clock:.2f}"
        elif isinstance(clock, Integer) or isinstance(clock, int):
            title = f"clock={clock}"
        else:
            title = f"clock={float(clock):.2f}={clock}"[:60]
        axes.set_title(title, loc="left")

    for index, circle in enumerate(circle_list):
        color = colors[index % len(colors)]
        axes.add_patch(
            patches.Circle((circle.x, circle.y), circle.r * draw_radius, color=color)
        )
        axes.annotate(
            "" if label_fun is None else label_fun(circle),
            xy=(circle.x + circle.vx, circle.y + circle.vy),
            xytext=(circle.x, circle.y),
            arrowprops=dict(arrowstyle="->", color="darkblue"),
        )

    for wall in wall_list:
        axes.axline(
            (float(wall.x0), float(wall.y0)),
            (float(wall.x1), float(wall.y1)),
            color="b",
        )
        if draw_wall_points:
            axes.scatter(wall.x0, wall.y0, color="tab:blue")
            axes.scatter(wall.x1, wall.y1, color="tab:blue")

    figure.set_dpi(100 * font_scale)
    figure.tight_layout()

    if show:
        new_manager = plt.figure().canvas.manager
        new_manager.canvas.figure = figure
        figure.set_canvas(new_manager.canvas)
        figure.set_dpi(100 * font_scale)
        figure.tight_layout()
        if show_fun is None:
            figure.show()
        else:
            show_fun(figure)
    return figure
