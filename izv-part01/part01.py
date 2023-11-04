#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: xvecer30 (Lukas Vecerka)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.
Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene
na prednasce
"""

from bs4 import BeautifulSoup
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def integrate(
    f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000
) -> float:
    """
    Function for calculating integral of given function f over interval <a, b>.
    Args:
        f (Callable[[NDArray], NDArray]): function to integrate
        a (float): start of interval
        b (float): end of interval
        steps (int): number of steps, default 1000
    Returns:
        float: value of definite integral of function f over interval <a, b>
    """
    x = np.linspace(a, b, steps)
    y = f((x[:-1] + x[1:]) / 2)
    return np.sum((x[1:] - x[:-1]) * y)


def generate_graph(
    a: List[float], show_figure: bool = False, save_path: str | None = None
):
    """
    Function for plotting mathematical function f(x) = a^2 * x^3 * sin(x)
    Args:
        a (List[float]): list of a values
        show_figure (bool): if True, shows figure
        save_path (str): if set, saves figure to path
    """
    a = np.array(a).reshape(3, 1)  # convert to NDArray
    x = np.linspace(-3, 3, 200)
    y = a**2 * x**3 * np.sin(x)

    fig = plt.figure(figsize=(10, 5), frameon=True)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-3, 5)
    ax.xaxis.set_ticks(np.arange(-3, 4))
    ax.set_ylim(0, 40)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f_a(x)$")

    # plot blue area
    (line1,) = plot_graph(ax, y[0], x, "blue", "$Y_{1.0}(x)$", 1.0)

    # plot orange area
    (line2,) = plot_graph(ax, y[1], x, "orange", "$Y_{1.5}(x)$", 1.5)

    # plot green area
    (line3,) = plot_graph(ax, y[2], x, "green", "$Y_{2.0}(x)$", 2.0)

    ax.legend(
        handles=[line1, line2, line3],
        bbox_to_anchor=(0.5, 1.15),
        loc="upper center",
        ncol=3,
    )
    if show_figure:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_graph(
    ax: Axes, arr: NDArray, time: NDArray, color: str, label: str, val: float
) -> [Line2D]:
    """
    Helper function for generating graphs
    Args:
        ax (Axes): Axes object from plot
        arr (NDArray): array with values
        time (NDArray): array with space
        color (string): color of plot
        label (string): label for plot
    Returns:
        [Line2D]: list of lines
    """
    lines = ax.plot(time, arr, label=label)
    ax.fill_between(x=time, y1=arr, color=color, alpha=0.1)
    ax.text(
        3,
        arr[-1],
        "$\\int f_{{{value}}}(x)dx = {int}$".format(
            value=val, int=round(np.trapz(arr, time), 2)
        ),
        fontsize=12,
        ha="left",
        va="center",
        color="black",
    )
    return lines


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Generates 3 sinus plots of 3 functions, f1 = 0.5 * cos(1/50*pi*t),
    f2 = 0.25 * (sin(pi*t) + sin(3/2*pi*t) and f3 = f1 + f2
    Args:
        show_figure (bool): if True, shows figure
        save_path (str): if set, saves figure to path
    """
    time = np.linspace(0, 100, 20000)
    f1 = 0.5 * np.cos(1 / 50 * np.pi * time)
    f2 = 0.25 * (np.sin(np.pi * time) + np.sin(3 / 2 * np.pi * time))
    f3 = f1 + f2

    _, axes = plt.subplots(
        ncols=1,
        nrows=3,
        constrained_layout=True,
        figsize=(8, 10)
        )
    (ax1, ax2, ax3) = axes

    # plot first sinus
    plot_sinus(ax1, "$f_1(t)$", f1, time)

    # plot second sinus
    plot_sinus(ax2, "$f_2(t)$", f2, time)

    # plot green part of third sinus
    green_line = np.where(f3 >= f1, f3, np.nan)
    plot_sinus(ax3, "$f_1(t) + f_2(t)$", green_line, time, color="green")

    # plot red part of third sinus
    red_line = np.where(f3 <= f1, f3, np.nan)
    ax3.plot(time, red_line, label="$f_3(t)$", color="red")

    if show_figure:
        plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_sinus(
    ax: Axes, label: str, arr: NDArray, time: NDArray, color: str | None = None
):
    """
    Helper function for generating sinus plots

    Args:
        ax (Axes): Axes object from plot
        label (string): label for plot
        arr (NDArray): array with values
        time (NDArray): array with space
        color (string | None): color of plot, default None
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_xlabel("$t$")
    ax.set_ylabel(label)
    ax.plot(time, arr, label=label, color=color)


def download_data() -> List[Dict[str, Any]]:
    """
    Downloads data from https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html
    and get data from the table
    Returns:
        List[Dict[str, Any]]: list of dictionaries with parsed data
    """
    html = requests.get("https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html")
    html.encoding = "utf-8"  # set encoding to utf-8
    html_text = html.text
    soup = BeautifulSoup(html_text, "html.parser")

    table = soup.find_all("table")[1]
    rows = table.find_all("tr")[1:]  # skip header

    data = []
    for row in rows:
        parsed_row = parse_table_row(row)
        data.append(parsed_row)

    return data


def parse_table_row(row: BeautifulSoup) -> Dict[str, Any]:
    """
    Parses row from table and returns dictionary with data
    Args:
        row (BeautifulSoup): row from table
    Returns:
        Dict[str, Any]: dictionary with data
    """
    row_data = row.find_all("td")
    row_data = [cell.text for cell in row_data]
    data_dict = {
        "position": row_data[0],
        "lat": float(row_data[2][:-1].replace(",", ".")),
        "long": float(row_data[4][:-1].replace(",", ".")),
        "height": float(row_data[6].replace("\xa0", "").replace(",", ".")),
    }
    return data_dict
