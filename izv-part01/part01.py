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
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    x = np.linspace(a, b, steps)
    y = f((x[:-1] + x[1:]) / 2)
    return np.sum((x[1:] - x[:-1]) * y)


def generate_graph(
    a: List[float], show_figure: bool = False, save_path: str | None = None
):
    a = np.array(a).reshape(3, 1)
    x = np.linspace(-3, 3, 200)
    y = a**2 * x**3 * np.sin(x)

    fig = plt.figure(figsize=(10, 5), frameon=True)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-3, 5)
    ax.xaxis.set_ticks(np.arange(-3, 4))
    ax.set_ylim(0, 40)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f_a(x)$")

    (line1,) = ax.plot(x, y[0], label="$Y_{1.0}(x)$")
    ax.fill_between(x=x, y1=y[0], color="b", alpha=0.1)
    ax.text(
        3,
        y[0][-1],
        "$\\int f_{{2.0}}(x)dx = {}$".format(round(np.trapz(y[0], x), 2)),
        fontsize=12,
        ha="left",
        va="center",
        color="black",
    )

    (line2,) = ax.plot(x, y[1], label="$Y_{1.5}(x)$")
    ax.fill_between(x=x, y1=y[1], color="orange", alpha=0.1)
    ax.text(
        3,
        y[1][-1],
        "$\\int f_{{2.0}}(x)dx = {}$".format(round(np.trapz(y[1], x), 2)),
        fontsize=12,
        ha="left",
        va="center",
        color="black",
    )

    (line3,) = ax.plot(x, y[2], label="$Y_{2.0}(x)$")
    ax.fill_between(x=x, y1=y[2], color="green", alpha=0.1)
    ax.text(
        3,
        y[2][-1],
        "$\\int f_{{2.0}}(x)dx = {}$".format(round(np.trapz(y[2], x), 2)),
        fontsize=12,
        ha="left",
        va="center",
        color="black",
    )

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


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    time = np.linspace(0, 100, 20000)
    f1 = 0.5 * np.cos(1 / 50 * np.pi * time)
    f2 = 0.25 * (np.sin(np.pi * time) + np.sin(3 / 2 * np.pi * time))
    f3 = f1 + f2

    _, axes = plt.subplots(ncols=1, nrows=3, constrained_layout=True, figsize=(8, 10))
    (ax1, ax2, ax3) = axes

    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.8, 0.8)
    ax1.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$f_1(t)$")
    ax1.plot(time, f1, label="$f_1(t)$")

    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax1.set_xlabel("$t$")
    ax2.set_ylabel("$f_2(t)$")
    ax2.plot(time, f2, label="$f_2(t)$")

    green_line = np.where(f3 >= f1, f3, np.nan)
    red_line = np.where(f3 <= f1, f3, np.nan)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-0.8, 0.8)
    ax3.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax3.set_xlabel("$t$")
    ax3.set_ylabel("$f_1(t) + f_2(t)$")
    ax3.plot(time, green_line, label="$f_3(t)$", color="green")
    ax3.plot(time, red_line, label="$f_3(t)$", color="red")

    if show_figure:
        plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300)


def download_data() -> List[Dict[str, Any]]:
    html = requests.get("https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html")
    html.encoding = "utf-8"
    html_text = html.text
    soup = BeautifulSoup(html_text, "html.parser")

    table = soup.find_all("table")[1]
    rows = table.find_all("tr")[1:]

    data = []
    for row in rows:
        row = row.find_all("td")
        row = [cell.text for cell in row]
        data_dict = {
            "position": row[0],
            "lat": float(row[2][:-1].replace(",", ".")),
            "long": float(row[4][:-1].replace(",", ".")),
            "height": float(row[6].replace("\xa0", "").replace(",", ".")),
        }
        data.append(data_dict)

    return data
