#!/usr/bin/python3.10
# coding=utf-8

# Author: Lukas Vecerka (xvecer30)

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster as cluster
import matplotlib as mpl
from shapely.geometry import MultiPoint


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Function that creates GeoDataFrame from DataFrame with accidents.
    :param df: DataFrame with accidents
    :return: GeoDataFrame with accidents
    """
    df = df.dropna(subset=["d", "e"])
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.d, df.e), crs="EPSG:5514"
    )
    return gdf


def plot_geo(
    gdf: geopandas.GeoDataFrame,
    fig_location: str = None,
    show_figure: bool = False
):
    """
    Function that plots accidents caused by animals in JHM region in 2021
    and 2022.
    :param gdf: GeoDataFrame with accidents
    :param fig_location: Location where to save figure
    :param show_figure: If True, figure is shown
    """
    gdf_2 = gdf.copy()
    gdf_2 = gdf_2[(gdf_2["region"] == "JHM") & (gdf_2["p10"] == 4)]
    gdf_2["date"] = pd.to_datetime(gdf_2["p2a"])

    gdf_2021 = gdf_2[gdf_2["date"].dt.year == 2021]
    gdf_2022 = gdf_2[gdf_2["date"].dt.year == 2022]

    gdf_2021 = gdf_2021.to_crs(epsg=3857)
    gdf_2022 = gdf_2022.to_crs(epsg=3857)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)

    # Plotting for 2021
    gdf_2021.plot(ax=axes[0], color="blue", markersize=5)
    ctx.add_basemap(
        axes[0],
        crs=gdf_2021.crs.to_string(),
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.9,
    )
    axes[0].set_title("JHM kraj (2021)")
    axes[0].set_axis_off()

    # Plotting for 2022
    gdf_2022.plot(ax=axes[1], color="red", markersize=5)
    ctx.add_basemap(
        axes[1],
        crs=gdf_2022.crs.to_string(),
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.9,
    )
    axes[1].set_title("JHM kraj (2022)")
    axes[1].set_axis_off()

    plt.tight_layout()

    if fig_location is not None:
        fig.savefig(fig_location)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def plot_cluster(
    gdf: geopandas.GeoDataFrame,
    fig_location: str = None,
    show_figure: bool = False
):
    """
    Function which plots clusters of accidents where alcohol
    was involved in JHM.
    :param gdf: GeoDataFrame with accidents
    :param fig_location: Location where to save figure
    :param show_figure: If True, figure is shown
    """
    gdf_3 = gdf.copy()
    gdf_3 = gdf_3[(gdf_3["region"] == "JHM") & (gdf_3["p11"] >= 4)]

    gdf_jhm = gdf[(gdf["region"] == "JHM") & (gdf["date"].dt.year == 2021)]

    gdf_jhm = gdf_jhm.to_crs(epsg=3857)
    gdf_3 = gdf_3.to_crs(epsg=3857)

    gdf_3 = geopandas.clip(gdf_3, gdf_jhm.total_bounds)

    coords = gdf_3.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

    kmeans = cluster.KMeans(n_clusters=12, n_init=10)
    gdf_3["cluster"] = kmeans.fit_predict(coords)

    accident_counts = gdf_3["cluster"].value_counts()
    max_accidents = accident_counts.max()

    norm = mpl.colors.Normalize(vmin=0, vmax=max_accidents)

    colormap = mpl.colormaps["viridis"]

    gdf_3["color"] = gdf_3["cluster"].apply(
        lambda x: colormap(norm(accident_counts[x]))
    )

    fig, ax = plt.subplots(figsize=(15, 12))

    for cluster_id in range(12):
        cluster_gdf = gdf_3[gdf_3["cluster"] == cluster_id]
        if not cluster_gdf.geometry.empty:
            multipoint = MultiPoint(cluster_gdf.geometry.tolist())
            polygon = multipoint.convex_hull
            gdf_poly = geopandas.GeoDataFrame(
                [polygon], columns=["geometry"], crs=gdf_3.crs
            )
            gdf_poly.plot(ax=ax, color="gray", alpha=0.4)
            ax.scatter(
                cluster_gdf.geometry.x,
                cluster_gdf.geometry.y,
                color=mpl.colors.to_hex(cluster_gdf["color"].iloc[0]),
                label=f"Cluster {cluster_id}",
                s=5,
            )

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    plt.colorbar(
        sm,
        ax=ax,
        label="Počet nehod v úseku",
        orientation="horizontal",
        fraction=0.062,
        pad=0.05,
    )

    ctx.add_basemap(
        ax,
        crs=gdf_3.crs.to_string(),
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.9,
    )

    ax.set_axis_off()

    if fig_location is not None:
        fig.savefig(fig_location)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo3.png", True)
