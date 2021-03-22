# based on: Perez, J., Mendez, F. J., Menendez, M., & Losada, I. J. (2014).
# ESTELA: a method for evaluating the source and travel time of the wave energy reaching a local area.
# Ocean Dynamics, 64(8), 1181–1191. https://doi.org/10.1007/s10236-014-0740-7
#
# Examples (from olas.estela import calc, plot):
# calc with constant spread: estelas = calc("/data_local/tmp/glob2018??01T00.nc", 46, -131, mask="MAPSTA")
# calc with spread: estelas = calc("/data_local/tmp/ww3.glob_24m.2010??.nc", 46, -131, "hs.", "tp.", "th.", "si.", "MAPSTA")
# plot: plot(estelas, outdir=".")

import argparse
import datetime
import os
import re
from glob import glob

import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from cartopy.io import shapereader as shpreader
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from scipy import special

d2r = np.pi / 180.0


def parser():
    parser = argparse.ArgumentParser(description="Calculate estelas")
    parser.add_argument("datafiles", type=str, help="Files with wave data")
    parser.add_argument("lat0", type=float, help="Latitude of the target point")
    parser.add_argument("lon0", type=float, help="Longitude of the target point")
    parser.add_argument("--hs", type=str, default="hs", help="Significant wave height fieldnames")
    parser.add_argument("--tp", type=str, default="tp", help="Peak period fieldnames")
    parser.add_argument("--dp", type=str, default="dp", help="Wave direction fieldnames")
    parser.add_argument("--si", default=20, help="Directional spread fieldnames")
    parser.add_argument("-m", "--mask", type=str, default=None, help="mask fieldname")
    parser.add_argument("-g", "--groupers", nargs="*", default=None, help="groupers for results")
    parser.add_argument("-p", "--proj", type=str, default=None, help="projection")
    parser.add_argument("-o", "--outdir", type=str, default=None, help="output directory")
    args = parser.parse_args()

    estelas = calc(args.datafiles, args.lat0, args.lon0, args.hs, args.tp, args.dp, args.si, args.mask, args.groupers)
    plot(estelas, groupers=args.groupers, proj=args.proj, outdir=args.outdir)
    plt.show()


def calc(datafiles, lat0, lon0, hs="phs.", tp="ptp.", dp="pdir.", si=20, mask=None, groupers=None):
    """Calculate ESTELA dataset for a target point.

    Args:
        datafiles (str/list): Regular expression or list of data files.
        lat0 (float): Latitude of target point.
        lon0 (float): Longitude of target point.
        hs (str/list): regex/list of hs field names in datafiles
        tp (str/list): regex/list of tp field names in datafiles
        dp (str/list): regex/list of dp field names in datafiles
        si (str/list/float): Value or regex/list of directional spread field names
        mask (str): Information of mask
        groupers (list, optional): values used to group the results.

    Returns:
        xr.dataset: ESTELA dataset with F and traveltime fields.
    """
    if isinstance(datafiles, str):
        flist = sorted(glob(datafiles))
    else:
        flist = sorted(datafiles)
    print(f"{datetime.datetime.utcnow():%Y%m%d %H:%M:%S} Processing {len(flist)} files")
    groupers = get_groupers(groupers)

    lon0 %= 360.0
    lat0_arr = xr.DataArray(dims="site", data=np.array(lat0).flatten())
    lon0_arr = xr.DataArray(dims="site", data=np.array(lon0).flatten())
    sites = xr.Dataset(dict(lat0=lat0_arr, lon0=lon0_arr))
    # TODO calculate several sites at the same time. Problematic memory usage but much faster (if data reading is slow)

    dsf = xr.open_mfdataset(flist[0])
    spec_info = dict(hs=hs, tp=tp, dp=dp, si=si)
    for k, value in spec_info.items():
        if isinstance(value, str):  # expand regular expressions
            spec_info[k] = sorted(v for v in dsf.variables if re.fullmatch(value, v))
    npart = len(spec_info["hs"])
    num_si = isinstance(spec_info["si"], (int, float))
    print(spec_info)

    # geographical constants and initialization
    dists, bearings = dist_and_bearing(lat0, dsf.latitude, lon0, dsf.longitude)
    dist_m = dists * 6371000 * d2r
    va = 1.4 * 10 ** -5
    rowroa = 1 / 0.0013
    # sigma = 2 * np.pi / ds.tp  # Lemax = (rowroa * 9.81 ** 2) / (4 * sigma ** 3 * (2 * va * sigma) ** 0.5)
    k_dissipation = (
        -dist_m / (rowroa * 9.81 ** 2) * 4 * (2 * va) ** 0.5 * (2 * np.pi) ** 3.5
    )  # coef_dissipation = np.exp(-dist_m / Lemax)
    th1_sin = np.sin(0.5 * bearings * d2r)
    th1_cos = np.cos(0.5 * bearings * d2r)

    if isinstance(mask, str):
        mask = dsf[mask]
    vland = geographic_mask(lat0, lon0, dists, bearings, mask=mask)

    # S and Stp calculations
    si_calculations = True
    grouped_results = dict()
    for f in flist:
        print(f"{datetime.datetime.utcnow():%Y%m%d %H:%M:%S} Processing {f}")

        dsf = xr.open_mfdataset(f).chunk("auto")
        file_results = xr.Dataset()
        for ipart in range(npart):
            hs = dsf[spec_info["hs"][ipart]]
            tp = dsf[spec_info["tp"][ipart]]
            dp = dsf[spec_info["dp"][ipart]]

            coef_dissipation = np.exp(k_dissipation * (tp ** -3.5))

            if si_calculations:
                if num_si:  # don't repeat calculations
                    si = spec_info["si"]
                    si_calculations = False
                else:
                    si = dsf[spec_info["si"][ipart]].clip(15., 45.)
                    # TODO find better solution to avoid invalid A2 values

                s = (2 / (si * np.pi / 180) ** 2) - 1
                A2 = special.gamma(s + 1) / (special.gamma(s + 0.5) * 2 * np.pi ** 0.5)
                # TODO find faster spread approach (use normal distribution or table?)
                coef_spread = A2 * np.pi / 180  # deg
                # TODO review coef_spread units and compare with wavespectra

            th2 = 0.5 * dp * d2r
            coef_direction = abs(np.cos(th2) * th1_cos + np.sin(th2) * th1_sin) ** (
                2.0 * s
            )

            Spart_th = hs ** 2 / 16 * coef_dissipation * coef_direction * coef_spread
            file_results["S_th"] = file_results.get("S_th", 0) + (Spart_th)
            file_results["Stp_th"] = file_results.get("Stp_th", 0) + (tp * Spart_th)

        with ProgressBar():
            file_results.load()

        for grouper in groupers:
            if grouper == "ALL":
                grouped_results["ALL"] = grouped_results.get(
                    "ALL", 0
                ) + file_results.sum("time").assign(ntime=len(dsf.time))
            else:
                for k, v in file_results.groupby(grouper):
                    kstr = f"m{k:02g}" if grouper == "time.month" else str(k)
                    grouped_results[kstr] = grouped_results.get(kstr, 0) + v.sum(
                        "time"
                    ).assign(ntime=len(v.time))

    # Saving estelas
    time = xr.Variable(data=sorted(grouped_results), dims="time")
    estelas_aux = xr.concat([grouped_results[k] for k in time.values], dim=time)
    # TODO Te instead of Tp.  tp_te_ratio = 1.1 ?
    Fdeg = (
        1.025 * 9.81 * estelas_aux["Stp_th"] / estelas_aux["ntime"] * 9.81 / 4 / np.pi
    )
    cg_mps = (estelas_aux["Stp_th"] / estelas_aux["S_th"]) * 9.81 / 4 / np.pi

    estelas_dict = {"F": 360 * Fdeg, "traveltime": dist_m / 3600 / 24 / cg_mps}
    estelas = xr.Dataset(estelas_dict).where(vland, np.nan).merge(sites)
    estelas.F.attrs["units"] = "360 * kW / m / degree"
    estelas.traveltime.attrs["units"] = "days"
    estelas.attrs["start_time"] = str(xr.open_mfdataset(flist[0]).time[0].values)
    estelas.attrs["end_time"] = str(xr.open_mfdataset(flist[-1]).time[-1].values)
    return estelas


def plot(estelas, groupers=None, proj=None, cmap="plasma", figsize=[25, 10], outdir=None):
    """Plot ESTELA maps for one or several time periods

    Args:
        estelas (xr.dataset): ESTELA dataset with F and traveltime fields.
        proj (cartopy.crs, optional): Map projection. Defaults to PlateCarree.
        cmap (str, optional): Colormap. Defaults to "plasma".
        figsize (list, optional): Figure size. Defaults to [25,10].

    Returns:
        figs: list of figure handles
    """
    lon0 = float(estelas.lon0)
    lat0 = float(estelas.lat0)
    if proj is None:
        # proj = ccrs.Orthographic(lon0, lat0)
        proj = ccrs.PlateCarree(central_longitude=lon0)

    c1day = dict(levels=np.linspace(1, 30, 30), colors="grey", linewidths=0.5)
    c3day = dict(levels=np.linspace(3, 30, 10), colors="black", linewidths=1.0)

    figs = []
    groupers = get_groupers(groupers)
    for grouper in groupers:
        if grouper == "time.season":
            time = ["DJF", "MAM", "JJA", "SON"]
        elif grouper == "time.month":
            time = [f"m{m:02g}" for m in range(1, 13)]
        else:
            time = [grouper]

        timesize = len(time)
        ds = estelas.sel(time=[t for t in time if t in estelas.time])
        Faux = [ds.isel(time=0).assign(time=t)["F"]*np.nan for t in time if t not in estelas.time]
        F = xr.concat([ds.F] + Faux, dim="time").sel(time=time)

        print(f"Plotting estelas for time={time} from {ds}\n")  # TODO refactor plotting
        if timesize == 1:
            fig = plt.figure(figsize=figsize)
            F.plot(
                subplot_kws={"projection": proj},
                transform=ccrs.PlateCarree(),
                cmap=cmap,
            )
        else:
            g = F.plot(
                subplot_kws={"projection": proj},
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                col="time",
                col_wrap=2 if timesize <= 4 else 3,
            )
            fig = g.fig
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])

        axes = fig.axes[:-1]
        for iax, ax in enumerate(axes):
            if time[iax] not in estelas.time:
                continue

            ttime = ds.traveltime.sel(time=time[iax])
            for ic, c_args in enumerate([c1day, c3day]):
                p = ttime.plot.contour(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    linestyles="solid",
                    add_labels=False,
                    zorder=2*(1+ic),
                    **c_args,
                )
            if timesize == 1:
                ax.clabel(p, c3day["levels"], colors="black", fmt="%.0fdays")

            ax.set_global()
            ax.coastlines()
            ax.stock_img()
            ax.plot(lon0, lat0, "or", transform=ccrs.PlateCarree())

        fig.suptitle(f"[{ds.start_time[:10]} - {ds.end_time[:10]}]")
        figs.append(fig)
        if outdir is not None:
            fig.savefig(os.path.join(outdir, f"estela_{grouper}.png"))
    return figs


def dist_and_bearing(lat1, lat2, lon1, lon2):
    """Calculate distances and bearings from one point to others

    Args:
        lat1 (float): Latitude origin point
        lat2 (float/array): Latitude end points
        lon1 (float): Longitude origin point
        lon2 (float/array): Longitude end points

    Returns:
        float/array: distances and bearings in degrees
    """
    lat1r = lat1 * d2r
    lat2r = lat2 * d2r
    latdifr = (lat2 - lat1) * d2r
    londifr = (lon2 - lon1) * d2r

    a = np.sin(latdifr / 2) * np.sin(latdifr / 2) + np.cos(lat1r) * np.cos(
        lat2r
    ) * np.sin(londifr / 2) * np.sin(londifr / 2)
    a = a.clip(0., 1.) # to avoid warning for a=1.0000001,
    degdist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) / d2r

    y = np.sin(londifr) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(londifr)
    brng = (np.arctan2(y, x) / d2r).transpose() % 360
    return (degdist, brng)


def geographic_mask(lat0, lon0, dists, bearings, mask=None):
    # TODO this approach for geographic_mask is not clean enough
    """Check the great circles points to find points not blocked by land

    Args:
        lat0 (float): Latitude origin point
        lon0 (float): Longitude origin point
        dists (array): Distances
        bearings (array): Bearings
        mask (array, optional): mask where ocean points are True

    Returns:
        array: points not blocked by land
    """

    def update_dmax(dmax, dists, bearings):
        ibearings = np.trunc(bearings % 360).astype(int)
        for d, b in zip(dists, ibearings):
            dmax[b] = min(dmax[b], d)
        return dmax

    dmax = 180 * np.ones(360)
    if mask is None:
        mask = xr.ones_like(dists)
    mask = mask.where(dists > 0, np.nan)

    valid = (mask != 1).values.flatten()
    d_arr = dists.values.flatten()[valid]
    b_arr = bearings.values.flatten()[valid]
    dmax = update_dmax(dmax, d_arr, b_arr)

    # shpfilename = shpreader.gshhs(scale='c', level=1)
    shpfilename = shpreader.natural_earth(
        resolution="110m", category="physical", name="coastline"
    )
    coastlines = shpreader.Reader(shpfilename).records()
    for c in coastlines:
        lonland, latland = map(np.array, zip(*c.geometry.coords))
        distland, bearingland = dist_and_bearing(lat0, latland, lon0, lonland)
        dmax = update_dmax(dmax, distland, bearingland)
    vland = dists < dmax[np.trunc(bearings % 360).astype(int)]
    return vland


def get_groupers(groupers):
    if groupers is None:
        groupers = ["ALL", "time.season", "time.month"]
    return groupers


if __name__ == "__main__":
    parser()
