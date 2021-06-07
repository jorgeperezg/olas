"""Based on https://doi.org/10.1007/s10236-014-0740-7

Perez, J., Mendez, F. J., Menendez, M., & Losada, I. J. (2014).
ESTELA: a method for evaluating the source and travel time of the wave energy reaching a local area.
Ocean Dynamics, 64(8), 1181–1191.
"""
# Examples:
# from olas.estela import calc, plot
# calc with constant spread: estelas = calc("/data_local/tmp/glob2018??01T00.nc", 46, -131)
# calc with spread: estelas = calc("/data_local/tmp/ww3.glob_24m.2010??.nc", 46, -131, si="si.")
# plot energy maps: plot(estelas, outdir=".")
# plot gain/loss maps: plot(estelas, gainloss=True, outdir=".")

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

D2R = np.pi / 180.0
"""degrees to radians factor"""

def parser():
    """ Command-line entry point """
    parser = argparse.ArgumentParser(description="Calculate estelas")
    parser.add_argument("datafiles", type=str, help="Files with wave data")
    parser.add_argument("lat0", type=float, help="Latitude of the target point")
    parser.add_argument("lon0", type=float, help="Longitude of the target point")
    parser.add_argument("--hs", type=str, default="hs", help="Significant wave height fieldnames")
    parser.add_argument("--tp", type=str, default="tp", help="Peak period fieldnames")
    parser.add_argument("--dp", type=str, default="dp", help="Wave direction fieldnames")
    parser.add_argument("--si", default=20, help="Directional spread fieldnames")
    parser.add_argument("-g", "--groupers", nargs="*", default=None, help="groupers for results")
    parser.add_argument("-n", "--nblocks", type=int, default=1, help="Number of blocks for file calculations")
    parser.add_argument("-p", "--proj", type=str, default=None, help="projection")
    parser.add_argument("-o", "--outdir", type=str, default=None, help="output directory")
    args = parser.parse_args()

    estelas = calc(args.datafiles, args.lat0, args.lon0, args.hs, args.tp, args.dp, args.si, args.groupers, args.nblocks)
    plot(estelas, groupers=args.groupers, proj=args.proj, outdir=args.outdir)
    plt.show()

def calc(datafiles, lat0, lon0, hs="phs.|hs.", tp="ptp.|tp.", dp="pdir.|th.", si=20, groupers=None, nblocks=1):
    """Calculate ESTELA dataset for a target point.

    Args:
        datafiles (str or sequence of str): Regular expression or list of data files.
        lat0 (float): Latitude of target point.
        lon0 (float): Longitude of target point.
        hs (str or sequence of str): regex/list of hs field names in datafiles.
        tp (str or sequence of str): regex/list of tp field names in datafiles.
        dp (str or sequence of str): regex/list of dp field names in datafiles.
        si (str or sequence of str or float): Value or regex/list of directional spread field names.
        groupers (sequence of str, optional): Values used to group the results.
        nblocks (int, optional): Number of blocks. More blocks need less memory but calculations are slower.

    Returns:
        xarray.Dataset: ESTELA dataset with F and traveltime fields.
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
    # geographical constants and initialization
    dists, bearings = dist_and_bearing(lat0, dsf.latitude, lon0, dsf.longitude)
    vland = geographic_mask(lat0, lon0, dists, bearings)
    dist_m = dists * 6371000 * D2R
    va = 1.4 * 10 ** -5
    rowroa = 1 / 0.0013
    # sigma = 2 * np.pi / ds.tp  # Lemax = (rowroa * 9.81 ** 2) / (4 * sigma ** 3 * (2 * va * sigma) ** 0.5)
    k_dissipation = (
        -dist_m / (rowroa * 9.81 ** 2) * 4 * (2 * va) ** 0.5 * (2 * np.pi) ** 3.5
    )  # coef_dissipation = np.exp(-dist_m / Lemax)
    th1_sin = np.sin(0.5 * bearings * D2R)
    th1_cos = np.cos(0.5 * bearings * D2R)

    # S and Stp calculations
    si_calculations = True
    grouped_results = dict()
    for f in flist:
        dsf = xr.open_mfdataset(f).chunk("auto")

        if nblocks > 1:
            dsf_blocks = [g for _, g in dsf.groupby_bins("time", nblocks)]
        else:
            dsf_blocks = [dsf]
        print(f"{datetime.datetime.utcnow():%Y%m%d %H:%M:%S} Processing {f} (nblocks={nblocks})")

        spec_info = dict(hs=hs, tp=tp, dp=dp, si=si)
        for k, value in spec_info.items():
            if isinstance(value, str):  # expand regular expressions
                spec_info[k] = sorted(v for v in dsf.variables if re.fullmatch(value, v))
        npart = len(spec_info["hs"])
        num_si = isinstance(spec_info["si"], (int, float))
        print(spec_info)

        for dsi in dsf_blocks:
            block_results = xr.Dataset()
            for ipart in range(npart):
                hs_data = dsi[spec_info["hs"][ipart]]
                tp_data = dsi[spec_info["tp"][ipart]]
                dp_data = dsi[spec_info["dp"][ipart]]

                coef_dissipation = np.exp(k_dissipation * (tp_data ** -3.5))

                if si_calculations:
                    if num_si:  # don't repeat calculations
                        si_data = spec_info["si"]
                        si_calculations = False
                    else:
                        si_data = dsi[spec_info["si"][ipart]].clip(15., 45.)
                        # TODO find better solution to avoid invalid A2 values

                    s = (2 / (si_data * np.pi / 180) ** 2) - 1
                    A2 = special.gamma(s + 1) / (special.gamma(s + 0.5) * 2 * np.pi ** 0.5)
                    # TODO find faster spread approach (use normal distribution or table?)
                    coef_spread = A2 * np.pi / 180  # deg
                    # TODO review coef_spread units and compare with wavespectra

                th2 = 0.5 * dp_data * D2R
                coef_direction = abs(np.cos(th2) * th1_cos + np.sin(th2) * th1_sin) ** (
                    2.0 * s
                )

                Spart_th = hs_data ** 2 / 16 * coef_dissipation * coef_direction * coef_spread
                block_results["S_th"] = block_results.get("S_th", 0) + (Spart_th)
                block_results["Stp_th"] = block_results.get("Stp_th", 0) + (tp_data * Spart_th)

            with ProgressBar():
                block_results.load()

            for grouper in groupers:
                if grouper == "ALL":
                    grouped_results["ALL"] = grouped_results.get(
                        "ALL", 0
                    ) + block_results.sum("time").assign(ntime=len(dsi.time))
                else:
                    for k, v in block_results.groupby(grouper):
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
    estelas_dict = {"F": 360 * Fdeg, "traveltime": (3600 * 24 * cg_mps / dist_m)**-1}  # dimensions order tyx
    estelas = xr.Dataset(estelas_dict).where(vland, np.nan).merge(sites)
    estelas.F.attrs["units"] = "$\\frac{kW}{m\\circ}$"
    estelas.traveltime.attrs["units"] = "days"
    estelas.attrs["start_time"] = str(xr.open_mfdataset(flist[0]).time[0].values)
    estelas.attrs["end_time"] = str(xr.open_mfdataset(flist[-1]).time[-1].values)
    return estelas


def plot(estelas, groupers=None, gainloss=False, proj=None, set_global=False, cmap=None, figsize=[25, 10], outdir=None):
    """Plot ESTELA maps for one or several time periods

    Args:
        estelas (xarray.Dataset): ESTELA dataset with F and traveltime fields.
        groupers (sequence of str, optional): Values used to group the results.
        gainloss (bool, optional): Flag to plot maps of gain/loss of energy.
        proj (cartopy.crs, optional): Map projection. Defaults to PlateCarree.
        cmap (str, optional): Colormap.
        figsize (tuple, optional): A tuple (width, height) of the figure in inches.
        outdir (str): Path to save figures.

    Returns:
        figs: list of figure handles
    """
    lat0 = float(estelas.lat0)
    lon0 = float(estelas.lon0)
    gc = great_circles(lat0, lon0, ngc=16)
    c1day = dict(levels=np.linspace(1, 30, 30), colors="grey", linewidths=0.5)
    c3day = dict(levels=np.linspace(3, 30, 10), colors="black", linewidths=1.0)
    # TODO: type of plots where traveltimes are included should not be hardcoded

    if proj is None:
        # proj = ccrs.Orthographic(lon0, lat0)
        proj = ccrs.PlateCarree(central_longitude=lon0)

    if cmap is None:
        cmap = "seismic" if gainloss else "inferno"

    figs = []
    groupers = get_groupers(groupers)
    for grouper in groupers:
        if grouper == "time.season":
            time = ["DJF", "MAM", "JJA", "SON"]
        elif grouper == "time.month":
            time = [f"m{m:02g}" for m in range(1, 13)]
        else:
            time = [grouper]

        ds = estelas.sel(time=[t for t in time if t in estelas.time])
        aux = [ds.isel(time=0).assign(time=t)["F"] * np.nan for t in time if t not in estelas.time]
        F = xr.concat([ds["F"]] + aux, dim="time").sel(time=time)
        F = F.dropna("longitude", how="all").dropna("latitude", how="all")

        if gainloss:
            # TODO: change gainloss argument to fieldname with F as default and calculate Fgl in calc
            ngc = 360
            polar_grid = great_circles(lat0, lon0, ngc)
            polarF = F.interp(polar_grid)
            dist_midpoints = (polarF.distance.values[1:] + polarF.distance.values[:-1]) / 2
            cosd = np.cos(polarF.distance * D2R)
            S = 4 * np.pi * 6371**2 / ngc * abs(cosd.diff("distance")) / 2  # km**2
            incF = (polarF.diff("distance") / S).assign_coords(distance=dist_midpoints)
            F *= np.nan  # empty pcolors, using contourf
            F.attrs["standard_name"] = "${\\Delta}F$"
            F.attrs["units"] = "$\\frac{kW}{m\\circ{km^2}}$"  # colorbar defined for pcolors
        else:
            F = 360 * F
            F.attrs["units"] = "$360\\times\\frac{kW}{m\\circ}$"

        print(f"Plotting estelas for time={time} from {ds}\n")
        # TODO refactor plotting and choose sensible colorbar limits
        if len(time) == 1:
            fig = plt.figure(figsize=figsize)
            plt.axes(projection=proj)
            F.plot(
                transform=ccrs.PlateCarree(),
                cmap=cmap,
            )
        else:
            g = F.plot(
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                col="time",
                col_wrap=2 if len(time) <= 4 else 3,
                subplot_kws={"projection": proj},
            )
            fig = g.fig
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])

        for iax, ax in enumerate(fig.axes[:-1]):
            extent = ax.get_extent()
            if time[iax] not in ds.time:
                continue

            if gainloss:
                Fi = incF.sel(time=time[iax])
                clim = float(abs(Fi).quantile(0.95))
                p = ax.contourf(
                    incF.longitude,
                    incF.latitude,
                    Fi.clip(-clim, clim),
                    transform=ccrs.PlateCarree(),
                    levels=15,
                    cmap=cmap,
                )

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
            if len(time) == 1:
                ax.clabel(p, c3day["levels"], colors="black", fmt="%.0fdays")
                ax.plot(gc.longitude, gc.latitude, ".r", markersize=1, transform=ccrs.PlateCarree())

            if set_global or (extent[1]-extent[0]) % 360 == 0:
                ax.set_global()
            else:
                ax.set_extent(extent, crs=proj)
            ax.coastlines()
            ax.stock_img()
            ax.plot(lon0, lat0, "ok", transform=ccrs.PlateCarree())

        fig.suptitle(f"[{ds.start_time[:10]} - {ds.end_time[:10]}]")
        figs.append(fig)
        if outdir is not None:
            maptype = "gainloss" if gainloss else "base"
            fig.savefig(os.path.join(outdir, f"estela_{grouper}_{maptype}.png"))
    return figs


def great_circles(lat1, lon1, ngc=16):
    """ Calculate great circles

    Args:
        lat1 (float): Latitude origin point
        lon1 (float): Longitude origin point
        ngc (int, optional): Number of great circles

    Returns:
        xarray.Dataset: dataset with distance and bearing dimensions
    """
    lat1_r = float(lat1) * D2R
    lon1_r = float(lon1) * D2R
    dist_r = xr.DataArray(dims="distance", data=np.linspace(0.5, 179.5, 180) * D2R)
    brng_r = xr.DataArray(dims="bearing", data=np.linspace(0, 360, ngc+1)[:-1] * D2R)

    sin_lat1 = np.sin(lat1_r)
    cos_lat1 = np.cos(lat1_r)
    sin_dR = np.sin(dist_r)
    cos_dR = np.cos(dist_r)

    lat2 = np.arcsin(sin_lat1*cos_dR + cos_lat1*sin_dR*np.cos(brng_r))
    lon2 = lon1_r + np.arctan2(np.sin(brng_r)*sin_dR*cos_lat1, cos_dR-sin_lat1*np.sin(lat2))
    gc = xr.Dataset({"latitude": lat2 / D2R, "longitude": (lon2 / D2R % 360).transpose()})
    gc["distance"] = dist_r / D2R
    gc["bearing"] = brng_r / D2R
    return gc


def dist_and_bearing(lat1, lat2, lon1, lon2):
    """Calculate distances and bearings from one point to others

    Args:
        lat1 (float): Latitude origin point
        lat2 (float or array): Latitude end points
        lon1 (float): Longitude origin point
        lon2 (float or array): Longitude end points

    Returns:
        float or array: distances and bearings in degrees
    """
    lat1_r = lat1 * D2R
    lat2_r = lat2 * D2R
    latdif_r = (lat2 - lat1) * D2R
    londif_r = (lon2 - lon1) * D2R

    a = np.sin(latdif_r / 2) * np.sin(latdif_r / 2) + np.cos(lat1_r) * np.cos(
        lat2_r
    ) * np.sin(londif_r / 2) * np.sin(londif_r / 2)
    a = a.clip(0., 1.) # to avoid warning for a=1.0000001,
    degdist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) / D2R

    y = np.sin(londif_r) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(londif_r)
    brng = (np.arctan2(y, x) / D2R).transpose() % 360
    return (degdist, brng)


def geographic_mask(lat0, lon0, dists, bearings):
    """Check the great circles points to find points not blocked by land

    Args:
        lat0 (float): Latitude origin point
        lon0 (float): Longitude origin point
        dists (array): Distances
        bearings (array): Bearings

    Returns:
        array: points not blocked by land
    """
    def circ_closed_range(i1, i2, n=360):
        if abs(i2 - i1) <= n / 2:
            if i2 > i1:
                rng = range(i1, i2 + 1, +1)
            else:
                rng = range(i1, i2 - 1, -1)
        else:
            if i2 > i1:
                rng = range(i1 + n, i2 - 1, -1)
            else:
                rng = range(i1, i2 + n + 1, +1)
        indices = [x % n for x in rng]
        return indices

    def update_dmax(dmax, dists, ibearings):
        for idist in range(len(dists)-1):
            d1 = dists[idist]
            d2 = dists[idist+1]
            i1 = ibearings[idist]
            i2 = ibearings[idist+1]
            for i in circ_closed_range(i1, i2):
                if i2 == i1:
                    d = d1
                else:
                    d = d1 + (d2-d1)/(i2-i1)*(i-i1)
                dmax[i] = min(dmax[i], d)
        return dmax

    dmax = 179.99 * np.ones(360)
    shpfilename = shpreader.natural_earth(
        resolution="110m", category="physical", name="coastline"
    )
    # shpfilename = shpreader.gshhs(scale='c', level=1)
    coastlines = shpreader.Reader(shpfilename).records()
    for c in coastlines:
        lonland, latland = map(np.array, zip(*c.geometry.coords))
        distland, bearingland = dist_and_bearing(lat0, latland, lon0, lonland)
        ibearingland = np.trunc(bearingland % 360).astype(int)
        dmax = update_dmax(dmax, distland, ibearingland)
    vland = dists < dmax[np.trunc(bearings % 360).astype(int)]
    return vland


def get_groupers(groupers):
    """ Get default groupers if the input is None

    Args:
        groupers (sequence of str or None): groupers for calc and plot functions

    Returns:
        list: groupers for calc and plot functions
    """
    if groupers is None:
        groupers = ["ALL", "time.season", "time.month"]
    return groupers


if __name__ == "__main__":
    parser()
