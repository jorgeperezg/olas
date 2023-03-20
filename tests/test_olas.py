import os

import numpy as np
import xarray as xr

from olas import estela

FILES_DIR = os.path.join(os.path.dirname(__file__), "sample_files")


def test_dist_and_bearing():
    lat1 = 0
    lat2 = np.array([1, 0, -3, 0])
    lon1 = 0
    lon2 = np.array([0, 2, 0, -4])
    dists, bearings = estela.dist_and_bearing(lat1, lat2, lon1, lon2)
    print(dists, bearings)
    dist_ok = all(dists == np.array([1.0, 2.0, 3.0, 4.0]))
    bearings_ok = all(bearings == np.array([0.0, 90.0, 180.0, 270.0]))
    assert dist_ok and bearings_ok


def test_geographic_mask():
    lat0 = -38
    lon0 = 174.5
    dsf = xr.open_mfdataset(os.path.join(FILES_DIR, "test20180101T12.nc"))
    dists, bearings = estela.dist_and_bearing(lat0, dsf.latitude, lon0, dsf.longitude)
    estela.geographic_mask(lat0, lon0, dists, bearings)


def test_estela():
    datafiles = os.path.join(FILES_DIR, "test20180101T??.nc")
    lat0 = 46  # -38  #, -13.76
    lon0 = -131  # 174.5  #, -172.07
    groupers = ["ALL", "time.season", "time.month"]
    estela.calc(datafiles, lat0, lon0, "hs", "tp", "dp", 20, groupers=groupers)
