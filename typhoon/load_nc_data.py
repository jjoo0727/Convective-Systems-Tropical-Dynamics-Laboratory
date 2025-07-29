import xarray as xr
import numpy as np
import pandas as pd

# variable
# (time, level, lat, lon)
# if mean_dim = level -> (time, lat, lon)
# load day unit unless nc_time_range is a single timestamp
def load_data_4d(var_short='pv', nc_time_range=[pd.Timestamp('2007-01-01'), pd.Timestamp('2007-01-02')], 
                 lev_range=[100, 300], lat_range=[40, 0], lon_range=[100, 180], mean_dim=[], zonal=False, as_ds=False):
    if not hasattr(nc_time_range, '__iter__'):
        return load_data_4d_hour(var_short=var_short, nc_date_hour=nc_time_range,
                                 lev_range=lev_range, lat_range=lat_range, lon_range=lon_range, mean_dim=mean_dim, zonal=zonal, as_ds=as_ds)
    else:
        nc_date_range = pd.date_range(pd.Timestamp(nc_time_range[0]).date(), pd.Timestamp(nc_time_range[-1]).date(), freq='D')
        ncds_arr = []
        for i_time in nc_date_range:
            date_string = i_time.strftime('%Y%m%d')
            nc_file_pattern = f'/data1/ERA-5/hourly/{var_short}/ERA-5_{var_short}_{date_string}.nc'

            ncds = xr.open_dataset(nc_file_pattern)
            if zonal:
                ncds = ncds.sel(level=slice(*lev_range), latitude=slice(*lat_range))
                ncds = ncds.sel(longitude=slice(*lon_range)) - ncds.mean(dim='longitude', skipna=False)
            else:
                ncds = ncds.sel(level=slice(*lev_range), latitude=slice(*lat_range), longitude=slice(*lon_range))

            ncds_arr.append(ncds)

        ncds = xr.concat(ncds_arr, dim='time')

        for i_mean_dim in mean_dim:
            ncds = ncds.mean(dim=i_mean_dim, skipna=False)

        if as_ds:
            return ncds
        else:
            return np.squeeze(ncds[var_short].values), ncds.coords

def load_data_4d_hour(var_short='pv', nc_date_hour=pd.Timestamp('2007-01-01'), 
                      lev_range=[100, 300], lat_range=[40, 0], lon_range=[100, 180], mean_dim=[], zonal=False, as_ds=False):
    nc_date_hour = pd.to_datetime(nc_date_hour)
    date_string = nc_date_hour.strftime('%Y%m%d')
    nc_file_pattern = f'/data1/ERA-5/hourly/{var_short}/ERA-5_{var_short}_{date_string}.nc'

    ncds = xr.open_dataset(nc_file_pattern)
    if zonal:
        ncds = ncds.sel(time=nc_date_hour, level=slice(*lev_range), latitude=slice(*lat_range))
        ncds = ncds.sel(longitude=slice(*lon_range)) - ncds.mean(dim='longitude', skipna=False)
    else:
        ncds = ncds.sel(time=nc_date_hour, level=slice(*lev_range), latitude=slice(*lat_range), longitude=slice(*lon_range))
    for i_mean_dim in mean_dim:
        ncds = ncds.mean(dim=i_mean_dim, skipna=False)
    
    if as_ds:
        return ncds
    else:
        return np.squeeze(ncds[var_short].values), ncds.coords

# using multiprocessing with given client
# as_ds always False (speed problem)
def load_data_4d_multi(var_short='pv', nc_time_range=[pd.Timestamp('2007-01-01'), pd.Timestamp('2007-01-02')], 
                       lev_range=[100, 300], lat_range=[40, 0], lon_range=[100, 180], mean_dim=[], zonal=False, client=None):
    nc_date_range = pd.date_range(pd.Timestamp(nc_time_range[0]).date(), pd.Timestamp(nc_time_range[-1]).date(), freq='D')
    futures = []
    nc_date_split = np.array_split(nc_date_range, len(client.ncores()))
    for i_range in nc_date_split:
        futures.append(client.submit(load_data_4d, var_short=var_short, nc_time_range=i_range,
                                     lev_range=lev_range, lat_range=lat_range, lon_range=lon_range, mean_dim=mean_dim, zonal=zonal, as_ds=False))

    results = client.gather(futures)

    # Combine the results from each processor
    results_tuple = list(zip(*results))
    nc_data = np.concatenate(results_tuple[0], axis=0)
    ncds_coords = xr.concat([xr.Dataset(result) for result in results_tuple[1][0:]], dim='time').coords

    return nc_data, ncds_coords

def load_data_4d_isen(nc_time_range=[pd.Timestamp('2007-01-01'), pd.Timestamp('2007-01-02')], 
                      lat_range=[40, 0], lon_range=[100, 180]):
    nc_date_range = pd.date_range(pd.Timestamp(nc_time_range[0]).date(), pd.Timestamp(nc_time_range[-1]).date(), freq='D')
    ncds_arr = []
    for i_time in nc_date_range:
        date_string = i_time.strftime('%Y%m%d')
        nc_file_pattern = f'/data/ERA-5/hourly/pv_350K/ERA-5_pv_350K_{date_string}.nc'

        ncds = xr.open_dataset(nc_file_pattern)
        ncds = ncds.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))

        ncds_arr.append(ncds)

    ncds = xr.concat(ncds_arr, dim='time')

    return ncds['pv'].values, ncds.coords

def load_data_4d_isen_hour(nc_date_hour=pd.Timestamp('2007-01-01'), 
                           lat_range=[40, 0], lon_range=[100, 180]):
    nc_date_hour = pd.to_datetime(nc_date_hour)
    date_string = nc_date_hour.strftime('%Y%m%d')
    nc_file_pattern = f'/data/ERA-5/hourly/pv_350K/ERA-5_pv_350K_{date_string}.nc'

    ncds = xr.open_dataset(nc_file_pattern)
    ncds = ncds.sel(time=nc_date_hour, latitude=slice(*lat_range), longitude=slice(*lon_range))
    
    return np.squeeze(ncds['pv'].values), ncds.coords





