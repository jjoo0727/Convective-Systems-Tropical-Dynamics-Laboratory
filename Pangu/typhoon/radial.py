def radial_plot_6var(TC_n, time_delta): # radial data plot 6 variables
    TC_time = TCs[TC_n].YYYYMMDDHH
    TCG_time = TCs[TC_n].YYYYMMDDHH[TCs[TC_n].TCG_idx] + np.timedelta64(time_delta, 'h')
    TCG_loc = TCs[TC_n].track[TC_time == TCG_time][0]

    BT_lat = TCG_loc[0]
    BT_lon = TCG_loc[1]
    if BT_lon < 0:
        BT_lon += 360

    t0 = TCG_time
    lat0 = np.round(BT_lat*4)/4
    lon0 = np.round(BT_lon*4)/4

    lat_range = [lat0 + 7, lat0 - 7]
    lon_range = [lon0 - 8, lon0 + 8]

    d_ds = lnd.load_data_4d_hour(var_short='d', nc_date_hour=t0, lev_range=[100, 1000],
                                lat_range=lat_range, lon_range=lon_range, as_ds=True)
    z_ds = lnd.load_data_4d_hour(var_short='z', nc_date_hour=t0, lev_range=[100, 1000],
                                lat_range=lat_range, lon_range=lon_range, as_ds=True)
    r_ds = lnd.load_data_4d_hour(var_short='r', nc_date_hour=t0, lev_range=[100, 1000],
                                lat_range=lat_range, lon_range=lon_range, as_ds=True)
    t_ds = lnd.load_data_4d_hour(var_short='t', nc_date_hour=t0, lev_range=[100, 1000],
                                lat_range=lat_range, lon_range=lon_range, as_ds=True)

    u_ds = lnd.load_data_4d_hour(var_short='u', nc_date_hour=t0, lev_range=[100, 1000],
                                lat_range=lat_range, lon_range=lon_range, as_ds=True)
    v_ds = lnd.load_data_4d_hour(var_short='v', nc_date_hour=t0, lev_range=[100, 1000],
                                lat_range=lat_range, lon_range=lon_range, as_ds=True)
    w_ds = lnd.load_data_4d_hour(var_short='w', nc_date_hour=t0, lev_range=[100, 1000],
                                lat_range=lat_range, lon_range=lon_range, as_ds=True)

    coord_lat = d_ds['latitude'].values
    coord_lon = d_ds['longitude'].values
    coord_lev = d_ds['level'].values

    distances = np.arange(0, 501, 5)
    bearing = np.linspace(0, 358, 180)
    circles = hd.concentric_circles(BT_lat, BT_lon, distances, bearing)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(2):
        for j in range(3):
            n = i*3 + j
            if n == 0:
                data_str = 'Divergence'
                cmap = 'bwr'
                level_min = -10
                level_max = 10
                multiplier = 1e5
                unit_str = r'$\times$ 10$^5$ s$^{-1}$'
                data = d_ds.d.values*multiplier
            elif n == 1:
                data = z_ds.z.values/9.81
                data_str = 'Geopotential height anomaly'
                cmap = 'bwr'
                level_min = -25
                level_max = 25
                unit_str = r'm'
            elif n == 2:
                data = t_ds.t.values
                data_str = 'Temperature anomaly'
                cmap = 'bwr'
                level_min = -2
                level_max = 2
                unit_str = r'$^\circ$C'
            elif n == 3:
                data_u = u_ds.u.values
                data_v = v_ds.v.values
                data_str = 'Tangential wind'
                cmap = 'bwr'
                level_min = -20
                level_max = 20
                unit_str = r'm/s'
            elif n == 4:
                data_u = u_ds.u.values
                data_v = v_ds.v.values
                data_str = 'Radial wind'
                cmap = 'bwr'
                level_min = -10
                level_max = 10
                unit_str = r'm/s'
            elif n == 5:
                data = w_ds.w.values
                data_str = 'Vertical wind'
                cmap = 'bwr_r'
                level_min = -2.5
                level_max = 2.5
                unit_str = r'Pa s$^{-1}$'
            
            data_azi_mean = np.zeros((len(distances), len(coord_lev)))*np.nan

            for k in range(len(coord_lev)):
                temp_polar_lat = np.zeros((len(distances), len(bearing)))*np.nan
                temp_polar_lon = np.zeros((len(distances), len(bearing)))*np.nan
                for l in range(len(distances)):
                    for m in range(len(bearing)):
                        temp_polar_lat[l, m] = circles[distances[l]][m][0]
                        temp_polar_lon[l, m] = circles[distances[l]][m][1]

                # for vt or vr
                if n == 3:
                    data_u_interp = interpn((coord_lon, coord_lat), data_u[k, :, :].T,
                                            (temp_polar_lon, temp_polar_lat), method="linear", bounds_error=False, fill_value=np.nan)
                    data_v_interp = interpn((coord_lon, coord_lat), data_v[k, :, :].T,
                                            (temp_polar_lon, temp_polar_lat), method="linear", bounds_error=False, fill_value=np.nan)
                    data_interp = np.repeat([-np.cos(bearing/180*np.pi)], len(distances), axis=0)*data_u_interp + np.repeat([np.sin(bearing/180*np.pi)], len(distances), axis=0)*data_v_interp
                elif n == 4:
                    data_u_interp = interpn((coord_lon, coord_lat), data_u[k, :, :].T,
                                            (temp_polar_lon, temp_polar_lat), method="linear", bounds_error=False, fill_value=np.nan)
                    data_v_interp = interpn((coord_lon, coord_lat), data_v[k, :, :].T,
                                            (temp_polar_lon, temp_polar_lat), method="linear", bounds_error=False, fill_value=np.nan)
                    data_interp = np.repeat([np.sin(bearing/180*np.pi)], len(distances), axis=0)*data_u_interp + np.repeat([np.cos(bearing/180*np.pi)], len(distances), axis=0)*data_v_interp
                else:
                    data_interp = interpn((coord_lon, coord_lat), data[k, :, :].T,
                                        (temp_polar_lon, temp_polar_lat), method="linear", bounds_error=False, fill_value=np.nan)
                data_azi_mean[:, k] = np.nanmean(data_interp, axis=1)

            if n == 1 or n == 2:
                data_azi_mean = data_azi_mean - np.mean(data_azi_mean, axis=0)
            
            ctf = ax[i, j].contourf(distances, coord_lev, data_azi_mean.T, cmap=cmap, levels=np.linspace(level_min, level_max, 21), extend='both')
            cbar = fig.colorbar(ctf, ax=ax[i, j], ticks=np.linspace(level_min, level_max, 11), orientation='horizontal')
            cbar.ax.set_xlabel(f'{unit_str}')
            if n ==5:
                cbar.ax.invert_xaxis()
            ax[i, j].invert_yaxis()
            ax[i, j].set_title(f'{data_str}')
            ax[i, j].set_xlabel('Distance from center (km)')
            ax[i, j].set_ylabel('Pressure (hPa)')
            ax[i, j].set_xticks(np.arange(0, 501, 100))
            ax[i, j].set_yticks(np.arange(200, 1001, 200))
    fig.suptitle(f'{TCs[TC_n].__str__()}\n{t0.astype("datetime64[h]").__str__()}\nTCG{time_delta:+0}', fontsize=16)
    plt.tight_layout()
    plt.show()