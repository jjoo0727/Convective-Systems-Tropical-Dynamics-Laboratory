#%%
#group1, group2 평균장 태풍 추적
group_position = {0:[],1:[]}

for predict_interval in np.arange(0,169,6):
    group1_values = []
    group2_values = []
    predict_time = first_time + timedelta(hours=int(predict_interval))
    predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
    
    # fig, ax = plt.subplots(1, 3, figsize=(10*latlon_ratio*3, 10), subplot_kw={'projection': proj})
    # fig.suptitle(f'{predict_str} {alt}hPa height(m) & wind', fontsize=45)

    for ens in group1:
        output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
        met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        z_diff = met.met_data('z', level = 300) - met.met_data('z', level = 500)
        mslp = met.met_data('MSLP', level = 'sf')
        ws = met.wind_speed(level = 'sf')
        
        # ax.contour(lon_grid, lat_grid, z, colors='red', levels = levels, alpha=0.1)
        group1_values.append(np.array([mslp, z_diff, ws]))
        
    for ens in group2:
        output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
        met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        mslp = met.met_data('MSLP', level = 'sf')
        z_diff = met.met_data('z', level = 300) - met.met_data('z', level = 500)
        ws = met.wind_speed(level = 'sf')
        
        # ax.contour(lon_grid, lat_grid, z, colors='red', levels = levels, alpha=0.1)
        group2_values.append(np.array([mslp,z_diff,ws]))
        
    group1_mean = np.mean(np.stack(group1_values), axis=0)
    group2_mean = np.mean(np.stack(group2_values), axis=0)
    group_mean = [group1_mean, group2_mean]
    # plt.imshow(group2_mean[0])
    # plt.show()
    for i in range(2):
        group_position[i] = plot_min_value(group_mean[i][0], lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                    group_mean[i][2], predict_str, group_mean[i][1], storm_lon, storm_lat, storm_mslp, storm_time, 
                                    group_position[i], ax = None, mask_size = 10)
    
    

    
fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
ax.set_title(f'{first_str} (+{predict_interval_list[-1]}h)\n{predict_str}', fontsize=20, loc = 'left')
ax.set_title(f'ENS{surface_str}{upper_str}{perturation_scale} Track\n{storm_name}', fontsize=20, loc = 'right')
ax.set_extent(extent, crs=proj)
setup_map(ax)

ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'best track', transform=ax.projection, zorder=3)
model_pred_sc = ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=3)
plt.colorbar(model_pred_sc, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)


for i in range(len(storm_time)):
    new_time = storm_time[i].strftime("%Y/%m/%d/%HUTC")
    if new_time.endswith('00UTC'):
        dx, dy = 3, -3  # 시간 나타낼 위치 조정
        new_lon, new_lat = storm_lon[i] + dx, storm_lat[i] + dy
        
        # annotate를 사용하여 텍스트와 함께 선(화살표)을 그림
        ax.text(storm_lon[i], storm_lat[i], new_time[8:-6]
                , horizontalalignment='center', verticalalignment='bottom', fontsize=10)

for i in range(2):
    lons = [pos['lon'] for pos in group_position[i]]
    lats = [pos['lat'] for pos in group_position[i]]
    min_values = [pos['mslp'] for pos in group_position[i]]
    pred_times = [pos['time'] for pos in group_position[i]]

    lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1)

    for i in range(len(pred_times)):
        if pred_times[i].endswith('00UTC'):
            ax.text(lons[i],lats[i], pred_times[i][8:-6]
                , horizontalalignment='center', verticalalignment='bottom', fontsize=10)

#%%
group_position = {0:[],1:[]}

# for predict_interval in np.arange(0,169,6):
#     group1_values = []
#     group2_values = []
#     predict_time = first_time + timedelta(hours=int(predict_interval))
#     predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
    
    # fig, ax = plt.subplots(1, 3, figsize=(10*latlon_ratio*3, 10), subplot_kw={'projection': proj})
    # fig.suptitle(f'{predict_str} {alt}hPa height(m) & wind', fontsize=45)

for ens in group1:    
    for mp in min_position[ens]:
        predict_interval = datetime.strptime(mp['time'],"%Y/%m/%d/%HUTC")-first_time
        predict_interval = int(predict_interval.total_seconds() / 3600)
        output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
        met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        z = met.met_data('z', level = 'all')
        t = met.met_data('t', level = 'all')
        q = met.met_data('q', level = 'all')
        u = met.met_data('u', level = 'all')
        v = met.met_data('v', level = 'all')
        
                
        
        # ax.contour(lon_grid, lat_grid, z, colors='red', levels = levels, alpha=0.1)
        # group1_values.append(np.array([mslp, z_diff, ws]))
#%%
#group1, group2 평균장 태풍 위치에서 연직 그림
for gn in group_position:
    for gp in group_position[gn]:
        
        group_values = []

        predict_interval = datetime.strptime(gp['time'], "%Y/%m/%d/%HUTC") - first_time   
        predict_interval = int(predict_interval.total_seconds() / 3600)
        output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
        
        if gn == 0:
            for ens in group1:
                output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
                met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                z = met.met_data('z', level = 'all')
                t = met.met_data('t', level = 'all')
                q = met.met_data('q', level = 'all')
                v = met.met_data('v', level = 'all')
                group_values.append(np.array([z,t,q,v]))

        else:
            for ens in group2:
                output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
                met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                z = met.met_data('z', level = 'all')
                t = met.met_data('t', level = 'all')
                q = met.met_data('q', level = 'all')
                v = met.met_data('v', level = 'all')
                group_values.append(np.array([z,t,q,v]))
        
        group_mean = np.mean(np.stack(group_values), axis=0)
        
        lon_space = 40
        
        x = np.arange(gp['lon']-lon_space/4,gp['lon']+lon_space/4, 0.25)
        y = np.array(pres_list, dtype=np.float32)

        xx, yy = np.meshgrid(x, y)
        
        z = group_mean[0][:,gp['idx'][0],gp['idx'][1]-lon_space:gp['idx'][1]+lon_space]
        t = group_mean[1][:,gp['idx'][0],gp['idx'][1]-lon_space:gp['idx'][1]+lon_space]
        q = group_mean[2][:,gp['idx'][0],gp['idx'][1]-lon_space:gp['idx'][1]+lon_space]
        v = group_mean[3][:,gp['idx'][0],gp['idx'][1]-lon_space:gp['idx'][1]+lon_space]
        
        plt.title(f'{gp["time"]} group {gn+1}')
        plot1 = plt.contourf(xx, yy, ep_t(t,yy,q), cmap = 'gist_ncar', levels = np.linspace(320,360,41), extend = 'both')
        plot2 = plt.contour(xx, yy, v, cmap = 'seismic', levels = np.linspace(-50,50,21))
        # plt.imshow(z[:,mp['idx'][0],mp['idx'][1]-lon_space:mp['idx'][1]+lon_space])
        # plt.imshow(ep_t(t, yy, q))
        plt.gca().invert_yaxis()
        cbar1 = plt.colorbar(plot1)
        cbar1.set_label('Equivalent Potential Temperature(K)') 
        plt.clabel(plot2, plot2.levels, fmt={level: str(int(level))+'m/s' for level in plot2.levels})
        plt.show()
        