#%%
import os
import glob
import netCDF4 as nc
from datetime import datetime
import xml.etree.ElementTree as ET


# Parse coordinate with unit handling
def parse_coord(node):
    v = float(node.text)
    u = (node.get("units") or "").lower()   # e.g., "deg w", "deg n"
    if "deg w" in u or "deg s" in u:
        v = -abs(v)
    return v

def ensemble_data_maker(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    ensemble_data = {}
    for data in root.findall('.//data'):
        if data.get('member') is None:
            continue
        member = int(data.get('member'))
        ensemble_data.setdefault(member, {})

        for disturbance in data.findall('.//disturbance'):
            disturbance_id = disturbance.get('ID')
            if disturbance_id is None:
                continue

            # ✅ cycloneName 파싱 (없으면 None)
            cyclone_name = disturbance.findtext('cycloneName')  # XML 태그가 <cycloneName>인 경우
            # 필요하면 추가 메타도 같이
            basin = disturbance.findtext('basin')
            cyclone_number = disturbance.findtext('cycloneNumber')

            if disturbance_id not in ensemble_data[member]:
                ensemble_data[member][disturbance_id] = {
                    "_meta": {
                        "cycloneName": cyclone_name,
                        "basin": basin,
                        "cycloneNumber": cyclone_number,
                    },
                    "_track": {}
                }

            track = ensemble_data[member][disturbance_id]["_track"]

            for fix in disturbance.findall('fix'):
                valid_time = fix.findtext('validTime')
                if not valid_time:
                    continue
                valid_time = datetime.strptime(valid_time, "%Y-%m-%dT%H:%M:%SZ")

                lat_node = fix.find('latitude')
                lon_node = fix.find('longitude')
                pres_node = fix.find('.//pressure')

                if lat_node is None or lon_node is None or pres_node is None or pres_node.text is None:
                    continue

                latitude = parse_coord(lat_node)
                longitude = parse_coord(lon_node)
                pressure = float(pres_node.text)

                track[valid_time] = {'lon': longitude, 'lat': latitude, 'pres': pressure}

    return ensemble_data




def xml_to_netcdf(xml_filename, nc_filename):
    try:
        ensemble_data = ensemble_data_maker(xml_filename)

        with nc.Dataset(nc_filename, 'w', format='NETCDF4') as dataset:
            for member, disturbances in ensemble_data.items():
                member_group = dataset.createGroup(f'member_{member}')

                for disturbance_id, payload in disturbances.items():
                    disturbance_group = member_group.createGroup(disturbance_id)
                    disturbance_group.setncattr('disturbance_id', disturbance_id)

                    meta = payload.get("_meta", {})
                    # ✅ name을 netCDF attribute로 저장
                    disturbance_group.setncattr('cycloneName', meta.get("cycloneName") or "")
                    disturbance_group.setncattr('basin', meta.get("basin") or "")
                    disturbance_group.setncattr('cycloneNumber', meta.get("cycloneNumber") or "")

                    records = payload.get("_track", {})
                    times = sorted(records.keys())

                    disturbance_group.createDimension('time', len(times))
                    times_var = disturbance_group.createVariable('time', 'f8', ('time',))
                    times_var.units = 'hours since 1970-01-01 00:00:00'
                    times_var.calendar = 'standard'
                    times_var[:] = nc.date2num(times, units=times_var.units, calendar=times_var.calendar)

                    lat_var = disturbance_group.createVariable('lat', 'f4', ('time',))
                    lon_var = disturbance_group.createVariable('lon', 'f4', ('time',))
                    pres_var = disturbance_group.createVariable('pres', 'f4', ('time',))

                    for i, t in enumerate(times):
                        lat_var[i] = records[t]['lat']
                        lon_var[i] = records[t]['lon']
                        pres_var[i] = records[t]['pres']

    except Exception as e:
        print(f"Error processing {xml_filename}: {e}")
        
        
from dask import delayed, compute
from dask.diagnostics import ProgressBar

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def process_all_xml_to_netcdf_dask_batched(directory, batch=10, overwrite=False):
    xml_files = glob.glob(os.path.join(directory, '**', '*.xml'), recursive=True)

    for xml_batch in chunks(xml_files, batch):
        tasks = []
        for xml_file in xml_batch:
            nc_file = os.path.splitext(xml_file)[0] + '.nc'
            if (not overwrite) and os.path.exists(nc_file):
                continue
            tasks.append(delayed(xml_to_netcdf)(xml_file, nc_file))

        if tasks:
            with ProgressBar():
                compute(*tasks, scheduler="threads")
# 실행
directory = '/data09/TC/TIGGE/ecmf/'
process_all_xml_to_netcdf_dask_batched(directory, overwrite=True)