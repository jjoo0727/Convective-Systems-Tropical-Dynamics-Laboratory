#%%
import os
import glob
import netCDF4 as nc
from datetime import datetime
import xml.etree.ElementTree as ET

def ensemble_data_maker(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    ensemble_data = {}
    for data in root.findall('.//data'):
        if data.get('member') is None:
            continue
        member = int(data.get('member'))

        if member not in ensemble_data:
            ensemble_data[member] = {}

        for disturbance in data.findall('.//disturbance'):
            disturbance_id = disturbance.get('ID')
            basin = disturbance.find('basin').text

            if disturbance_id.endswith('00E') or basin != "Northwest Pacific":
                continue

            if disturbance_id not in ensemble_data[member]:
                ensemble_data[member][disturbance_id] = {}

            for fix in disturbance.findall('fix'):
                valid_time = fix.find('validTime').text
                valid_time = datetime.strptime(valid_time, "%Y-%m-%dT%H:%M:%SZ")
                latitude = float(fix.find('latitude').text)
                longitude = float(fix.find('longitude').text)
                pressure = float(fix.find('.//pressure').text)

                if valid_time not in ensemble_data[member][disturbance_id]:
                    ensemble_data[member][disturbance_id][valid_time] = {}
                ensemble_data[member][disturbance_id][valid_time] = {
                    'lon': longitude, 'lat': latitude, 'pres': pressure
                }
    return ensemble_data



def xml_to_netcdf(xml_filename, nc_filename):
    try:
        # Load and process XML data
        ensemble_data = ensemble_data_maker(xml_filename)
        
        # Open a new NetCDF file
        with nc.Dataset(nc_filename, 'w', format='NETCDF4') as dataset:
            # Create dimensions at the dataset level
            dataset.createDimension('member', None)

            # Create variables and groups for each member and disturbance
            for member, disturbances in ensemble_data.items():
                member_group = dataset.createGroup(f'member_{member}')
                for disturbance_id, records in disturbances.items():
                    disturbance_group = member_group.createGroup(disturbance_id)
                    disturbance_group.setncattr('disturbance_id', disturbance_id)
                    
                    # Prepare time data for this disturbance
                    times = sorted(records.keys())
                    # Create time dimension and variable for this specific disturbance
                    time_dim = disturbance_group.createDimension('time', len(times))
                    times_var = disturbance_group.createVariable('time', 'f8', ('time',))
                    # Set the units to 'hours since 1970-01-01 00:00:00'
                    times_var.units = 'hours since 1970-01-01 00:00:00'
                    times_var.calendar = 'standard'
                    times_var[:] = nc.date2num(times, units=times_var.units, calendar=times_var.calendar)
                    
                    # Create other variables
                    lat_var = disturbance_group.createVariable('lat', 'f4', ('time',))
                    lon_var = disturbance_group.createVariable('lon', 'f4', ('time',))
                    pres_var = disturbance_group.createVariable('pres', 'f4', ('time',))
                    
                    # Populate data
                    for i, valid_time in enumerate(times):
                        lat_var[i] = records[valid_time]['lat']
                        lon_var[i] = records[valid_time]['lon']
                        pres_var[i] = records[valid_time]['pres']

    except Exception as e:
        print(f"Error processing {xml_filename}: {e}")
        
        
def process_all_xml_to_netcdf(directory):
    xml_files = glob.glob(os.path.join(directory, '**', '*.xml'), recursive=True)
    
    for xml_file in xml_files:
        nc_file = os.path.splitext(xml_file)[0] + '.nc'
        print(f"Converting {xml_file} to {nc_file}")
        xml_to_netcdf(xml_file, nc_file)
        print(f"Conversion complete for {nc_file}")

directory = '/home1/jek/Pangu-Weather/input_data/TIGGE/ecmf/'
process_all_xml_to_netcdf(directory)