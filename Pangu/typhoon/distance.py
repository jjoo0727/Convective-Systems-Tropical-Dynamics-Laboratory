import numpy as np

def haversine_distance(point1, point2):
    point1 = np.squeeze(np.asarray(point1))
    point2 = np.squeeze(np.asarray(point2))

    single_point1 = point1.ndim == 1
    single_point2 = point2.ndim == 1

    if single_point1 and single_point2:
        point1 = point1.reshape(1, -1)
        point2 = point2.reshape(1, -1)
    elif single_point1:
        point1 = np.tile(point1, (len(point2), 1))
    elif single_point2:
        point2 = np.tile(point2, (len(point1), 1))
    else:
        return distance_matrix_haver(point1, point2)

    lat1, lon1 = point1[:, 0], point1[:, 1]
    lat2, lon2 = point2[:, 0], point2[:, 1]

    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a + 1e-10))

    # Earth radius in kilometers (mean radius)
    R = 6371.01

    # Calculate the distance
    distance = R * c

    return distance

def distance_matrix_haver(point1, point2):
    point1 = np.squeeze(np.asarray(point1))
    point2 = np.squeeze(np.asarray(point2))

    single_point1 = point1.ndim == 1
    single_point2 = point2.ndim == 1

    if single_point1 and single_point2:
        point1 = point1.reshape(1, -1)
        point2 = point2.reshape(1, -1)
    
    dist_matrix = np.zeros((len(point1), len(point2)))
    for i, point1 in enumerate(point1):
        dist_matrix[i, :] = haversine_distance(point1, point2)
    return dist_matrix

def dx_to_dlon(dx, lat):
    R = 6371.01
    return dx * 360 / (2 * np.pi * R * np.cos(np.deg2rad(lat)))

def dy_to_dlat(dy):
    R = 6371.01
    return dy * 360 / (2 * np.pi * R) 

# Function to calculate a point given latitude, longitude, bearing, and distance
def calculate_point(lat, lon, bearing, distance, radius=6371.01):
    lat, lon, bearing = np.radians([lat, lon, bearing])
    new_lat = np.arcsin(np.sin(lat) * np.cos(distance/radius) +
                        np.cos(lat) * np.sin(distance/radius) * np.cos(bearing))
    new_lon = lon + np.arctan2(np.sin(bearing) * np.sin(distance/radius) * np.cos(lat),
                               np.cos(distance/radius) - np.sin(lat) * np.sin(new_lat))
    return np.degrees(new_lat), np.degrees(new_lon)

# Function to generate concentric circles
def concentric_circles(center_lat, center_lon, distances, bearing):
    circles = {}
    for distance in distances:
        circle_points = []
        for i_b in bearing:
            point = calculate_point(center_lat, center_lon, i_b, distance)
            circle_points.append(point)
        circles[distance] = circle_points
    return circles