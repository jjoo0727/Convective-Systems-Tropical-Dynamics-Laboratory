import numpy as np

class WindFieldSolver:
    
    #######################################################################
    # Get TY wind field solving Poisson eq
    # Presume source term from average relative vort, divergence in TY area
    # lon, lat 그리드, 태풍 중심 위치, vorticity, divergence, 태풍 반경, 그리드 사이즈 대입하여 태풍에 의한 wind field 생성
    #######################################################################
    def __init__(self, lat_grid, lon_grid, center_lat, center_lon, vort, div, radius=333, dx=111e3/4, dy=111e3/4):
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.dx = dx
        self.dy = dy
        self.vort = vort
        self.div = div
        self.R = 6371  # Earth's radius in kilometers
        
    #두 위경도가 주어지면 거리를 구하는 함수
    @staticmethod
    @jit(nopython=True)
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1, lon1 = [lat1, lon1]/180*np.pi
        lat2, lon2 = [lat2, lon2]/180*np.pi
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = 6371 * c
        return distance

    
    #태풍 없앨 때 참조할 vort, div를 지정하기(제거한 vort, div 영역 지정), 본인이 원하는대로 수정하길 요망
    #vorticity가 radius 증가에 따라 감소하다가 증가하는 시점의 반경을 태풍의 반지름으로 지정
    
    def create_source_term(self):
        # Create arrays for center latitude and longitude
        center_lat_array = np.ones_like(self.lat_grid) * self.center_lat
        center_lon_array = np.ones_like(self.lon_grid) * self.center_lon

        # Calculate distances using the Haversine formula
        distances = haversine_distance(center_lat_array, center_lon_array, self.lat_grid, self.lon_grid)

        # Define the maximum distance to calculate intervals dynamically
        max_distance = 1500 
        interval_size = 50
        intervals = [(i, i + interval_size) for i in range(0, max_distance, interval_size)]
        
        previous_average = None
        optimal_radius = None
        vort_avg = []

        # Calculate average values for each interval and check for increasing pattern
        for i, (low, high) in enumerate(intervals):
            mask = (distances > low) & (distances <= high)
            if np.any(mask):
                masked_sf = self.vort[mask]
                current_average = np.mean(masked_sf)
                vort_avg.append([high, current_average])
                
                # Check if the average is increasing
                if previous_average is not None and current_average > previous_average:
                    optimal_radius = intervals[i-1][1]  # The upper bound of the previous interval
                    break
                previous_average = current_average

        # If no increase is found, consider the last interval's upper bound as optimal radius
        if optimal_radius is None:
            optimal_radius = intervals[-1][1]

        # Create final mask based on optimal radius and apply to data
        optimal_radius = 250
        final_mask = distances <= optimal_radius
        sf_data = np.zeros_like(self.vort)
        vp_data = np.zeros_like(self.div)
        sf_data[final_mask] = self.vort[final_mask]
        vp_data[final_mask] = self.div[final_mask]
        print(optimal_radius)
        vort_avg = np.array(vort_avg)
        plt.plot(vort_avg[:,0],vort_avg[:,1])
        return sf_data, vp_data, optimal_radius
        
    #vorticity로부터 stream function(sf)을 divergence로부터 velocity potential(vp)구하기
    #1000번 반복해서 수치적인 해 구하기
    @staticmethod
    @jit(nopython=True)
    def solve_poisson_with_scaling(lat_grid, b, dx, dy, scaling_factors):
        p = np.zeros_like(lat_grid)
        for _ in range(1000):  # Number of iterations for convergence
            p_old = p.copy()
            for i in range(1, lat_grid.shape[0]-1):
                for j in range(1, lat_grid.shape[1]-1):
                    p[i, j] = ((p_old[i+1, j] + p_old[i-1, j]) * dy**2 +
                               (p_old[i, j+1] + p_old[i, j-1]) * dx**2 * scaling_factors[i, j]**2 -
                               b[i, j] * dy**2 * dx**2 * scaling_factors[i, j]**2) / (2 * (dy**2 + dx**2 *scaling_factors[i, j]**2))
        return p
    
    
    #stream function(sf)에 의한 바람장, velocity potential(vp)에 의한 바람장 유도
    def compute_wind_field(self, psi, type='sf'):
        if type == 'vp':
            u = np.gradient(psi, axis=1) / (np.cos(np.radians(self.lat_grid)) * self.dx)
            v = -np.gradient(psi, axis=0) / self.dy
        else:
            u = np.gradient(psi, axis=0) / self.dy
            v = np.gradient(psi, axis=1) / (np.cos(np.radians(self.lat_grid)) * self.dx)
        return u, v

    def solve(self):
        #위도 변화에 따른 dx 변화 고려
        scaling_factors = np.cos(np.radians(self.lat_grid))
        
        #태풍 없앨 때 참조할 vort, div를 지정하기(제거한 vort, div 영역 지정, 본인이 원하는대로 수정하길 요망)
        sf_source, vp_source, remove_r = self.create_source_term()
        
        #vorticity로부터 stream function(sf)을 divergence로부터 velocity potential(vp)구하기
        sf = self.solve_poisson_with_scaling(self.lat_grid, sf_source, self.dx, self.dy, scaling_factors)
        vp = self.solve_poisson_with_scaling(self.lat_grid, vp_source, self.dx, self.dy, scaling_factors)
        
        #stream function(sf)에 의한 바람장, velocity potential(vp)에 의한 바람장 유도
        u_sf, v_sf = self.compute_wind_field(sf, 'sf')
        u_vp, v_vp = self.compute_wind_field(vp, 'vp')
        
        #stream function(sf)에 의한 바람장, velocity potential(vp)에 의한 바람장 더해서 태풍 바람장 구하기
        u = u_sf + u_vp
        v = v_sf + v_vp
        
        return u, v