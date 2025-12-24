"""ESTELA for Unstructured Grid
Based on https://doi.org/10.1007/s10236-014-0740-7

Modified to read unstructured grid data from WaveWatch III
OPTIMIZED VERSION: Memory efficient, month-by-month processing
UPDATED: Added Gain/Loss calculation and plotting for unstructured grids
"""

import logging
import os
from glob import glob
from pathlib import Path
from datetime import datetime
from dateutil.rrule import rrule, MONTHLY

import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from cartopy.io import shapereader
from matplotlib import pyplot as plt
from scipy import special
from scipy.spatial import KDTree

D2R = np.pi / 180.0
"""degrees to radians factor"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _check_spr_available(data_dir, start_date):
    """Check if spr (directional spread) files are available
    
    Args:
        data_dir: Base directory containing WW3 output
        start_date: Start date in 'YYYYMM' format
    
    Returns:
        bool: True if spr file exists for the start date
    """
    dt = datetime.strptime(start_date, '%Y%m')
    date_str = dt.strftime('%Y%m')
    dir_name = f"glo.{date_str}0100"
    dir_path = Path(data_dir) / dir_name
    
    if not dir_path.exists():
        return False
    
    spr_file = dir_path / f"ww3.{date_str}_spr.nc"
    return spr_file.exists()


def _read_spatial_coords(data_dir, start_date):
    """Read spatial coordinates from first file only (they don't change)"""
    dt = datetime.strptime(start_date, '%Y%m')
    date_str = dt.strftime('%Y%m')
    dir_name = f"glo.{date_str}0100"
    dir_path = Path(data_dir) / dir_name
    
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {dir_path}")
    
    hs_file = dir_path / f"ww3.{date_str}_hs.nc"
    
    with xr.open_dataset(hs_file) as ds:
        if 'longitude' in ds.coords:
            lon_data = ds.longitude.values
            lat_data = ds.latitude.values
        elif 'longitude' in ds.variables:
            lon_data = ds['longitude'].values
            lat_data = ds['latitude'].values
        else:
            raise ValueError("Cannot find longitude/latitude in dataset")
        
        # Handle multi-dimensional coordinates
        if lon_data.ndim > 1:
            if lon_data.shape[0] < lon_data.shape[1]:
                lon_data = lon_data[0, :]
                lat_data = lat_data[0, :]
            else:
                lon_data = lon_data[:, 0]
                lat_data = lat_data[:, 0]
        
        # Convert longitude to [0, 360] if needed
        if np.any(lon_data < 0):
            lon_data = lon_data % 360.0
    
    return lon_data, lat_data


def _read_triangulation(data_dir, start_date):
    """Read triangulation connectivity from first file"""
    dt = datetime.strptime(start_date, '%Y%m')
    date_str = dt.strftime('%Y%m')
    dir_name = f"glo.{date_str}0100"
    dir_path = Path(data_dir) / dir_name
    
    hs_file = dir_path / f"ww3.{date_str}_hs.nc"
    
    with xr.open_dataset(hs_file) as ds:
        # Look for triangulation variable (common names)
        tri_var = None
        for var_name in ['tri', 'nv', 'element', 'connectivity']:
            if var_name in ds.variables:
                tri_var = var_name
                break
        
        if tri_var is None:
            logging.warning("No triangulation variable found in dataset")
            return None
        
        tri = ds[tri_var].values
        
        # Ensure zero-based indexing
        if tri.min() == 1:
            tri = tri - 1
        
        # Transpose if needed to get (3, n_triangles) shape
        if tri.shape[0] != 3:
            tri = tri.T
        
        return tri
    
    return None


def _load_month_data(data_dir, year_month, valid_nodes_mask=None, load_spr=False):
    """Load data for a single month, optionally only valid nodes
    
    Args:
        data_dir: Base directory containing WW3 output
        year_month: Date string in 'YYYYMM' format
        valid_nodes_mask: Optional mask for subsetting nodes
        load_spr: Whether to load directional spread (spr) data
    
    Returns:
        Dataset with hs, tp, dp, and optionally spr
    """
    date_str = year_month
    dir_name = f"glo.{date_str}0100"
    dir_path = Path(data_dir) / dir_name
    
    if not dir_path.exists():
        logging.warning(f"Directory not found: {dir_path}")
        return None
    
    hs_file = dir_path / f"ww3.{date_str}_hs.nc"
    fp_file = dir_path / f"ww3.{date_str}_fp.nc"
    dir_file = dir_path / f"ww3.{date_str}_dir.nc"
    spr_file = dir_path / f"ww3.{date_str}_spr.nc"
    
    if not all([hs_file.exists(), fp_file.exists(), dir_file.exists()]):
        logging.warning(f"Missing files in {dir_path}")
        return None
    
    # Check if spr file exists if needed
    if load_spr and not spr_file.exists():
        logging.warning(f"SPR file not found: {spr_file}")
        return None
    
    logging.info(f"Loading {date_str}...")
    
    # Open with chunking for memory efficiency
    ds_hs = xr.open_dataset(hs_file, chunks={'time': 50})
    ds_fp = xr.open_dataset(fp_file, chunks={'time': 50})
    ds_dir = xr.open_dataset(dir_file, chunks={'time': 50})
    
    # If we have valid nodes mask, subset immediately
    if valid_nodes_mask is not None:
        node_dim = [d for d in ds_hs['hs'].dims if d != 'time'][0]
        ds_hs = ds_hs.isel({node_dim: valid_nodes_mask})
        ds_fp = ds_fp.isel({node_dim: valid_nodes_mask})
        ds_dir = ds_dir.isel({node_dim: valid_nodes_mask})
    
    # Combine datasets
    ds_dict = {
        'hs': ds_hs['hs'],
        'tp': 1.0 / ds_fp['fp'].where(ds_fp['fp'] > 0),
        'dp': ds_dir['dir']
    }
    
    # Load spr if requested
    if load_spr:
        ds_spr = xr.open_dataset(spr_file, chunks={'time': 50})
        if valid_nodes_mask is not None:
            ds_spr = ds_spr.isel({node_dim: valid_nodes_mask})
        ds_dict['spr'] = ds_spr['spr']
        ds_spr.close()
    
    # Combine and load to memory (only one month)
    ds = xr.Dataset(ds_dict).load()
    
    # Close file handles
    ds_hs.close()
    ds_fp.close()
    ds_dir.close()
    
    return ds


def calc_unstructured(data_dir, start_date, end_date, lat0, lon0, si=None, groupers=None):
    """
    Calculate ESTELA dataset for unstructured grid
    OPTIMIZED: Processes month-by-month with geographic masking before data loading
    
    Args:
        data_dir (str): Base directory containing WW3 output
        start_date (str): Start date 'YYYYMM'
        end_date (str): End date 'YYYYMM'
        lat0 (float): Latitude of target point
        lon0 (float): Longitude of target point
        si (float or None): Directional spread value (constant). If None, will try to read 
                           from spr files. If no spr files, defaults to 20.
        groupers (list): Time groupings ['ALL', 'time.season', 'time.month']
    
    Returns:
        xarray.Dataset: ESTELA dataset with F and traveltime fields
    """
    logging.info("="*80)
    logging.info(f"ESTELA calculation for lat0={lat0}, lon0={lon0}")
    logging.info(f"Period: {start_date} to {end_date}")
    logging.info("="*80)
    
    groupers = get_groupers(groupers)
    lon0 = lon0 % 360.0
    
    # Determine si strategy:
    # 1. If si is explicitly provided (not None), use it (fixed value)
    # 2. If si is None, check if spr files exist
    # 3. If no spr files, use default value of 20
    
    use_variable_si = False
    if si is None:
        # Check if spr files are available
        if _check_spr_available(data_dir, start_date):
            logging.info("No fixed SI provided. SPR files found - will use variable directional spread from files")
            use_variable_si = True
        else:
            logging.info("No fixed SI provided and no SPR files found. Using default SI = 20")
            si = 20
    else:
        logging.info(f"Using fixed directional spread: SI = {si}")
    
    # Step 1: Read coordinates and triangulation ONCE
    logging.info("Reading spatial coordinates (once only)...")
    lons, lats = _read_spatial_coords(data_dir, start_date)
    nnodes = len(lons)
    logging.info(f"Total nodes: {nnodes}")
    
    # Read triangulation
    logging.info("Reading triangulation...")
    tri = _read_triangulation(data_dir, start_date)
    
    # Step 2: Calculate geographic mask BEFORE loading data
    logging.info("Calculating geographic mask...")
    dists, bearings = dist_and_bearing(lat0, lats, lon0, lons)
    vland = geographic_mask(lat0, lon0, dists, bearings)
    valid_nodes = np.where(vland)[0]
    nvalid = len(valid_nodes)
    
    logging.info(f"Valid nodes: {nvalid} ({100*nvalid/nnodes:.1f}%)")
    logging.info(f"Filtered {nnodes - nvalid} nodes blocked by land")
    
    # Step 3: Pre-calculate coefficients for valid nodes only
    logging.info("Pre-calculating coefficients...")
    dist_m_valid = dists[valid_nodes] * 6371000 * D2R
    bearings_valid = bearings[valid_nodes]
    
    va = 1.4 * 10**-5
    rowroa = 1 / 0.0013
    k_dissipation_valid = (
        -dist_m_valid / (rowroa * 9.81**2) * 4 * (2 * va) ** 0.5 * (2 * np.pi) ** 3.5
    )
    
    th1_sin_valid = np.sin(0.5 * bearings_valid * D2R)
    th1_cos_valid = np.cos(0.5 * bearings_valid * D2R)
    
    # Directional spread - prepare for fixed or variable
    if use_variable_si:
        # Will compute s, A2, coef_spread for each timestep based on spr data
        logging.info("Variable directional spread mode - will compute coefficients per timestep")
        s_fixed = None
        coef_spread_fixed = None
    else:
        # Fixed directional spread - precompute once
        logging.info(f"Fixed directional spread mode - precomputing coefficients for SI = {si}")
        s_fixed = (2 / (si * np.pi / 180) ** 2) - 1
        A2 = special.gamma(s_fixed + 1) / (special.gamma(s_fixed + 0.5) * 2 * np.pi**0.5)
        coef_spread_fixed = A2 * np.pi / 180
    
    # Step 4: Initialize accumulators
    grouped_results = {}
    
    # Step 5: Process month by month
    start = datetime.strptime(start_date, '%Y%m')
    end = datetime.strptime(end_date, '%Y%m')
    
    n_months = 0
    total_timesteps = 0
    
    for dt in rrule(MONTHLY, dtstart=start, until=end):
        year_month = dt.strftime('%Y%m')
        
        # Load only this month's data (only valid nodes)
        ds_month = _load_month_data(data_dir, year_month, valid_nodes, load_spr=use_variable_si)
        
        if ds_month is None:
            continue
        
        n_months += 1
        ntimes = len(ds_month.time)
        total_timesteps += ntimes
        
        logging.info(f"Processing {year_month}: {ntimes} timesteps")
        
        # Get data arrays
        hs_data = ds_month['hs'].values
        tp_data = ds_month['tp'].values
        dp_data = ds_month['dp'].values
        
        # Calculate dissipation coefficients
        with np.errstate(divide='ignore', invalid='ignore'):
            coef_dissipation = np.exp(k_dissipation_valid[np.newaxis, :] * (tp_data**-3.5))
            coef_dissipation = np.nan_to_num(coef_dissipation, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate directional coefficients
        th2 = 0.5 * dp_data * D2R
        
        # Handle directional spread
        if use_variable_si:
            # Variable spread from spr data
            spr_data = ds_month['spr'].values
            # Clip to reasonable range (15-45 degrees as in original estela.py)
            spr_data = np.clip(spr_data, 15.0, 45.0)
            
            # Calculate s and A2 for each timestep and node
            s = (2 / (spr_data * np.pi / 180) ** 2) - 1
            A2 = special.gamma(s + 1) / (special.gamma(s + 0.5) * 2 * np.pi**0.5)
            coef_spread = A2 * np.pi / 180
        else:
            # Fixed spread - use precomputed values
            s = s_fixed
            coef_spread = coef_spread_fixed
        
        coef_direction = np.abs(
            np.cos(th2) * th1_cos_valid[np.newaxis, :] + 
            np.sin(th2) * th1_sin_valid[np.newaxis, :]
        ) ** (2.0 * s)
        
        # Calculate energy
        S_th = hs_data**2 / 16 * coef_dissipation * coef_direction * coef_spread
        Stp_th = tp_data * S_th
        
        # Accumulate by groupers
        month_num = dt.month
        
        for grouper in groupers:
            if grouper == "ALL":
                keys = ["ALL"]
            elif grouper == "time.season":
                if month_num in [12, 1, 2]:
                    keys = ["DJF"]
                elif month_num in [3, 4, 5]:
                    keys = ["MAM"]
                elif month_num in [6, 7, 8]:
                    keys = ["JJA"]
                else:
                    keys = ["SON"]
            elif grouper == "time.month":
                keys = [f"m{month_num:02d}"]
            else:
                keys = [grouper]
            
            for key in keys:
                if key not in grouped_results:
                    grouped_results[key] = {
                        'S_th': np.zeros(nvalid),
                        'Stp_th': np.zeros(nvalid),
                        'ntime': 0
                    }
                
                grouped_results[key]['S_th'] += S_th.sum(axis=0)
                grouped_results[key]['Stp_th'] += Stp_th.sum(axis=0)
                grouped_results[key]['ntime'] += ntimes
        
        # Clean up
        del ds_month, S_th, Stp_th
        
        if n_months % 6 == 0:
            logging.info(f"Completed {n_months} months, {total_timesteps} timesteps")
    
    if n_months == 0:
        raise ValueError(f"No data found between {start_date} and {end_date}")
    
    logging.info(f"Processed {n_months} months, {total_timesteps} total timesteps")
    
    # Step 6: Calculate final results
    logging.info("Creating ESTELA dataset...")
    
    time_keys = sorted(grouped_results.keys())
    F_list = []
    traveltime_list = []
    
    for key in time_keys:
        result = grouped_results[key]
        ntime = result['ntime']
        
        if ntime == 0:
            continue
        
        # Calculate for valid nodes
        Fdeg_valid = 1.025 * 9.81 * result['Stp_th'] / ntime * 9.81 / 4 / np.pi
        F_valid = 360 * Fdeg_valid
        
        # Calculate group velocity with robust handling
        mask_sufficient = result['S_th'] > 1e-10
        
        # Initialize with NaN
        cg_mps_valid = np.full(nvalid, np.nan)
        traveltime_valid = np.full(nvalid, np.nan)
        
        # Calculate only for nodes with sufficient energy
        if mask_sufficient.sum() > 0:
            Tp_mean = result['Stp_th'][mask_sufficient] / result['S_th'][mask_sufficient]
            cg_mps_valid[mask_sufficient] = Tp_mean * 9.81 / (4 * np.pi)
            
            # Calculate travel time
            traveltime_valid[mask_sufficient] = (
                dist_m_valid[mask_sufficient] / (cg_mps_valid[mask_sufficient] * 86400)
            )
            
            # Clip to physically reasonable range
            traveltime_valid = np.clip(traveltime_valid, 0.01, 365)
        
        logging.info(f"  {key}: {mask_sufficient.sum()}/{nvalid} nodes with valid data")
        
        # Create full arrays with NaN
        F_full = np.full(nnodes, np.nan)
        traveltime_full = np.full(nnodes, np.nan)
        
        F_full[valid_nodes] = F_valid
        traveltime_full[valid_nodes] = traveltime_valid
        
        F_list.append(F_full)
        traveltime_list.append(traveltime_full)
    
    # Create dataset
    lat0_arr = xr.DataArray(dims="site", data=np.array([lat0]))
    lon0_arr = xr.DataArray(dims="site", data=np.array([lon0]))
    sites = xr.Dataset(dict(lat0=lat0_arr, lon0=lon0_arr))
    
    estelas = xr.Dataset({
        'F': xr.DataArray(
            data=np.array(F_list),
            dims=['time', 'node'],
            coords={
                'time': time_keys,
                'longitude': ('node', lons),
                'latitude': ('node', lats)
            },
            attrs={'units': '$\\frac{kW}{m\\circ}$'}
        ),
        'traveltime': xr.DataArray(
            data=np.array(traveltime_list),
            dims=['time', 'node'],
            coords={
                'time': time_keys,
                'longitude': ('node', lons),
                'latitude': ('node', lats)
            },
            attrs={'units': 'days'}
        )
    }).merge(sites)
    
    estelas.attrs['start_time'] = start_date
    estelas.attrs['end_time'] = end_date
    estelas.attrs['nvalid_nodes'] = nvalid
    estelas.attrs['ntotal_nodes'] = nnodes
    
    # Store triangulation if available
    if tri is not None:
        estelas.attrs['has_triangulation'] = 1  # Use int instead of bool for NetCDF compatibility
        estelas['triangulation'] = xr.DataArray(
            data=tri,
            dims=['vertex', 'triangle'],
            attrs={'description': 'Triangulation connectivity (0-based indices)'}
        )
    else:
        estelas.attrs['has_triangulation'] = 0  # Use int instead of bool for NetCDF compatibility
    
    logging.info("="*80)
    logging.info(f"ESTELA calculation completed!")
    logging.info(f"Time groups: {list(estelas.time.values)}")
    logging.info("="*80)
    
    return estelas


def _filter_valid_triangles(tri, valid_mask):
    """
    Filter triangles to keep only those with all three vertices having valid data
    
    Args:
        tri: (3, n_triangles) array of node indices
        valid_mask: boolean mask indicating which nodes have valid data
    
    Returns:
        filtered_tri: (3, n_valid_triangles) array
        valid_tri_mask: boolean mask of valid triangles
    """
    # Check if all three vertices of each triangle are valid
    vertex0_valid = valid_mask[tri[0, :]]
    vertex1_valid = valid_mask[tri[1, :]]
    vertex2_valid = valid_mask[tri[2, :]]
    
    valid_tri_mask = vertex0_valid & vertex1_valid & vertex2_valid
    filtered_tri = tri[:, valid_tri_mask]
    
    return filtered_tri, valid_tri_mask


def _compute_triangle_centroids(lons, lats, tri):
    """
    Compute centroids of triangles
    
    Args:
        lons: longitude of all nodes
        lats: latitude of all nodes
        tri: (3, n_triangles) array of node indices
    
    Returns:
        centroid_lons: longitude of centroids
        centroid_lats: latitude of centroids
    """
    # Get vertices of all triangles
    v0_lons = lons[tri[0, :]]
    v0_lats = lats[tri[0, :]]
    v1_lons = lons[tri[1, :]]
    v1_lats = lats[tri[1, :]]
    v2_lons = lons[tri[2, :]]
    v2_lats = lats[tri[2, :]]
    
    # Compute centroids (simple average)
    centroid_lons = (v0_lons + v1_lons + v2_lons) / 3.0
    centroid_lats = (v0_lats + v1_lats + v2_lats) / 3.0
    
    return centroid_lons, centroid_lats


def _point_in_triangle(p, v0, v1, v2):
    """
    Check if point p is inside triangle (v0, v1, v2) using barycentric coordinates
    
    Args:
        p: (2,) array - point coordinates [lon, lat]
        v0, v1, v2: (2,) arrays - triangle vertex coordinates
    
    Returns:
        inside: boolean - whether point is inside triangle
        bary_coords: (3,) array - barycentric coordinates if inside, else None
    """
    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    
    # Compute dot products
    dot00 = np.dot(v0v1, v0v1)
    dot01 = np.dot(v0v1, v0v2)
    dot02 = np.dot(v0v1, v0p)
    dot11 = np.dot(v0v2, v0v2)
    dot12 = np.dot(v0v2, v0p)
    
    # Compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # Check if point is in triangle
    if (u >= 0) and (v >= 0) and (u + v <= 1):
        w = 1 - u - v
        return True, np.array([w, u, v])
    
    return False, None


def _interpolate_to_polar_grid(estelas, lat0, lon0, ngc=360):
    """
    Interpolate unstructured grid data to polar grid using triangulation
    
    Args:
        estelas: ESTELA dataset with unstructured grid
        lat0: target point latitude
        lon0: target point longitude
        ngc: number of directional sectors (default 360)
    
    Returns:
        polar_F: DataArray with F on polar grid (distance, bearing, time)
    """
    logging.info("Interpolating to polar grid for G/L calculation...")
    
    # Get triangulation
    if not estelas.attrs.get('has_triangulation', 0):
        logging.error("No triangulation available for G/L calculation")
        return None
    
    tri = estelas['triangulation'].values
    lons = estelas.longitude.values
    lats = estelas.latitude.values
    
    # For each time step
    n_times = len(estelas.time)
    polar_grids = []
    
    for itime in range(n_times):
        time_key = estelas.time.values[itime]
        logging.info(f"  Processing time: {time_key}")
        
        # Get F values for this time
        F_values = estelas['F'].isel(time=itime).values
        
        # Create mask for nodes with valid data
        valid_mask = ~np.isnan(F_values)
        
        # Filter triangles - keep only those with all three vertices valid
        filtered_tri, _ = _filter_valid_triangles(tri, valid_mask)
        n_tri = filtered_tri.shape[1]
        logging.info(f"    Valid triangles: {n_tri} / {tri.shape[1]}")
        
        if n_tri == 0:
            logging.warning(f"    No valid triangles for {time_key}")
            polar_grids.append(None)
            continue
        
        # Compute centroids of valid triangles
        centroid_lons, centroid_lats = _compute_triangle_centroids(lons, lats, filtered_tri)
        
        # Build KDTree from centroids
        # Convert to Cartesian for better distance calculation
        centroid_x = np.cos(centroid_lats * D2R) * np.cos(centroid_lons * D2R)
        centroid_y = np.cos(centroid_lats * D2R) * np.sin(centroid_lons * D2R)
        centroid_z = np.sin(centroid_lats * D2R)
        centroid_xyz = np.column_stack([centroid_x, centroid_y, centroid_z])
        
        tree = KDTree(centroid_xyz)
        
        # Create polar grid
        gc = great_circles(lat0, lon0, ngc=ngc)
        polar_lons = gc.longitude.values
        polar_lats = gc.latitude.values
        n_dist = len(gc.distance)
        n_bear = len(gc.bearing)
        
        # Initialize output grid
        polar_F = np.full((n_dist, n_bear), np.nan)
        
        # For each point in polar grid
        for idist in range(n_dist):
            for ibear in range(n_bear):
                plon = polar_lons[idist, ibear]
                plat = polar_lats[idist, ibear]
                
                # Convert to Cartesian
                px = np.cos(plat * D2R) * np.cos(plon * D2R)
                py = np.cos(plat * D2R) * np.sin(plon * D2R)
                pz = np.sin(plat * D2R)
                p_xyz = np.array([px, py, pz])
                
                # Find 3 nearest centroids
                distances, indices = tree.query(p_xyz, k=min(3, n_tri))
                
                # Try each of the 3 nearest triangles
                found = False
                for idx in indices:
                    # Get triangle vertices
                    tri_idx = filtered_tri[:, idx]
                    v0 = np.array([lons[tri_idx[0]], lats[tri_idx[0]]])
                    v1 = np.array([lons[tri_idx[1]], lats[tri_idx[1]]])
                    v2 = np.array([lons[tri_idx[2]], lats[tri_idx[2]]])
                    p = np.array([plon, plat])
                    
                    # Check if point is in triangle
                    inside, bary_coords = _point_in_triangle(p, v0, v1, v2)
                    
                    if inside:
                        # Interpolate F using barycentric coordinates
                        f0 = F_values[tri_idx[0]]
                        f1 = F_values[tri_idx[1]]
                        f2 = F_values[tri_idx[2]]
                        
                        polar_F[idist, ibear] = (
                            bary_coords[0] * f0 + 
                            bary_coords[1] * f1 + 
                            bary_coords[2] * f2
                        )
                        found = True
                        break
        
        # Create DataArray for this time
        polar_F_da = xr.DataArray(
            data=polar_F,
            dims=['distance', 'bearing'],
            coords={
                'distance': gc.distance,
                'bearing': gc.bearing,
                'longitude': (('distance', 'bearing'), polar_lons),
                'latitude': (('distance', 'bearing'), polar_lats),
            }
        )
        
        polar_grids.append(polar_F_da)
    
    # Combine all times
    valid_grids = [g for g in polar_grids if g is not None]
    if len(valid_grids) == 0:
        logging.error("No valid polar grids generated")
        return None
    
    # Stack along time dimension
    polar_F_all = xr.concat(valid_grids, dim=estelas.time)
    
    logging.info("Polar grid interpolation completed")
    return polar_F_all


def plot_unstructured(estelas, groupers=None, gainloss=False, proj=None, 
                      set_global=False, cmap=None, figsize=None, outdir=None, 
                      scatter_size=1):
    """
    Plot ESTELA maps for unstructured grid
    
    Args:
        estelas: ESTELA dataset
        groupers: Time groupings
        gainloss: Whether to plot gain/loss maps
        proj: Cartopy projection
        set_global: Set global extent
        cmap: Colormap
        figsize: Figure size
        outdir: Output directory
        scatter_size: Size of scatter points
    """
    if figsize is None:
        figsize = [25, 10]

    # Handle both scalar and array lat0/lon0 (in case of concatenated datasets)
    lat0_data = estelas.lat0.values
    lon0_data = estelas.lon0.values

    if np.isscalar(lat0_data):
        lat0 = float(lat0_data)
        lon0 = float(lon0_data)
    else:
        # If array, take the first value (should all be the same for same target point)
        lat0 = float(lat0_data.flat[0])
        lon0 = float(lon0_data.flat[0])
    
    gc = great_circles(lat0, lon0, ngc=16)
    c1day = dict(levels=np.linspace(1, 30, 30), colors='grey', linewidths=0.5)
    c3day = dict(levels=np.linspace(3, 30, 10), colors='black', linewidths=1.0)
    
    if proj is None:
        proj = ccrs.PlateCarree(central_longitude=lon0)
    
    if cmap is None:
        cmap = 'seismic' if gainloss else 'inferno'
    
    figs = []
    groupers = get_groupers(groupers)
    
    for grouper in groupers:
        if grouper == "time.season":
            time = ["DJF", "MAM", "JJA", "SON"]
        elif grouper == "time.month":
            time = [f"m{m:02d}" for m in range(1, 13)]
        else:
            time = [grouper]
        
        # Filter available times
        available_times = [t for t in time if t in estelas.time.values]
        if not available_times:
            continue
        
        ds = estelas.sel(time=available_times)
        
        # Handle gain/loss plotting
        if gainloss:
            if not estelas.attrs.get('has_triangulation', 0):
                logging.warning("Cannot compute G/L without triangulation, skipping")
                continue
            
            # Interpolate to polar grid
            polar_F = _interpolate_to_polar_grid(ds, lat0, lon0, ngc=360)
            
            if polar_F is None:
                logging.warning("Polar grid interpolation failed, skipping G/L plot")
                continue
            
            # Compute gain/loss
            dist_midpoints = (polar_F.distance.values[1:] + polar_F.distance.values[:-1]) / 2
            cosd = np.cos(polar_F.distance * D2R)
            
            # Surface area of each cell
            ngc = len(polar_F.bearing)
            S = 4 * np.pi * 6371**2 / ngc * abs(cosd.diff("distance")) / 2  # km**2
            
            # Calculate gradient along distance
            incF = (polar_F.diff("distance") / S).assign_coords(distance=dist_midpoints)
            
            plot_data = incF
            plot_data.attrs['standard_name'] = '${\\Delta}F$'
            plot_data.attrs['units'] = '$\\frac{kW}{m\\circ{km^2}}$'
        else:
            plot_data = ds['F']
            plot_data.attrs['units'] = '$360\\times\\frac{kW}{m\\circ}$'
        
        logging.info(f"Plotting estelas for time={available_times}")
        
        # Create figure
        nplots = len(available_times)
        if nplots == 1:
            fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': proj})
            axes = [ax]
        else:
            ncols = 2 if nplots <= 4 else 3
            nrows = int(np.ceil(nplots / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                                    subplot_kw={'projection': proj})
            axes = axes.flatten()
        
        # Store the last plot object for colorbar
        last_plot_obj = None
        
        for iax, (ax, t) in enumerate(zip(axes[:nplots], available_times)):
            extent = ax.get_extent()
            
            if gainloss:
                # Plot G/L as contourf on polar grid
                plot_t = plot_data.sel(time=t)
                lons = plot_t.longitude.values
                lats = plot_t.latitude.values
                values = plot_t.values
                
                clim = float(np.nanquantile(np.abs(values), 0.95))
                
                p = ax.contourf(
                    lons, lats, np.clip(values, -clim, clim),
                    transform=ccrs.PlateCarree(),
                    levels=15,
                    cmap=cmap
                )
                last_plot_obj = p
            else:
                # Plot F as scatter
                F_t = plot_data.sel(time=t)
                lons = F_t.longitude.values
                lats = F_t.latitude.values
                values = F_t.values
                
                # Remove NaN values for plotting
                valid = ~np.isnan(values)
                
                scatter = ax.scatter(lons[valid], lats[valid], c=values[valid],
                                   s=scatter_size, cmap=cmap, transform=ccrs.PlateCarree(),
                                   vmin=np.nanpercentile(values, 5),
                                   vmax=np.nanpercentile(values, 95))
                last_plot_obj = scatter
            
            # Plot travel time contours (if available and not G/L)
            if not gainloss:
                ttime = ds.traveltime.sel(time=t)
                # Note: Contours for unstructured grids would require interpolation
                # Skipping for now as it's complex
            
            # Add great circles
            if nplots == 1:
                ax.plot(gc.longitude, gc.latitude, '.r', markersize=0.5, 
                       transform=ccrs.PlateCarree())
            
            # Set extent and add features
            if set_global:
                ax.set_global()
            else:
                ax.set_extent(extent, crs=proj)
            
            ax.coastlines()
            ax.stock_img()
            ax.plot(lon0, lat0, 'ok', transform=ccrs.PlateCarree())
            ax.set_title(t)
        
        # Hide extra axes
        for ax in axes[nplots:]:
            ax.set_visible(False)
        
        # Add colorbar below all subplots for multi-plot figures
        if nplots > 1 and last_plot_obj is not None:
            # Adjust layout to make room for colorbar
            fig.subplots_adjust(bottom=0.15)
            
            # Create colorbar axis below all subplots
            cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])  # [left, bottom, width, height]
            cbar = fig.colorbar(last_plot_obj, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(plot_data.attrs['units'], fontsize=12)
        elif nplots == 1 and last_plot_obj is not None:
            # For single plot, add colorbar as before
            cbar = plt.colorbar(last_plot_obj, ax=axes[0], orientation='horizontal', 
                              pad=0.05, shrink=0.8)
            cbar.set_label(plot_data.attrs['units'])
        
        fig.suptitle(f"[{ds.start_time} - {ds.end_time}]")
        figs.append(fig)
        
        if outdir is not None:
            maptype = "gainloss" if gainloss else "base"
            fname = os.path.join(outdir, f"estela_unstructured_{grouper}_{maptype}.png")
            logging.info(f"Saving {fname}")
            fig.savefig(fname, dpi=150, bbox_inches='tight')
    
    return figs


def great_circles(lat1, lon1, ngc=16):
    """Calculate great circles"""
    lat1_r = float(lat1) * D2R
    lon1_r = float(lon1) * D2R
    dist_r = xr.DataArray(dims="distance", data=np.linspace(0.5, 179.5, 180) * D2R)
    brng_r = xr.DataArray(dims="bearing", data=np.linspace(0, 360, ngc + 1)[:-1] * D2R)
    
    sin_lat1 = np.sin(lat1_r)
    cos_lat1 = np.cos(lat1_r)
    sin_dR = np.sin(dist_r)
    cos_dR = np.cos(dist_r)
    
    lat2 = np.arcsin(sin_lat1 * cos_dR + cos_lat1 * sin_dR * np.cos(brng_r))
    lon2 = lon1_r + np.arctan2(np.sin(brng_r) * sin_dR * cos_lat1, 
                                cos_dR - sin_lat1 * np.sin(lat2))
    gc = xr.Dataset({
        "latitude": lat2 / D2R, 
        "longitude": (lon2 / D2R % 360).transpose()
    })
    gc["distance"] = dist_r / D2R
    gc["bearing"] = brng_r / D2R
    return gc


def dist_and_bearing(lat1, lat2, lon1, lon2):
    """Calculate distances and bearings from one point to others"""
    lat1_r = lat1 * D2R
    lat2_r = lat2 * D2R
    latdif_r = (lat2 - lat1) * D2R
    londif_r = (lon2 - lon1) * D2R
    
    a = np.sin(latdif_r / 2) * np.sin(latdif_r / 2) + np.cos(lat1_r) * np.cos(
        lat2_r
    ) * np.sin(londif_r / 2) * np.sin(londif_r / 2)
    a = np.clip(a, 0.0, 1.0)
    degdist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) / D2R
    
    y = np.sin(londif_r) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(londif_r)
    brng = (np.arctan2(y, x) / D2R) % 360
    return (degdist, brng)


def geographic_mask(lat0, lon0, dists, bearings):
    """Check the great circles points to find points not blocked by land"""
    
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
        for idist in range(len(dists) - 1):
            d1 = dists[idist]
            d2 = dists[idist + 1]
            i1 = ibearings[idist]
            i2 = ibearings[idist + 1]
            for i in circ_closed_range(i1, i2):
                if i2 == i1:
                    d = d1
                else:
                    d = d1 + (d2 - d1) / (i2 - i1) * (i - i1)
                dmax[i] = min(dmax[i], d)
        return dmax
    
    dmax = 179.99 * np.ones(360)
    shpfilename = shapereader.natural_earth(
        resolution="110m", category="physical", name="coastline"
    )
    coastlines = shapereader.Reader(shpfilename).records()
    
    for c in coastlines:
        lonland, latland = map(np.array, zip(*c.geometry.coords))
        distland, bearingland = dist_and_bearing(lat0, latland, lon0, lonland)
        ibearingland = np.trunc(bearingland % 360).astype(int)
        dmax = update_dmax(dmax, distland, ibearingland)
    
    vland = dists < dmax[np.trunc(bearings % 360).astype(int)]
    return vland


def get_groupers(groupers):
    """Get default groupers if the input is None"""
    if groupers is None:
        groupers = ["ALL", "time.season", "time.month"]
    return groupers
