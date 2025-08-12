import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # Added for parallelization

print("Script started: SCDHW_analysis.py", flush=True)

def calculate_percentiles(data, window=5, baseline_start=2009, baseline_end=2018):
    """
    Calculate percentiles for each day of year using a 5-day window
    during the baseline period (2009-2018).
    """
    # Create a DOY index
    doy = data.time.dt.dayofyear
    
    # Initialize arrays for percentiles (separate arrays for 10th and 90th)
    p10 = np.zeros((366, *data.shape[1:]))
    p90 = np.zeros((366, *data.shape[1:]))
    
    # Calculate percentiles for each DOY
    for d in range(1, 367):
        # Get the 5-day window
        window_days = [(d-2+i) % 366 + 1 for i in range(5)]
        
        # Select data for the baseline period and window days
        baseline_data = data.sel(time=slice(f"{baseline_start}-01-01", f"{baseline_end}-12-31"))
        window_data = baseline_data.where(baseline_data.time.dt.dayofyear.isin(window_days))
        
        # Calculate percentiles
        percentiles = np.percentile(window_data.values, [10, 90], axis=0)
        p10[d-1] = percentiles[0]  # 10th percentile
        p90[d-1] = percentiles[1]  # 90th percentile
    
    return np.stack([p10, p90])  # Return as a stacked array with shape (2, 366, lat, lon)

def identify_extreme_events(data, percentiles, threshold_type='lower', min_consecutive_periods=2):
    """
    Identify extreme events based on percentiles and minimum duration.
    For 8-day data, we consider 1 period (8 days) as minimum duration.
    threshold_type: 'lower' for drought (10th percentile), 'upper' for heatwave (90th percentile)
    """
    # Get day of year for each time step
    doy = data.time.dt.dayofyear.values - 1  # Convert to 0-based index
    
    # Create binary mask for extreme conditions
    if threshold_type == 'lower':
        extreme_mask = data.values < percentiles[0, doy]  # 10th percentile
    else:
        extreme_mask = data.values > percentiles[1, doy]  # 90th percentile
    
    # Helper function for a single grid point
    def process_grid_point(lat_idx, lon_idx):
        ts = extreme_mask[:, lat_idx, lon_idx]
        event_duration = np.zeros(ts.shape, dtype=data.dtype)
        event_intensity = np.zeros(ts.shape, dtype=data.dtype)
        event_severity = np.zeros(ts.shape, dtype=data.dtype)
        event_start = None
        current_duration = 0
        for t in range(len(ts)):
            if ts[t]:
                if event_start is None:
                    event_start = t
                current_duration += 1
            else:
                if event_start is not None and current_duration >= min_consecutive_periods:
                    event_duration[event_start:event_start+current_duration] = current_duration * 8
                    if threshold_type == 'lower':
                        event_intensity[event_start:event_start+current_duration] = \
                            percentiles[0, doy[event_start:event_start+current_duration], lat_idx, lon_idx] - \
                            data.values[event_start:event_start+current_duration, lat_idx, lon_idx]
                    else:
                        event_intensity[event_start:event_start+current_duration] = \
                            data.values[event_start:event_start+current_duration, lat_idx, lon_idx] - \
                            percentiles[1, doy[event_start:event_start+current_duration], lat_idx, lon_idx]
                    event_severity[event_start:event_start+current_duration] = \
                        event_intensity[event_start:event_start+current_duration] * (current_duration * 8)
                event_start = None
                current_duration = 0
        # Handle case where event extends to end of time series
        if event_start is not None and current_duration >= min_consecutive_periods:
            event_duration[event_start:] = current_duration * 8
            if threshold_type == 'lower':
                event_intensity[event_start:] = \
                    percentiles[0, doy[event_start:], lat_idx, lon_idx] - \
                    data.values[event_start:, lat_idx, lon_idx]
            else:
                event_intensity[event_start:] = \
                    data.values[event_start:, lat_idx, lon_idx] - \
                    percentiles[1, doy[event_start:], lat_idx, lon_idx]
            event_severity[event_start:] = \
                event_intensity[event_start:] * (current_duration * 8)
        return event_duration, event_intensity, event_severity

    # Parallelize over all grid points
    shape = data.shape
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(process_grid_point)(lat_idx, lon_idx)
        for lat_idx in range(shape[1])
        for lon_idx in range(shape[2])
    )
    # Reconstruct the full arrays
    event_duration = np.zeros(shape, dtype=data.dtype)
    event_intensity = np.zeros(shape, dtype=data.dtype)
    event_severity = np.zeros(shape, dtype=data.dtype)
    idx = 0
    for lat_idx in range(shape[1]):
        for lon_idx in range(shape[2]):
            ed, ei, es = results[idx]
            event_duration[:, lat_idx, lon_idx] = ed
            event_intensity[:, lat_idx, lon_idx] = ei
            event_severity[:, lat_idx, lon_idx] = es
            idx += 1
    return event_duration, event_intensity, event_severity

def identify_scdhw_events(sm_data, st_data, sm_percentiles, st_percentiles, min_consecutive_periods=2):
    """
    Identify SCDHW events by combining soil moisture drought and soil heatwave events.
    For 8-day data, we consider 1 period (8 days) as minimum duration.
    """
    # Identify soil moisture droughts
    sm_duration, sm_intensity, sm_severity = identify_extreme_events(
        sm_data, sm_percentiles, threshold_type='lower', min_consecutive_periods=min_consecutive_periods
    )
    
    # Identify soil heatwaves
    st_duration, st_intensity, st_severity = identify_extreme_events(
        st_data, st_percentiles, threshold_type='upper', min_consecutive_periods=min_consecutive_periods
    )
    
    # Combine events to identify SCDHWs
    scdhw_mask = (sm_duration > 0) & (st_duration > 0)
    
    # Calculate SCDHW properties
    scdhw_duration = np.where(scdhw_mask, np.minimum(sm_duration, st_duration), 0)
    scdhw_intensity = np.where(scdhw_mask, np.maximum(sm_intensity, st_intensity), 0)
    scdhw_severity = np.where(scdhw_mask, scdhw_intensity * scdhw_duration, 0)
    
    return scdhw_duration, scdhw_intensity, scdhw_severity

def main():
    print("Loading soil moisture dataset...", flush=True)
    sm_filtered = xr.open_dataset("Computation/filtered_maps/soil_moisture_filtered.nc")
    print("Soil moisture dataset loaded.", flush=True)
    print("Loading soil temperature dataset...", flush=True)
    st_filtered = xr.open_dataset("Computation/filtered_maps/soil_temp_filtered.nc")
    print("Soil temperature dataset loaded.", flush=True)
    # Extract variables
    sm_var = list(sm_filtered.data_vars)[0]
    st_var = list(st_filtered.data_vars)[0]
    print(f"Variables extracted: {sm_var}, {st_var}", flush=True)
    # Filter for summer months (April to October)
    print("Applying summer mask...", flush=True)
    summer_mask = (sm_filtered.time.dt.month >= 4) & (sm_filtered.time.dt.month <= 10)
    sm_summer = sm_filtered[sm_var].where(summer_mask)
    st_summer = st_filtered[st_var].where(summer_mask)
    print("Summer mask applied.", flush=True)
    # Calculate percentiles for baseline period
    print("Calculating percentiles...", flush=True)
    sm_percentiles = calculate_percentiles(sm_summer)
    st_percentiles = calculate_percentiles(st_summer)
    print("Percentiles calculated.", flush=True)
    
    # Identify extreme events separately for debugging
    print("Identifying soil moisture droughts and soil heatwaves (separately for debugging)...")
    sm_drought_duration, sm_drought_intensity, sm_drought_severity = identify_extreme_events(
        sm_summer, sm_percentiles, threshold_type='lower', min_consecutive_periods=1
    )
    st_hw_duration, st_hw_intensity, st_hw_severity = identify_extreme_events(
        st_summer, st_percentiles, threshold_type='upper', min_consecutive_periods=1
    )
    
    # Debug: Print number of drought and heatwave events
    print(f"Total number of drought events (duration > 0): {np.sum(sm_drought_duration > 0)}")
    print(f"Total number of heatwave events (duration > 0): {np.sum(st_hw_duration > 0)}")
    
    # Plot and save maps of total drought and heatwave events per grid cell
    drought_occurrence = (sm_drought_duration > 0).sum(axis=0)
    heatwave_occurrence = (st_hw_duration > 0).sum(axis=0)
    plt.figure(figsize=(12, 5))
    plt.imshow(drought_occurrence, origin='lower', aspect='auto',
               extent=[float(sm_summer.lon.min()), float(sm_summer.lon.max()), float(sm_summer.lat.min()), float(sm_summer.lat.max())],
               cmap='Blues')
    plt.colorbar(label='Number of Drought Events')
    plt.title('Soil Moisture Drought Occurrence (April-Oct, 2009-2018)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig('drought_occurrence_map.png')
    plt.close()
    plt.figure(figsize=(12, 5))
    plt.imshow(heatwave_occurrence, origin='lower', aspect='auto',
               extent=[float(st_summer.lon.min()), float(st_summer.lon.max()), float(st_summer.lat.min()), float(st_summer.lat.max())],
               cmap='Oranges')
    plt.colorbar(label='Number of Heatwave Events')
    plt.title('Soil Temperature Heatwave Occurrence (April-Oct, 2009-2018)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig('heatwave_occurrence_map.png')
    plt.close()
    print("Saved drought and heatwave occurrence maps.")
    
    # Identify SCDHW events
    print("Identifying SCDHW events...")
    scdhw_duration, scdhw_intensity, scdhw_severity = identify_scdhw_events(
        sm_summer, st_summer, sm_percentiles, st_percentiles, min_consecutive_periods=1
    )
    
    # Create output dataset
    scdhw_ds = xr.Dataset(
        {
            'duration': (['time', 'lat', 'lon'], scdhw_duration),
            'intensity': (['time', 'lat', 'lon'], scdhw_intensity),
            'severity': (['time', 'lat', 'lon'], scdhw_severity)
        },
        coords={
            'time': sm_summer.time,
            'lat': sm_summer.lat,
            'lon': sm_summer.lon
        }
    )
    
    # Save results
    print("Saving results...")
    os.makedirs("scdhw_analysis", exist_ok=True)
    scdhw_ds.to_netcdf("scdhw_analysis/scdhw_events.nc")
    
    # Calculate and print summary statistics
    print("\nSCDHW Summary Statistics:")
    print(f"Total number of SCDHW events: {np.sum(scdhw_duration > 0)}")
    print(f"Mean duration: {np.mean(scdhw_duration[scdhw_duration > 0]):.2f} days")
    print(f"Mean intensity: {np.mean(scdhw_intensity[scdhw_intensity > 0]):.2f}")
    print(f"Mean severity: {np.mean(scdhw_severity[scdhw_severity > 0]):.2f}")
    
    # Print time range information
    print("\nTime Range Information:")
    print(f"Analysis period: {sm_summer.time.values[0]} to {sm_summer.time.values[-1]}")
    print(f"Number of 8-day periods: {len(sm_summer.time)}")
    print(f"Total number of days: {len(sm_summer.time) * 8}")

if __name__ == "__main__":
    main() 