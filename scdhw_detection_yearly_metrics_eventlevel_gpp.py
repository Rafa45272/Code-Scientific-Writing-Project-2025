import numpy as np
import xarray as xr
from scipy.stats import percentileofscore

def detect_compound_events_percentile(sm: xr.DataArray, st: xr.DataArray, window=1, sm_q=0.1, st_q=0.9):
    """
    Detect compound dry-hot events and return a dataset with event mask,
    soil moisture percentile, and soil temperature percentile.
    """
    assert sm.shape == st.shape, "Input DataArrays must have the same shape"
    assert 'time' in sm.dims and 'lat' in sm.dims and 'lon' in sm.dims, "Expected (time, lat, lon) dims"

    step_index = (
        sm['time'].to_index()
        .to_series()
        .groupby(sm['time'].dt.year.values)
        .cumcount()
        .to_numpy()
    )
    step_of_year = xr.DataArray(step_index, dims='time', coords={'time': sm['time']})
    sm = sm.assign_coords(step_of_year=('time', step_of_year.data))
    st = st.assign_coords(step_of_year=('time', step_of_year.data))

    n_steps = int(sm['step_of_year'].max().values) + 1
    n_time = sm.sizes['time']

    sm_thresh = xr.full_like(sm, np.nan)
    st_thresh = xr.full_like(st, np.nan)
    sm_percentile = xr.full_like(sm, np.nan)
    st_percentile = xr.full_like(st, np.nan)

    for t_idx in range(n_time):
        current_step = sm['step_of_year'][t_idx].values
        window_steps = [(current_step + i) % n_steps for i in range(-window, window + 1)]
        mask = sm['step_of_year'].isin(window_steps)
        sm_window = sm.where(mask, drop=True)
        st_window = st.where(mask, drop=True)
        sm_thresh[t_idx] = sm_window.quantile(sm_q, dim='time', skipna=True)
        st_thresh[t_idx] = st_window.quantile(st_q, dim='time', skipna=True)

        for i in range(sm.sizes['lat']):
            for j in range(sm.sizes['lon']):
                if np.isnan(sm[t_idx, i, j]) or np.isnan(st[t_idx, i, j]):
                    continue
                # Soil moisture percentile
                sm_vals = sm_window[:, i, j].values
                sm_val = sm[t_idx, i, j].values
                if np.isfinite(sm_val) and np.any(np.isfinite(sm_vals)):
                    sm_percentile[t_idx, i, j] = percentileofscore(sm_vals[~np.isnan(sm_vals)], sm_val, kind='rank') / 100.0
                # Soil temperature percentile
                st_vals = st_window[:, i, j].values
                st_val = st[t_idx, i, j].values
                if np.isfinite(st_val) and np.any(np.isfinite(st_vals)):
                    st_percentile[t_idx, i, j] = percentileofscore(st_vals[~np.isnan(st_vals)], st_val, kind='rank') / 100.0

    compound = (sm < sm_thresh) & (st > st_thresh)

    # Print stats
    total_events = int(compound.sum().values)
    total_timesteps = compound.sizes['time']
    n_cells = compound.sizes['lat'] * compound.sizes['lon']
    print(f"\n✅ Compound event detection complete!")
    print(f"→ Total compound events found: {total_events}")
    print(f"→ Average events per timestep: {total_events / total_timesteps:.2f}")
    print(f"→ Average events per grid cell: {total_events / n_cells:.2f}")

    return xr.Dataset({
        'compound': compound,
        'sm_percentile': sm_percentile,
        'st_percentile': st_percentile
    })

def calculate_compound_event_metrics_yearly(compound_ds, time_coord='time'):
    """
    Calculate duration, peak intensity, severity, and frequency for compound events, yearly.
    Returns: xr.Dataset with dims (year, lat, lon) and metrics as DataArrays.
    """
    times = compound_ds[time_coord].values
    years = pd.to_datetime(times).year
    year_list = np.unique(years)
    lat_vals = compound_ds.lat.values
    lon_vals = compound_ds.lon.values

    # Prepare output arrays
    shape = (len(year_list), len(lat_vals), len(lon_vals))
    duration_mean = np.full(shape, np.nan)
    duration_max = np.full(shape, np.nan)
    peak_intensity_mean = np.full(shape, np.nan)
    peak_intensity_max = np.full(shape, np.nan)
    severity_mean = np.full(shape, np.nan)
    severity_max = np.full(shape, np.nan)
    frequency_per_year = np.full(shape, np.nan)

    for yi, year in enumerate(year_list):
        year_mask = (years == year)
        for i, lat_val in enumerate(lat_vals):
            for j, lon_val in enumerate(lon_vals):
                mask = compound_ds['compound'][year_mask, i, j].values
                sm_p = compound_ds['sm_percentile'][year_mask, i, j].values
                st_p = compound_ds['st_percentile'][year_mask, i, j].values

                # Find contiguous events
                event_id = (mask.astype(int) * (np.diff(np.concatenate(([0], mask.astype(int)))) != 0).cumsum())
                event_id[~mask] = 0
                unique_events = np.unique(event_id[event_id > 0])

                durations = []
                peak_intensities = []
                severities = []

                for eid in unique_events:
                    idx = np.where(event_id == eid)[0]
                    if len(idx) == 0:
                        continue
                    duration = len(idx)
                    st_dev = st_p[idx] - 0.9
                    sm_dev = 0.1 - sm_p[idx]
                    peak_intensity = np.nanmax(np.abs(st_dev) + np.abs(sm_dev))
                    severity = np.nansum(np.abs(st_dev) + np.abs(sm_dev))

                    durations.append(duration)
                    peak_intensities.append(peak_intensity)
                    severities.append(severity)

                # Fill output arrays
                duration_mean[yi, i, j] = np.mean(durations) if durations else 0
                duration_max[yi, i, j] = np.max(durations) if durations else 0
                peak_intensity_mean[yi, i, j] = np.mean(peak_intensities) if peak_intensities else 0
                peak_intensity_max[yi, i, j] = np.max(peak_intensities) if peak_intensities else 0
                severity_mean[yi, i, j] = np.mean(severities) if severities else 0
                severity_max[yi, i, j] = np.max(severities) if severities else 0
                frequency_per_year[yi, i, j] = len(durations)

    ds = xr.Dataset(
        {
            'duration_mean': (('year', 'lat', 'lon'), duration_mean),
            'duration_max': (('year', 'lat', 'lon'), duration_max),
            'peak_intensity_mean': (('year', 'lat', 'lon'), peak_intensity_mean),
            'peak_intensity_max': (('year', 'lat', 'lon'), peak_intensity_max),
            'severity_mean': (('year', 'lat', 'lon'), severity_mean),
            'severity_max': (('year', 'lat', 'lon'), severity_max),
            'frequency_per_year': (('year', 'lat', 'lon'), frequency_per_year),
        },
        coords={
            'year': year_list,
            'lat': lat_vals,
            'lon': lon_vals
        }
    )
    return ds


import pandas as pd
from scipy.ndimage import label

def extract_event_summaries(compound_ds, gpp, baseline='outside'):
    """
    Extract event-level summaries for each contiguous compound event in each grid cell.
    Returns a DataFrame with one row per event, including GPP anomaly
    (anomaly is computed relative to the same time index across all years).
    """

    compound = compound_ds['compound'].values
    sm_percentile = compound_ds['sm_percentile'].values
    st_percentile = compound_ds['st_percentile'].values
    times = compound_ds['time'].values
    lats = compound_ds['lat'].values
    lons = compound_ds['lon'].values

    # Compute step-of-year for each time index
    step_of_year = pd.Series(pd.to_datetime(times)).groupby(pd.to_datetime(times).year).cumcount().values

    # Compute seasonal climatology: mean GPP for each step-of-year, lat, lon
    gpp_steps = xr.DataArray(step_of_year, dims='time', coords={'time': times})
    gpp_with_step = gpp.assign_coords(step_of_year=('time', gpp_steps.data))
    gpp_clim = gpp_with_step.groupby('step_of_year').mean(dim='time')

    event_rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            mask_1d = compound[:, i, j]
            labeled, n_events = label(mask_1d)
            for event_id in range(1, n_events + 1):
                idx = np.where(labeled == event_id)[0]
                if len(idx) == 0:
                    continue
                start_idx, end_idx = idx[0], idx[-1]
                start_time = times[start_idx]
                end_time = times[end_idx]
                duration = end_idx - start_idx + 1

                sm_p = sm_percentile[idx, i, j]
                st_p = st_percentile[idx, i, j]
                st_dev = st_p - 0.9
                sm_dev = 0.1 - sm_p
                intensity = np.nanmax(np.abs(st_dev) + np.abs(sm_dev))
                severity = np.nansum(np.abs(st_dev) + np.abs(sm_dev))

                # GPP anomaly relative to seasonal climatology
                gpp_event = gpp[:, i, j].values[idx]
                steps_idx = step_of_year[idx]
                gpp_clim_event = gpp_clim[:, i, j].values[steps_idx]
                gpp_anomaly = np.nanmean(gpp_event - gpp_clim_event)
                gpp_mean = np.nanmean(gpp_event)
                gpp_base = np.nanmean(gpp_clim_event)

                event_rows.append({
                    'lat': lat,
                    'lon': lon,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'severity': severity,
                    'intensity': intensity,
                    'gpp_mean': gpp_mean,
                    'gpp_baseline': gpp_base,
                    'gpp_anomaly': gpp_anomaly,
                    'event_indices': idx
                })

    event_df = pd.DataFrame(event_rows)
    return event_df
