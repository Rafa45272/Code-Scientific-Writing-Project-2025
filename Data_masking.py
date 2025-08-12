import xarray as xr
import os
import fsspec
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_data_coverage(data, title, filename, month=8):
    """Plot data coverage on a map showing presence/absence of data for a specific month."""
    plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Select data for the specified month (default: August)
    data_month = data.sel(time=data.time.dt.month == month)
    
    # Convert to binary mask (1 for data, 0 for NaN) and average across time
    binary_data = ~np.isnan(data_month.values)
    avg_coverage = np.mean(binary_data, axis=0)  # Average across time dimension
    
    # Create DataArray with averaged coverage
    data_avg = xr.DataArray(
        avg_coverage,
        coords={'lat': data.lat, 'lon': data.lon},
        dims=['lat', 'lon']
    )
    
    # Plot average coverage
    data_avg.plot(ax=ax, transform=ccrs.PlateCarree(), 
                 cmap='binary', add_colorbar=True,
                 cbar_kwargs={'label': 'Data Coverage (%)'})
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    
    plt.title(f"{title} (Month: {month})")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_statistics(data, name):
    """Calculate and print statistics about the data."""
    total_points = data.size
    valid_points = np.sum(~np.isnan(data.values))
    nan_percentage = (total_points - valid_points) / total_points * 100
    
    print(f"\n{name} Statistics:")
    print(f"Total points: {total_points:,}")
    print(f"Valid points: {valid_points:,}")
    print(f"NaN percentage: {nan_percentage:.2f}%")
    
    return valid_points, nan_percentage

# === Configuration ===
start_year = 2009
end_year = 2019  # This will be adjusted to 2019 in the file list
sm_dir = "/home/sc.uni-leipzig.de/ek84usuf/2nd_semester/sc_writing/st_era5_8d/regridded_sm"
st_dir = "/home/sc.uni-leipzig.de/ek84usuf/2nd_semester/sc_writing/st_era5_8d/regridded_st"
cube_path = '/software/databases/rsc4earth/EarthSystemDataCube/v2.1.1/esdc-8d-0.25deg-184x90x90-2.1.1.zarr/'

# Create output directory for plots
os.makedirs("masking_plots", exist_ok=True)
os.makedirs("filtered_maps", exist_ok=True)

# Get list of available years
available_years = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(sm_dir) if f.startswith('era5_sm_regridded_')])
start_year = min(available_years)
end_year = max(available_years) + 1

sm_files = [f"{sm_dir}/era5_sm_regridded_{year}.nc" for year in range(start_year, end_year)]
st_files = [f"{st_dir}/era5_st_regridded_{year}.nc" for year in range(start_year, end_year)]

# === Load and Concatenate ===
print("Loading soil moisture datasets...")
sm_datasets = [xr.open_dataset(os.path.expanduser(f)) for f in sm_files if os.path.exists(os.path.expanduser(f))]
if not sm_datasets:
    raise FileNotFoundError("❌ No soil moisture datasets found. Check your file paths.")
soil_moisture = xr.concat(sm_datasets, dim="time")

print("Loading soil temperature datasets...")
st_datasets = [xr.open_dataset(os.path.expanduser(f)) for f in st_files if os.path.exists(os.path.expanduser(f))]
if not st_datasets:
    raise FileNotFoundError("❌ No soil temperature datasets found. Check your file paths.")
soil_temperature = xr.concat(st_datasets, dim="time")

# Print time information
print("\nTime Information:")
print("Soil Moisture time range:", soil_moisture.time.values[0], "to", soil_moisture.time.values[-1])
print("Number of timesteps:", len(soil_moisture.time))
print("Soil Temperature time range:", soil_temperature.time.values[0], "to", soil_temperature.time.values[-1])
print("Number of timesteps:", len(soil_temperature.time))

# Get variable names
sm_var = list(soil_moisture.data_vars)[0]
st_var = list(soil_temperature.data_vars)[0]

# === Initial Statistics and Plotting ===
print("\n=== Initial Data Coverage ===")
# Calculate initial statistics
sm_valid_init, sm_nan_init = calculate_statistics(soil_moisture[sm_var], "Initial Soil Moisture")
st_valid_init, st_nan_init = calculate_statistics(soil_temperature[st_var], "Initial Soil Temperature")

# === Latitude > 50N Filter ===
print("\n=== After Latitude Filtering (>50°N) ===")
soil_moisture_nh = soil_moisture.sel(lat=soil_moisture.lat[soil_moisture.lat > 50])
soil_temperature_nh = soil_temperature.sel(lat=soil_temperature.lat[soil_temperature.lat > 50])

print("\nTime Information After Latitude Filter:")
print("Soil Moisture time range:", soil_moisture_nh.time.values[0], "to", soil_moisture_nh.time.values[-1])
print("Number of timesteps:", len(soil_moisture_nh.time))
print("Soil Temperature time range:", soil_temperature_nh.time.values[0], "to", soil_temperature_nh.time.values[-1])
print("Number of timesteps:", len(soil_temperature_nh.time))

# Calculate statistics after latitude filtering
sm_valid_nh, sm_nan_nh = calculate_statistics(soil_moisture_nh[sm_var], "Northern Hemisphere Soil Moisture")
st_valid_nh, st_nan_nh = calculate_statistics(soil_temperature_nh[st_var], "Northern Hemisphere Soil Temperature")

# === Summer Mask (April to October) ===
print("\n=== After Summer Masking (April to October) ===")
summer_mask = (soil_moisture_nh.time.dt.month >= 4) & (soil_moisture_nh.time.dt.month <= 10)

# Apply summer mask
soil_moisture_summer = soil_moisture_nh.where(summer_mask)
soil_temperature_summer = soil_temperature_nh.where(summer_mask)

print("\nTime Information After Summer Mask:")
print("Summer months data time range:", soil_moisture_summer.time.values[0], "to", soil_moisture_summer.time.values[-1])
print("Number of timesteps in summer:", len(soil_moisture_summer.time))

# === Temperature Mask: Unfrozen conditions ===
print("\n=== After Temperature Masking (Unfrozen Conditions) ===")
unfrozen_mask = soil_temperature_summer[st_var] > 272.15  # -1°C

print("\nTime Information After Temperature Mask:")
print("Unfrozen mask time range:", unfrozen_mask.time.values[0], "to", unfrozen_mask.time.values[-1])
print("Number of timesteps:", len(unfrozen_mask.time))

# === Apply mask to soil moisture and soil temperature ===
sm_filtered = soil_moisture_summer[sm_var].where(unfrozen_mask)
st_filtered = soil_temperature_summer[st_var].where(unfrozen_mask)

print("\nTime Information After Final Filtering:")
print("Filtered Soil Moisture time range:", sm_filtered.time.values[0], "to", sm_filtered.time.values[-1])
print("Number of timesteps:", len(sm_filtered.time))
print("Filtered Soil Temperature time range:", st_filtered.time.values[0], "to", st_filtered.time.values[-1])
print("Number of timesteps:", len(st_filtered.time))

# === Plotting ===
print("\n=== Generating Plots (August Data) ===")
# Plot initial data coverage
plot_data_coverage(soil_moisture[sm_var], 
                  "Initial Soil Moisture Coverage", 
                  "masking_plots/initial_sm_coverage.png")
plot_data_coverage(soil_temperature[st_var], 
                  "Initial Soil Temperature Coverage", 
                  "masking_plots/initial_st_coverage.png")

# Plot Northern Hemisphere coverage
plot_data_coverage(soil_moisture_nh[sm_var], 
                  "Soil Moisture Coverage >50°N", 
                  "masking_plots/sm_nh_coverage.png")
plot_data_coverage(soil_temperature_nh[st_var], 
                  "Soil Temperature Coverage >50°N", 
                  "masking_plots/st_nh_coverage.png")

# Plot summer mask coverage
plot_data_coverage(soil_moisture_summer[sm_var], 
                  "Summer Soil Moisture Coverage", 
                  "masking_plots/sm_summer_coverage.png")
plot_data_coverage(soil_temperature_summer[st_var], 
                  "Summer Soil Temperature Coverage", 
                  "masking_plots/st_summer_coverage.png")

# Plot unfrozen mask
plot_data_coverage(unfrozen_mask, 
                  "Unfrozen Conditions Mask", 
                  "masking_plots/unfrozen_mask.png")

# Plot final filtered data
plot_data_coverage(sm_filtered, 
                  "Final Filtered Soil Moisture", 
                  "masking_plots/final_sm_filtered.png")
plot_data_coverage(st_filtered, 
                  "Final Filtered Soil Temperature", 
                  "masking_plots/final_st_filtered.png")

# === Winter Plots (November to March) ===
print("\n=== Generating Winter Plots ===")
# Create winter mask
winter_mask = (soil_moisture_nh.time.dt.month <= 3) | (soil_moisture_nh.time.dt.month >= 11)

# Get winter data
soil_moisture_winter = soil_moisture_nh.where(winter_mask)
soil_temperature_winter = soil_temperature_nh.where(winter_mask)

# Apply unfrozen mask to winter data
winter_unfrozen_mask = soil_temperature_winter[st_var] > 272.15  # -1°C
sm_winter_filtered = soil_moisture_winter[sm_var].where(winter_unfrozen_mask)
st_winter_filtered = soil_temperature_winter[st_var].where(winter_unfrozen_mask)

# Plot winter data (using January as representative month)
plot_data_coverage(sm_winter_filtered, 
                  "Winter Soil Moisture Coverage (Unfrozen)", 
                  "masking_plots/sm_winter_coverage.png",
                  month=1)  # January

plot_data_coverage(st_winter_filtered, 
                  "Winter Soil Temperature Coverage (Unfrozen)", 
                  "masking_plots/st_winter_coverage.png",
                  month=1)  # January

# Print winter statistics
print("\nWinter Data Statistics:")
winter_sm_valid = np.sum(~np.isnan(sm_winter_filtered.values))
winter_st_valid = np.sum(~np.isnan(st_winter_filtered.values))
print(f"Winter Soil Moisture valid points: {winter_sm_valid:,}")
print(f"Winter Soil Temperature valid points: {winter_st_valid:,}")

# === Load and Mask GPP ===
print("\n=== GPP Processing ===")
ds_cube = xr.open_zarr(fsspec.get_mapper(cube_path), consolidated=True)
gpp = ds_cube['gross_primary_productivity']

# Debug GPP data
print("\nGPP Data Information:")
print("GPP dimensions:", gpp.dims)
print("GPP shape:", gpp.shape)
print("GPP coordinates:", gpp.coords)
print("GPP time range:", gpp.time.values[0], "to", gpp.time.values[-1])
print("GPP latitude range:", gpp.lat.values[0], "to", gpp.lat.values[-1])
print("GPP longitude range:", gpp.lon.values[0], "to", gpp.lon.values[-1])

# Check for any non-zero values
non_zero = np.sum(gpp.values > 0)
print(f"Number of non-zero values: {non_zero:,}")
print(f"Percentage of non-zero values: {(non_zero/gpp.size)*100:.2f}%")

# Ensure GPP is on the same grid as soil data before masking
print("\nResampling GPP to match soil data grid...")
gpp = gpp.interp_like(soil_moisture_nh[sm_var], method='nearest')
print("GPP shape after resampling:", gpp.shape)

# Apply summer mask to GPP
gpp_summer = gpp.where(summer_mask)
gpp_filtered = gpp_summer.where(unfrozen_mask)

# Plot GPP coverage
plot_data_coverage(gpp, 
                  "Initial GPP Coverage", 
                  "masking_plots/initial_gpp_coverage.png")
plot_data_coverage(gpp_summer, 
                  "Summer GPP Coverage", 
                  "masking_plots/gpp_summer_coverage.png")
plot_data_coverage(gpp_filtered, 
                  "Final Filtered GPP", 
                  "masking_plots/final_gpp_filtered.png")

# Calculate statistics
gpp_valid_init, gpp_nan_init = calculate_statistics(gpp, "Initial GPP")
gpp_valid_summer, gpp_nan_summer = calculate_statistics(gpp_summer, "Summer GPP")
gpp_valid_final, gpp_nan_final = calculate_statistics(gpp_filtered, "Final Filtered GPP")

# === Save filtered datasets ===
print("\n💾 Saving filtered datasets...")
os.makedirs("filtered_maps", exist_ok=True)
sm_filtered.to_netcdf("filtered_maps/soil_moisture_filtered.nc")
st_filtered.to_netcdf("filtered_maps/soil_temp_filtered.nc")
gpp_filtered.to_netcdf("filtered_maps/gpp_filtered_masked.nc")

# === Summary Statistics ===
print("\n=== Summary of Data Filtering ===")
print("\nSoil Moisture:")
print(f"Initial valid points: {sm_valid_init:,} ({100-sm_nan_init:.2f}% coverage)")
print(f"After latitude filter: {sm_valid_nh:,} ({100-sm_nan_nh:.2f}% coverage)")
print(f"After summer filter: {np.sum(~np.isnan(soil_moisture_summer[sm_var].values)):,} points")
print(f"Final valid points: {np.sum(~np.isnan(sm_filtered.values)):,} points")

print("\nSoil Temperature:")
print(f"Initial valid points: {st_valid_init:,} ({100-st_nan_init:.2f}% coverage)")
print(f"After latitude filter: {st_valid_nh:,} ({100-st_nan_nh:.2f}% coverage)")
print(f"After summer filter: {np.sum(~np.isnan(soil_temperature_summer[st_var].values)):,} points")
print(f"Final valid points: {np.sum(~np.isnan(st_filtered.values)):,} points")

print("\nGPP:")
print(f"Initial valid points: {gpp_valid_init:,} ({100-gpp_nan_init:.2f}% coverage)")
print(f"Summer valid points: {gpp_valid_summer:,} ({100-gpp_nan_summer:.2f}% coverage)")
print(f"Final valid points: {gpp_valid_final:,} ({100-gpp_nan_final:.2f}% coverage)")

print("\n✅ Processing complete. Check the masking_plots directory for visualization of each filtering step.")
