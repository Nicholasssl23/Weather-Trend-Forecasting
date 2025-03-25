import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from rapidfuzz import process, fuzz
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Data Loading and Basic Cleaning
path = r"C:\Users\nicho\PycharmProjects\WeatherForecastProject\GlobalWeatherRepository.csv"
df = pd.read_csv(path, parse_dates=["last_updated"])

# Convert date column and display summary
df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
print(df.describe(include="all"))
print(df.isna().sum())

# Check duplicates
duplicates = df[df.duplicated()]
if duplicates.empty:
    print("No duplicate rows found.")
else:
    print("Duplicate rows found.")

duplicate_names = df.columns[df.columns.duplicated()]
if len(duplicate_names) == 0:
    print("No duplicate column names")
else:
    print("Duplicate column names:", duplicate_names)

# Drop empty cells
df = df.dropna()

# Format column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Fuzzy Matching to Clean Location Names
def normalize_name(name):
    return name.strip().lower()


unique_names = df['location_name'].unique()
canonical_mapping = {}
canonical_list = []
for name in unique_names:
    norm_name = normalize_name(name)
    if canonical_list:
        best_match = process.extractOne(norm_name, canonical_list, scorer=fuzz.ratio)
        if best_match and best_match[1] > 90:
            canonical_mapping[name] = best_match[0]
        else:
            canonical_mapping[name] = norm_name
            canonical_list.append(norm_name)
    else:
        canonical_mapping[name] = norm_name
        canonical_list.append(norm_name)

# Create cleaned location names column
df['location_name_clean'] = df['location_name'].map(canonical_mapping)

# Cap wind speeds to a plausible maximum (100 kmph)
df.loc[df['wind_kph'] > 100, 'wind_kph'] = 100

# Grouping by Cleaned Location Names
df_location = df.groupby('location_name_clean').agg({
    'temperature_celsius': 'mean',
    'humidity': 'mean',
    'precip_mm': 'mean',
    'wind_kph': 'mean'
}).reset_index()

# Save a copy
df_location.to_csv("CleanedLocationData.csv", index=False)

# Scaling and Clustering
features_to_scale = ['temperature_celsius', 'humidity', 'precip_mm', 'wind_kph']
scaler = StandardScaler()
df_location_scaled = scaler.fit_transform(df_location[features_to_scale])

# Use an Elbow Plot to determine optimal k
sse = []
k_range = range(2, 10)
for k_val in k_range:
    kmeans_temp = KMeans(n_clusters=k_val, random_state=42)
    kmeans_temp.fit(df_location_scaled)
    sse.append(kmeans_temp.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors")
plt.title("Elbow Plot for K-Means Clustering")
plt.show()
# If the elbow appears around 5, then k=5 is justified

# Clustering with k=5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(df_location_scaled)
df_location['cluster'] = cluster_labels

# Map clusters to descriptive names based on cluster centers
# labels based on cluster centers
cluster_names = {
    0: "Tropical",  # high humidity, high precip
    1: "Cool-Dry",  # low temperature, low precip
    2: "Hot-Dry",  # high temp, very low humidity/precip
    3: "Temperate-Humid",  # moderate temp, high humidity
    4: "Hot-Humid"  # high temp, high humidity but less rain than tropical
}
df_location['cluster_label'] = df_location['cluster'].map(cluster_names)
print(df_location[['location_name_clean', 'cluster', 'cluster_label']].head(10))

# Print cluster centers (for interpretation)
cluster_centers_scaled = kmeans.cluster_centers_
cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
print("Cluster centers (original scale):")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: {center}")

# 2D PCA Visualization
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(df_location_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=df_location['cluster'], cmap='rainbow', alpha=0.7)
plt.xlabel("PC1 (Temperature-Precip Variation)")
plt.ylabel("PC2 (Humidity Variation)")
plt.title("Locations Clustered by Climate (PCA)")
plt.colorbar(label='Cluster')
plt.show()

# 3D PCA Visualization
pca_3d = PCA(n_components=3)
pca_coords_3d = pca_3d.fit_transform(df_location_scaled)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_coords_3d[:, 0], pca_coords_3d[:, 1], pca_coords_3d[:, 2],
                     c=df_location['cluster'], cmap='rainbow', alpha=0.7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA: Locations Clustered by Climate")
fig.colorbar(scatter, ax=ax, label='Cluster')
plt.show()

# Forecasting temperature for a single location (Kabul)
# First, extract the time series for location from the main dataset:
df_kabul = df[df['location_name_clean'] == 'kabul'].copy()
df_kabul.sort_values('last_updated', inplace=True)
df_kabul.set_index('last_updated', inplace=True)
ts_temp = df_kabul['temperature_celsius'].resample('D').mean()  # Daily average

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(ts_temp.index, ts_temp, marker='o', linestyle='-')
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Daily Average Temperature for Kabul")
plt.show()

# Assuming a seasonal period of 365/12 ~ 30 days if using daily data
model = SARIMAX(ts_temp, order=(1, 1, 1), seasonal_order=(1, 1, 1, 30), enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit(disp=False)
print(results.summary())

# Forecast the next 30 days
forecast = results.get_forecast(steps=30)
forecast_ci = forecast.conf_int()

plt.figure(figsize=(10, 5))
plt.plot(ts_temp.index, ts_temp, label='Observed')
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, color='red', label='Forecast')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Forecast for Kabul (Next 30 Days)")
plt.legend()
plt.show()

# Boxplots by Month
# Extract month information from last_updated column (before setting index)
df['month'] = df['last_updated'].dt.month
# Set last_updated as index for time series visualizations
df.set_index('last_updated', inplace=True)
df['month'] = df.index.month
plt.figure(figsize=(10, 5))
sns.boxplot(x='month', y='temperature_celsius', data=df)
plt.title("Temperature Distribution by Month (All Locations Combined)")
plt.show()
