import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots directory
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load data
df = pd.read_csv('rainfall_india_1901_2015.csv')

print("--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Basic Statistics
print("\n--- Annual Rainfall Stats ---")
print(df['ANNUAL'].describe())

# Check for highest and lowest rainfall years
mean_annual = df.groupby('YEAR')['ANNUAL'].mean()
print(f"\nYear with Highest Mean Rainfall: {mean_annual.idxmax()} ({mean_annual.max():.2f} mm)")
print(f"Year with Lowest Mean Rainfall: {mean_annual.idxmin()} ({mean_annual.min():.2f} mm)")

# --- Visualizations ---

# 1. Annual Rainfall Trend (All India)
plt.figure(figsize=(12, 6))
plt.plot(mean_annual.index, mean_annual.values, color='b', linestyle='-', marker='o', markersize=2)
plt.title('Average Annual Rainfall in India (1901-2015)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/india_annual_trend.png')
print("\nGenerated: plots/india_annual_trend.png")
plt.close()

# 2. Rainfall by Subdivision
subdivision_annual = df.groupby('DIVISION')['ANNUAL'].mean().sort_values()
plt.figure(figsize=(10, 12))  # Taller plot for horizontal bars
plt.barh(subdivision_annual.index, subdivision_annual.values, color='skyblue')
plt.title('Average Annual Rainfall by Subdivision')
plt.xlabel('Rainfall (mm)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/india_subdivision_rainfall.png')
print("Generated: plots/india_subdivision_rainfall.png")
plt.close()

# 3. Seasonal Rainfall
seasonal_cols = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
seasonal_means = df[seasonal_cols].mean()

plt.figure(figsize=(10, 6))
plt.bar(seasonal_means.index, seasonal_means.values, color=['green', 'orange', 'blue', 'brown'])
plt.title('Average Seasonal Rainfall Distribution')
plt.ylabel('Rainfall (mm)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(seasonal_means.values):
    plt.text(i, v + 10, f"{v:.1f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/india_seasonal_rainfall.png')
print("Generated: plots/india_seasonal_rainfall.png")
plt.close()

# Save summary to file
with open('analysis_summary.txt', 'w') as f:
    f.write("--- India Rainfall Analysis Summary ---\n\n")
    f.write(f"Data Range: {df['YEAR'].min()} - {df['YEAR'].max()}\n")
    f.write(f"Total Subdivisions: {df['DIVISION'].nunique()}\n\n")
    f.write(f"Average Annual Rainfall (All India): {df['ANNUAL'].mean():.2f} mm\n")
    f.write(f"Highest Rainfall Year: {mean_annual.idxmax()} ({mean_annual.max():.2f} mm)\n")
    f.write(f"Lowest Rainfall Year: {mean_annual.idxmin()} ({mean_annual.min():.2f} mm)\n\n")
    f.write("--- Seasonal Averages ---\n")
    for season, val in seasonal_means.items():
        f.write(f"{season}: {val:.2f} mm\n")
    f.write("\n--- Top 3 Wettest Subdivisions ---\n")
    f.write(subdivision_annual.tail(3).sort_values(ascending=False).to_string())
    f.write("\n\n--- Top 3 Driest Subdivisions ---\n")
    f.write(subdivision_annual.head(3).to_string())

print("\nSummary saved to analysis_summary.txt")
