#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 20:59:05 2025

@author: hannadelala
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw dataset
path = "/Users/hannadelala/Library/Mobile Documents/com~apple~CloudDocs/2-Study-Materials/Centennial-College/Semester-6/COMP309-Data-Warehouse/Bicycle_theft_Group_Group4_section_001#COMP309Project"
filename = "Bicycle_Thefts_Open_Data.csv"
fullpath = os.path.join(path, filename)
data_group4 = pd.read_csv(fullpath)

# View column names
print("Columns:")
print(data_group4.columns.values)
print()

# View data types
print("Column Types:")
print(data_group4.dtypes)
print()

# Shape of dataset
print("Shape (rows, columns):")
print(data_group4.shape)
print()

# Summary statistics on RAW data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print("Summary Statistics:")
print(data_group4.describe(include='all'))
print()

# Check value ranges for numeric columns
print("Numeric Column Ranges:")
for col in data_group4.select_dtypes(include=[np.number]).columns:
    print(f"{col}: min={data_group4[col].min()}, max={data_group4[col].max()}")
print()

# Missing data check
print("Missing Values (raw):")
print(data_group4.isnull().sum())
print()

# =========================
#       DATA CLEANING
# =========================

data_clean = data_group4.copy()

# Keep only years from 2013 to 2025
data_clean = data_clean[(data_clean['OCC_YEAR'] >= 2013) & (data_clean['OCC_YEAR'] <= 2025)]

# Convert date columns
# data_clean['OCC_DATE'] = pd.to_datetime(data_clean['OCC_DATE'], errors='coerce')
# data_clean['REPORT_DATE'] = pd.to_datetime(data_clean['REPORT_DATE'], errors='coerce')

# Replace negative BIKE_COST
bike_cost_median = data_clean['BIKE_COST'].median()
data_clean.loc[data_clean['BIKE_COST'] < 0, 'BIKE_COST'] = bike_cost_median

# Replace unrealistic outliers above 99th percentile with median
bike_cost_99 = data_clean['BIKE_COST'].quantile(0.99)
data_clean.loc[data_clean['BIKE_COST'] > bike_cost_99, 'BIKE_COST'] = bike_cost_median

# Replace impossible BIKE_SPEED (>120 km/h)
bike_speed_median = data_clean['BIKE_SPEED'].median()
data_clean.loc[data_clean['BIKE_SPEED'] > 120, 'BIKE_SPEED'] = bike_speed_median

# Replace invalid coordinates (0.0)
for col in ['LONG_WGS84', 'LAT_WGS84']:
    data_clean.loc[data_clean[col] == 0, col] = np.nan

# Fill missing categorical values
categorical_cols = ['BIKE_MAKE', 'BIKE_MODEL', 'BIKE_COLOUR']
for col in categorical_cols:
    data_clean[col] = data_clean[col].fillna("Unknown")

# Fill missing numeric values using median
data_clean['BIKE_COST'] = data_clean['BIKE_COST'].fillna(bike_cost_median)
data_clean['BIKE_SPEED'] = data_clean['BIKE_SPEED'].fillna(bike_speed_median)

print("Missing Values After Cleaning:")
print(data_clean.isnull().sum())
print()

# Drop unused cols
drop_cols = [
    "EVENT_UNIQUE_ID", "OCC_DATE", "REPORT_DATE",
    "NEIGHBOURHOOD_158", "NEIGHBOURHOOD_140",
    "x", "y", "LONG_WGS84", "LAT_WGS84", "HOOD_158", "HOOD_140"
]

data_clean = data_clean.drop(columns=drop_cols, errors='ignore')


# Save cleaned data to new file
cleaned_filename = "Bicycle_Thefts_Open_Data_cleaned.csv"
cleaned_path = os.path.join(path, cleaned_filename)
data_clean.to_csv(cleaned_path, index=False)
print("Cleaned data saved")
print()

# =========================
#       PLOTTING
# =========================

plot_df = pd.read_csv(cleaned_path)
print("Loaded cleaned data for plotting. Shape:", plot_df.shape)
print()

# Correlation matrix
numeric_df = plot_df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
print("Correlation Matrix:")
print(corr)
print()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Distribution of theft outcomes
if 'STATUS' in plot_df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=plot_df, x='STATUS')
    plt.title("Bike Theft Status Distribution")
    plt.show()

# Histogram of bike cost
if 'BIKE_COST' in plot_df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(plot_df['BIKE_COST'], kde=True)
    plt.title("Distribution of Bike Cost")
    plt.show()

# Theft count by year
if 'OCC_YEAR' in plot_df.columns:
    plt.figure(figsize=(7, 4))
    plot_df['OCC_YEAR'].value_counts().sort_index().plot(kind='bar')
    plt.title("Number of Bicycle Thefts Per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.show()

# Theft by neighbourhood (top 10)
if 'NEIGHBOURHOOD_158' in plot_df.columns:
    plt.figure(figsize=(10, 6))
    plot_df['NEIGHBOURHOOD_158'].value_counts().head(10).plot(kind='bar')
    plt.title("Top 10 Neighbourhoods for Bicycle Theft")
    plt.ylabel("Count")
    plt.show()
    
# Bicycle Theft Count by Hour of Day
data_group4["OCC_HOUR"] = pd.to_numeric(data_group4["OCC_HOUR"], errors="coerce")

plt.figure(figsize=(10,5))
sns.countplot(x="OCC_HOUR", data=data_group4, color="skyblue")
plt.title("Bicycle Theft Count by Hour of Day")
plt.xlabel("Hour (0–23)")
plt.ylabel("Number of Thefts")
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
