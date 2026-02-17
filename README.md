# data-visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = "Sales Dataset.csv"
df = pd.read_csv(file_path)

# -----------------------------
# 2. Basic Cleaning
# -----------------------------
# Drop unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Remove rows with missing Date (if any)
df = df.dropna(subset=["Date"])

# Ensure numeric columns are numeric
numeric_cols = ["Age", "Quantity", "Price per Unit", "Total Amount"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing numeric values
df = df.dropna(subset=numeric_cols)

# -----------------------------
# 3. Dataset Overview
# -----------------------------
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Columns ---")
print(df.columns)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe())

# -----------------------------
# 4. Basic Insights
# -----------------------------
print("\n--- Gender Distribution ---")
print(df["Gender"].value_counts())

print("\n--- Product Category Distribution ---")
print(df["Product Category"].value_counts())

print("\n--- Total Revenue ---")
print(df["Total Amount"].sum())

print("\n--- Average Bill Amount ---")
print(df["Total Amount"].mean())

# -----------------------------
# 5. Category-wise Analysis
# -----------------------------
category_sales = df.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False)
print("\n--- Category Wise Revenue ---")
print(category_sales)

category_quantity = df.groupby("Product Category")["Quantity"].sum().sort_values(ascending=False)
print("\n--- Category Wise Quantity Sold ---")
print(category_quantity)

# -----------------------------
# 6. Gender-wise Analysis
# -----------------------------
gender_revenue = df.groupby("Gender")["Total Amount"].sum().sort_values(ascending=False)
print("\n--- Gender Wise Revenue ---")
print(gender_revenue)

gender_avg_bill = df.groupby("Gender")["Total Amount"].mean().sort_values(ascending=False)
print("\n--- Gender Wise Average Bill ---")
print(gender_avg_bill)

# -----------------------------
# 7. Age Group Analysis
# -----------------------------
bins = [18, 25, 35, 45, 55, 65]
labels = ["18-25", "26-35", "36-45", "46-55", "56-65"]
df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)

agegroup_revenue = df.groupby("Age Group")["Total Amount"].sum()
print("\n--- Age Group Wise Revenue ---")
print(agegroup_revenue)

# -----------------------------
# 8. Monthly Sales Trend
# -----------------------------
df["Month"] = df["Date"].dt.to_period("M").astype(str)
monthly_sales = df.groupby("Month")["Total Amount"].sum()

print("\n--- Monthly Revenue ---")
print(monthly_sales)

# -----------------------------
# 9. Save Cleaned Dataset
# -----------------------------
df.to_csv("cleaned_sales_dataset.csv", index=False)
print("\nCleaned dataset saved as cleaned_sales_dataset.csv")

# -----------------------------
# 10. Visualizations
# -----------------------------

# 10.1 Category Revenue Bar Chart
plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")
plt.title("Total Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 10.2 Gender Revenue Bar Chart
plt.figure(figsize=(6, 4))
gender_revenue.plot(kind="bar")
plt.title("Total Revenue by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Revenue")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 10.3 Age Distribution Histogram
plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=15)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 10.4 Monthly Sales Trend Line Chart
plt.figure(figsize=(12, 5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 10.5 Age Group Revenue Bar Chart
plt.figure(figsize=(8, 5))
agegroup_revenue.plot(kind="bar")
plt.title("Revenue by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Revenue")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nEDA Completed Successfully.")
