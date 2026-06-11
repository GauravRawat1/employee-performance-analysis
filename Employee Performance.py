import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Data/employee_data.csv')

# Dataset Overview
print ("\t----- Employee performance Analysis -----\t\n")
print("overview of the dataset:\n", data.head())
print("Dataset Shape:\n", data.shape)
print("Data Types:\n", data.dtypes)
print("Basic Statistics:\n", data.describe())

# Data Cleaning
# Check for missing values
print("\nData Cleaning:-\n")
print("Checking for missing value:\n", data.isnull().sum())
if data.isnull().sum().any():
    print("Handling missing values...\n")
    data.dropna(inplace=True)
    print("Missing values after handling:\n", data.isnull().sum())
else:
    print("No missing values found.")

#Checking for Duplicates
print("\nChecking for duplicates:\n", data.duplicated().sum())
if data.duplicated().sum() > 0:
    print("Removing duplicates...\n")
    data.drop_duplicates(inplace=True)
    print("Duplicates after handling:\n", data.duplicated().sum())
else:
    print("No duplicates found.")

# Feature Engineering
data["Performance_Category"] = data["Performance_Score"].apply(
    lambda x:
    "Excellent" if x >= 90
    else "Good" if x >= 80
    else "Average" if x >= 70
    else "Poor"
)
data["Salary_Category"] = data["Salary"].apply(
    lambda x:
    "High" if x > 65000
    else "Medium" if x > 50000
    else "Low"
)

# Data Analysis
print("\nData Analysis:-\n")
print("Average Salary:\n",data["Salary"].mean())
print("Highest Salary:\n",data["Salary"].max())
print("Top Performer:\n",data.loc[data["Performance_Score"].idxmax()]["Name"])
print("Lowest Performer:\n",data.loc[data["Performance_Score"].idxmin()]["Name"])
print("Department-wise Performance:\n", data.groupby("Department")["Performance_Score"].mean())
print("Department-wise Average Salary:\n", data.groupby("Department")["Salary"].mean())

# Visualization
# Department-wise Performance
dept_perf = data.groupby("Department")["Performance_Score"].mean()
dept_perf.plot(kind="bar", color=["blue", "orange", "green", "red"])
plt.title("Department Wise Performance")
plt.xlabel("Department")
plt.ylabel("Average Performance Score")
plt.xticks(rotation=0)
plt.show()

# Salary Distribution
plt.hist(data["Salary"], bins=10, color="skyblue", edgecolor="black")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

#Employee Distribution
dept_count = data["Department"].value_counts()
plt.pie(dept_count, labels=dept_count.index, autopct="%1.1f%%", startangle=140)
plt.title("Employee by Department")
plt.show()

#Attendance vs Performance
plt.scatter(data["Attendance"], data["Performance_Score"], color="purple")
plt.title("Attendance vs Performance")
plt.xlabel("Attendance")
plt.ylabel("Performance Score")
plt.show()