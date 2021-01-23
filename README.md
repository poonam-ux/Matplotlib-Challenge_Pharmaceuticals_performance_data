# Pymaceuticals' Drug Treatments Results

## Background
As a senior data analyst at the company Pymaceuticals, you’ve been asked to generate all of the tables and figures needed for the technical report of the study. The executive team also has asked for a top-level summary of the study results. You are given an access to the complete data from their most recent animal study where 249 mice identified with SCC tumor growth were treated through a variety of drug regimens. Over the course of 45 days, tumor development was observed and measured. The purpose was to compare the performance of Pymaceuticals’ drug of interest, Capomulin, versus the other treatment regimens.

## Objectives
* Clean the dataset.
* Generate summary statistics table consisting of the mean, median, variance, standard deviation, and SEM of the tumor volume for each drug regimen.
* Generate a bar plot using both, Pandas’s DataFrame.plot() and Matplotlib’s pyplot.
* Generate a pie plot using both Pandas’s DataFrame.plot() and Matplotlib’s pyplot.
* Calculate the final tumor volume of each mouse across four of the most promising treatment regimens: Capomulin, Ramicane, Infubinol, and Ceftamin. Calculate the quartiles and IQR and quantitatively determine if there are any potential outliers across all four treatment regimens.
* Using Matplotlib, generate a box and whisker plot of the final tumor volume for all four treatment regimens and highlight any potential outliers in the plot by changing their color and style.
* Select a mouse that was treated with Capomulin and generate a line plot of tumor volume vs. time point for that mouse.
* Generate a scatter plot of mouse weight versus average tumor volume for the Capomulin treatment regimen.
* Calculate the correlation coefficient and linear regression model between mouse weight and average tumor volume for the Capomulin treatment. Plot the linear regression model.

## Observations and Insights
* The bar graph shows that the drug regimen Capomulin was tested on the maximum number of mice (230), followed by Ramicane (228).
* The male and female mice population was very close- 125 male, and 124 female mice in the study. This shows a balanced gender population.
* From statistics table and box plot, we can infer that Capomulin and Ramicane treatments were more effective than the other drugs. 
* Study of Capomulin drug on randomly selected mouse (ID="s185") shows positive results where the total tumor volume declines over the period of the study.
* The correlation between mouse weight and average tumor volume is 0.84. This shows that when the mouse weight increases the average tumor volume also increases. 
* The regression analysis shows how much the average tumor volume (dependent variable) will change when weight of the mouse changes (independent variable), with positive slope of the regression line. It also shows that the 70% of the model fits the data. This will help to predict the future studies.

## Script

```python
### Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.stats import linregress
from sklearn import datasets

# Study data files
mouse_metadata_path = "Resources/Mouse_metadata.csv"
study_results_path = "Resources/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)

# Combine the data into a single dataset
merged_data = pd.merge(mouse_metadata, study_results, how="left", on="Mouse ID")

# Display the data table for preview
merged_data
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/merged_dataframe.png)

```python
# Checking the number of mice.
number_of_mice = len(merged_data["Mouse ID"].unique())
print(f"Number of mice in the study: {number_of_mice}")
```
Number of mice in the study: 249

```python
# Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 
duplicate_miceID = merged_data.loc[merged_data.duplicated(subset=["Mouse ID", "Timepoint",]),"Mouse ID"].unique()
print(f"Duplicate Mice ID: {duplicate_miceID}")
```
Duplicate Mice ID: 'g989'

```python
# Optional: Get all the data for the duplicate mouse ID.
duplicate_mice_rows = merged_data[merged_data.duplicated(["Mouse ID", "Timepoint",])]
duplicate_mice_rows
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/mice_with_duplicate_rows%20.png)

```python
# Create a clean DataFrame by dropping the duplicate mouse by its ID.
combined_data = merged_data.drop_duplicates(subset=["Mouse ID", "Timepoint"])
combined_data
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/clean_data.png)

```python
# Checking the number of mice in the clean DataFrame
mice_count = len(combined_data["Mouse ID"].unique())
print(f"Number of mice in clean dataframe: {mice_count}")
```
Number of mice in clean dataframe: 249

## Summary Statistics
```python
# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen
# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: mean, median, variance, standard deviation, and SEM of the tumor volume. 
regimen_mean = combined_data.groupby("Drug Regimen").mean()["Tumor Volume (mm3)"]
regimen_mean

regimen_median = combined_data.groupby("Drug Regimen").median()["Tumor Volume (mm3)"]
regimen_median

regimen_variance = combined_data.groupby("Drug Regimen").var()["Tumor Volume (mm3)"]
regimen_variance

regimen_stdev = combined_data.groupby("Drug Regimen").std()["Tumor Volume (mm3)"]
regimen_stdev

regimen_sem = combined_data.groupby("Drug Regimen").sem()["Tumor Volume (mm3)"]
regimen_sem

# Assemble the resulting series into a single summary dataframe.
summary_stats1 = pd.DataFrame({"Mean": regimen_mean, "Median":regimen_median, "Variance":regimen_variance, 
                               "Standard Deviation": regimen_stdev, "SEM": regimen_sem})
summary_stats1
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/statistical_table1.png)

```python
# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen using the aggregation method, produce the same summary statistics in a single line
summary_stats2 = combined_data.groupby("Drug Regimen").agg(["mean", "median", "var", "std", "sem"])["Tumor Volume (mm3)"]
summary_stats2 = summary_stats2.rename(columns={"mean":"Mean", "median": "Median", "var": "Variance", 
                                                "std": "Standard Deviation", "sem": "SEM"})
summary_stats2
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/statistical_table1.png)

### Bar and Pie Charts
```python
# Generate a bar plot showing the total number of measurements taken on each drug regimen using pandas.
mice_per_drug = combined_data.groupby(["Drug Regimen"]).count()["Mouse ID"]

pandas_plot = mice_per_drug.plot(kind="bar", color='b', figsize=(15,10), fontsize = 14)
plt.title("Number of Mice per Treatment",fontsize = 20)
plt.xlabel("Drug Regimen",fontsize = 15)
plt.ylabel("Number of Mice Treated",fontsize = 15)

plt.savefig("./Images/pandas_bar_mice_per_treat.png", bbox_inches = "tight")
plt.show()
plt.tight_layout()

# mice_count
mice_per_drug
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/pandas_bar_mice_per_treat.png)

```python
# Generate a bar plot showing the total number of measurements taken on each drug regimen using pyplot.
# Get y_axis values by creating a list of number of mice for each drug regimen
treatment_count = (combined_data.groupby(["Drug Regimen"])["Mouse ID"].count()).tolist()
treatment_count

x_axis = np.arange(len(mice_per_drug))
tick_locations = [value+0.4 for value in x_axis]
fig1, ax1 = plt.subplots(figsize=(15,10))
plt.xticks(tick_locations, mice_per_drug.index.values, rotation="vertical")
plt.bar(x_axis, treatment_count, color='b', alpha=1, align="edge")
plt.title("Number of Mice per Treatment",fontsize = 20)
plt.xlabel("Drug Regimen",fontsize = 15)
plt.ylabel("Number of Mice Treated",fontsize = 15)
plt.savefig("./Images/pyplot_bar_mice_per_treat.png", bbox_inches = "tight")

treatment_count
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/pyplot_bar_mice_per_treat.png)

```python
# Generate a pie plot showing the distribution of female versus male mice using pandas
# Total count of mice based on gender
gender_rows = combined_data.loc[:, ["Sex", "Mouse ID"]]
gender_rows = gender_rows.drop_duplicates()
gender_count = gender_rows["Sex"].value_counts()

# Create a dataframe for count and percentage of male and female mice
gender_data = pd.DataFrame({"Total Count": gender_count})

# Generate a pie plot showing the distribution of female versus male mice
colors = ["blue", "coral"]
explode = (0.1, 0)
pandas_pie = gender_data.plot(kind="pie", figsize=(12,8), colors=colors, y="Total Count", title="Male vs Female Mice Population",
                              startangle=315, explode=explode, shadow=True, autopct="%1.1f%%")
plt.axis("equal")
plt.savefig("./Images/pandas_pie_male_vs_female.png", bbox_inches = "tight")
plt.show()

gender_data
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/pandas_pie_male_vs_female.png)

```python
# Generate a pie plot showing the distribution of female versus male mice using pyplot
# Total count of mice based on gender
gender_rows = combined_data.loc[:, ["Sex", "Mouse ID"]]
gender_rows = gender_rows.drop_duplicates()
gender_count = gender_rows["Sex"].value_counts()

labels = ["Male", "Female"]
colors = ["blue", "coral"]
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(12, 8))
plt.title("Male vs Female Mice Population",fontsize = 16)
plt.ylabel("Total Count",fontsize = 14)
plt.pie(gender_count, labels=labels, colors=colors, startangle=315, explode=explode, shadow=True, autopct="%1.1f%%")

#Set equal axis
plt.axis("equal")

plt.savefig("./Images/pyplot_pie_male_vs_female.png", bbox_inches = "tight")
plt.show()

gender_data
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/pyplot_pie_male_vs_female.png)

## Quartiles, Outliers and Boxplots
```python
# Calculate the final tumor volume of each mouse across four of the treatment regimens:  
# Capomulin, Ramicane, Infubinol, and Ceftamin

# Start by getting the last (greatest) timepoint for each mouse
last_timepoint = combined_data.groupby(["Mouse ID"])["Timepoint"].max()
last_timepoint = last_timepoint.reset_index()

# Merge this group df with the original dataframe to get the tumor volume at the last timepoint
merged_data_lasttp = last_timepoint.merge(combined_data, on=["Mouse ID", "Timepoint"], how="left")
merged_data_lasttp
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/mice_last_timepoint.png)

```python
# Put treatments into a list for for loop (and later for plot labels)
treatments = ["Capomulin", "Ramicane", "Infubinol", "Ceftamin"]

# Create empty list to fill with tumor vol data (for plotting)
total_tumor_vol = []

# Calculate the IQR and quantitatively determine if there are any potential outliers. 
for drug in treatments:

    # Locate the rows which contain mice on each drug and get the tumor volumes
    tumor_vol = merged_data_lasttp.loc[merged_data_lasttp["Drug Regimen"] == drug, "Tumor Volume (mm3)"]
    
    # add subset 
    total_tumor_vol.append(tumor_vol)
    
    # Determine outliers using upper and lower bounds
    quartiles = tumor_vol.quantile([.25, .5, .75])
    lowerq = quartiles[.25]
    upperq = quartiles[.75]
    iqr = upperq - lowerq 
    
    lower_bound = lowerq - (1.5*iqr)
    upper_bound = upperq + (1.5*iqr)
    outliers = tumor_vol.loc[(tumor_vol < lower_bound) | (tumor_vol > upper_bound)]
    print(f"For {drug}, the outliers are: {outliers}")
```
For Infubinol, the outliers are: 31    36.321346
```python
# Generate a box plot of the final tumor volume of each mouse across four regimens of interest
fig1, ax1 = plt.subplots(figsize=(12,8))
ax1.set_title('Tumor Volume for each Drug Regimen',fontsize =16)
ax1.set_ylabel('Final Tumor Volume (mm3)',fontsize = 14)
ax1.set_xlabel('Drug Regimen',fontsize = 14)
ax1.boxplot(total_tumor_vol, labels=treatments, widths = 0.4, patch_artist=True,vert=True)

plt.savefig("./Images/pyplot_boxplot_final_tumor_vol.png", bbox_inches = "tight")
plt.show()
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/pyplot_boxplot_final_tumor_vol.png)

## Line and Scatter Plots
```python
# Generate a line plot of tumor volume vs. time point for a mouse treated with Capomulin
capomulin_df = combined_data.loc[combined_data["Drug Regimen"] == "Capomulin",:]
capomulin_df

# Get data for a specific mouse (e.g. Mouse ID = "s185")
mouce_in_capomulin = capomulin_df.loc[capomulin_df["Mouse ID"] == "s185",:]
mouce_in_capomulin

# Specify plot parameters
x_axis = mouce_in_capomulin["Timepoint"]
tumor_volume = mouce_in_capomulin["Tumor Volume (mm3)"]

fig1, ax1 = plt.subplots(figsize=(12, 8))
plt.title("Capomulin treatmeant of Mouse ID 's185'",fontsize =16)
plt.plot(x_axis, tumor_volume,linewidth=2, markersize=12,marker="o",color="blue", label="mm3")
plt.xlabel("Timepoint (Days)",fontsize =14)
plt.ylabel("Tumor Volume (mm3)",fontsize =14)

plt.savefig("./Images/mouse_Capomulin_line_plot.png", bbox_inches = "tight")
plt.show()
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/mouse_Capomulin_line_plot.png)

```python
# Generate a scatter plot of average tumor volume vs. mouse weight for the Capomulin regimen
avg_capom_df = capomulin_df.groupby(['Mouse ID']).mean()
fig1, ax1 = plt.subplots(figsize=(12, 8))
marker_size=14
plt.scatter(avg_capom_df['Weight (g)'],avg_capom_df['Tumor Volume (mm3)'],s=90, color="blue")
plt.title("Mouse weight vs Average Tumor Volume",fontsize =20)
plt.xlabel("Weight (g)",fontsize =14)
plt.ylabel("Averag Tumor Volume (mm3)",fontsize =14)

plt.savefig("./Images/avg_tumor_vol_scatterplot.png", bbox_inches = "tight")
plt.show()

avg_capom_df.head()
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/Mouse%20weight%20vs%20Avg%20Tumor%20Volume.png)
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/avg_tumor_vol_scatterplot.png)

## Correlation and Regression
```python
# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen
correlation = round(st.pearsonr(avg_capom_df['Weight (g)'],avg_capom_df['Tumor Volume (mm3)'])[0],2)
print(f"The correlation between mouse weight and average tumor volume is: {correlation}")

fig1, ax1 = plt.subplots(figsize=(12, 8))
plt.scatter(avg_capom_df['Weight (g)'],avg_capom_df['Tumor Volume (mm3)'],s=90, color="blue")
plt.title("Correlation between Mouse weight & Average Tumor Volume",fontsize =20)
plt.xlabel("Weight (g)", fontsize = 14)
plt.ylabel("Averag Tumor Volume (mm3)", fontsize = 14)

# plt.savefig("./Images/corr_scatterplot.png", bbox_inches = "tight")
plt.show()
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/avg_tumor_vol_scatterplot.png)

```python
# Add the linear regression equation and line to the scatter plot
x_values = avg_capom_df['Weight (g)']
y_values = avg_capom_df['Tumor Volume (mm3)']

(slope, intercept, rvalue, pvalue, stderr) = linregress(x_values, y_values)
regress_values = x_values * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
print(line_eq)

fig1, ax1 = plt.subplots(figsize=(12, 8))
plt.scatter(x_values,y_values, s=90, color="blue")
plt.plot(x_values,regress_values,"r-")
plt.title("Linear Regression Equation", fontsize=20)
plt.xlabel("Mouse weight (g)", fontsize =14)
plt.ylabel("Average Tumor Volume (mm3)", fontsize=14)
plt.annotate(line_eq, xy=(20,40), xycoords="data",xytext=(0.2, 0.80), textcoords="axes fraction", fontsize=16,color="red")

plt.savefig("./Images/corr_avg_tumor_vol_and_weight_scatterplot.png", bbox_inches = "tight")
plt.show()
```
![](https://github.com/poonam-ux/Matplotlib-Challenge_Pharmaceuticals_performance_data/blob/main/Images/corr_avg_tumor_vol_and_weight_scatterplot.png)
