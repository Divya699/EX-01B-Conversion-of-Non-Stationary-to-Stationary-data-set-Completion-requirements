# EX-01B-Conversion-of-Non-Stationary-to-Stationary-data-set-Completion-requirements

## Date: 01/11/2026

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data

## Dataset Name: Canteen Shop Data
## Software Required: 
- Jupyter Notebook
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.

## PROGRAM:
## Developed By: DIVYA PRIYA S
## Register No: 212225240037

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Read dataset
data = pd.read_csv("C:\\Users\\acer\\Downloads\\canteen datasets\\canteen_shop_data.csv")

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Make Total numeric
data['Total'] = pd.to_numeric(data['Total'], errors='coerce')

# Remove missing values
data = data.dropna(subset=['Date', 'Total'])

# Combine all transactions for each date
data = data.groupby('Date')['Total'].sum().to_frame()

# Sort by date
data = data.sort_index()

# Regular Differencing
data['total_diff'] = data['Total'].diff()

# Seasonal Decomposition
result = seasonal_decompose(
    data['Total'],
    model='additive',
    period=7
)

# Seasonal Adjustment
data['total_sea_diff'] = result.resid

# Log Transformation
data['total_log'] = np.log(data['Total'])

# Log + Regular Differencing
data['total_log_diff'] = data['total_log'].diff()

# Create separate data for decomposition
log_diff_data = data['total_log_diff'].dropna()

# Seasonal decomposition of log difference
result2 = seasonal_decompose(
    log_diff_data,
    model='additive',
    period=7
)

# Store result directly
data['total_log_seasonal_diff'] = np.nan

data.loc[
    result2.resid.index,
    'total_log_seasonal_diff'
] = result2.resid


# ==========================================
# PLOTS
# ==========================================

plt.figure(figsize=(16, 16))

# 1. Original Data
plt.subplot(6, 1, 1)
plt.plot(data['Total'], label='Original')
plt.legend(loc='best')
plt.title('Original Data')
plt.xlabel('Date')
plt.ylabel('Total Sales')


# 2. Regular Differencing
plt.subplot(6, 1, 2)
plt.plot(data['total_diff'], label='Regular Difference')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('Date')
plt.ylabel('Differenced Total Sales')


# 3. Seasonal Adjustment
plt.subplot(6, 1, 3)
plt.plot(data['total_sea_diff'], label='Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')
plt.xlabel('Date')
plt.ylabel('Seasonally Adjusted Sales')


# 4. Log Transformation
plt.subplot(6, 1, 4)
plt.plot(data['total_log'], label='Log Transformation')
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Date')
plt.ylabel('Log(Total Sales)')


# 5. Log + Regular Differencing
plt.subplot(6, 1, 5)
plt.plot(
    data['total_log_diff'],
    label='Log Transformation and Regular Differencing'
)
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing')
plt.xlabel('Date')
plt.ylabel('Diff(Log(Total Sales))')


# 6. Log + Regular + Seasonal Differencing
plt.subplot(6, 1, 6)
plt.plot(
    data['total_log_seasonal_diff'],
    label='Log + Regular + Seasonal Differencing'
)
plt.legend(loc='best')
plt.title(
    'Log Transformation + Regular Differencing + Seasonal Differencing'
)
plt.xlabel('Date')
plt.ylabel('Seasonal Diff(Log(Total Sales))')


plt.tight_layout()
plt.show()


# Final line plot
data.plot(kind='line', figsize=(16, 8))
plt.title('Canteen Total Sales Time Series')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()
```
## OUTPUT 

ORIGINAL DATA:
<img width="1788" height="285" alt="image" src="https://github.com/user-attachments/assets/a96aa628-a163-4c13-8bd4-22bfe13c1c5b" />


REGULAR DIFFERENCING:
<img width="1695" height="262" alt="image" src="https://github.com/user-attachments/assets/9433b72f-7d7a-4e0d-9997-bab501eac224" />

SEASONAL ADJUSTMENT:
<img width="1654" height="264" alt="image" src="https://github.com/user-attachments/assets/1ec173ec-555b-4b14-9c38-4d179deeceec" />

LOG TRANSFORMATION:
<img width="1712" height="261" alt="image" src="https://github.com/user-attachments/assets/0c40c7a5-8377-4c4b-9b7a-3e975bae3aa1" />
LOG TRANSFORMATION AND REGULAR DIFFERENCING:
<img width="1798" height="287" alt="Screenshot 2026-11-01 150513" src="https://github.com/user-attachments/assets/a84a53f0-1786-484e-9954-8a9bed7abfa0" />


LOG TRANSFORMATION AND REGULAR DIFFERENCING AND SEASONAL ADJUSTMENT:

<img width="1815" height="290" alt="image" src="https://github.com/user-attachments/assets/e200974b-a413-434c-b951-1f6df912ce36" />

OVERALL VIEW:

<img width="1344" height="720" alt="image" src="https://github.com/user-attachments/assets/e6ceba33-f39e-4336-b74d-9bbb53005697" />


## RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on Canteen shop
data.
