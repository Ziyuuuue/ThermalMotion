import pandas as pd
import numpy as np

# Recreate the dataset
data = pd.DataFrame({
    'Trial': ['1.1', '1.2', '2.1', '2.2', '4.1', '4.2', '5.1', '5.2', '6.1', '6.2', '6.3', '7.1', '7.2'],
    'k': [9.12E-24, 1.25E-23, 9.99E-24, 1.14E-23, 1.07E-23, 1.24E-23, 1.20E-23, 1.03E-23, 1.33E-23, 9.57E-24, 9.93E-24, 1.08E-23, 1.03E-23],
    'delta k': [9.67E-25, 1.33E-24, 1.06E-24, 1.21E-24, 1.13E-24, 1.32E-24, 1.27E-24, 1.09E-24, 1.41E-24, 1.02E-24, 1.05E-24, 1.14E-24, 1.09E-24],
    'D': [1.61E-01, 2.21E-01, 1.76E-01, 2.01E-01, 1.88E-01, 2.19E-01, 2.12E-01, 1.81E-01, 2.34E-01, 1.69E-01, 1.75E-01, 1.89E-01, 1.82E-01],
    'delta D': [1.25E-03, 1.18E-03, 1.31E-03, 1.36E-03, 1.07E-03, 1.37E-03, 1.19E-03, 1.44E-03, 1.32E-03, 1.07E-03, 1.03E-03, 1.25E-03, 1.15E-03],
    'm': [6.42E-01, 8.84E-01, 7.04E-01, 8.05E-01, 7.52E-01, 8.75E-01, 8.47E-01, 7.23E-01, 9.37E-01, 6.74E-01, 7.00E-01, 7.58E-01, 7.27E-01],
    'delta m': [4.99E-03, 4.72E-03, 5.23E-03, 5.43E-03, 4.29E-03, 5.46E-03, 4.75E-03, 5.77E-03, 5.28E-03, 4.28E-03, 4.10E-03, 5.02E-03, 4.61E-03],
    'b': [0.4755, -0.52975, -1.03943, 0.805, -0.52542, -0.14181, -0.59338, 2.87331, 0.29262, -0.37648, -0.3746, -0.45819, -0.39679],
    'delta b': [0.09891, 0.04336, 0.09841, 0.10468, 0.03271, 0.10468, 0.05349, 0.1317, 0.08002, 0.05137, 0.02818, 0.08402, 0.06482],
    'Chi-Square': [1.12E+02, 107.69106, 813.08005, 273.98222, 212.49334, 345.68877, 138.04635, 455.42092, 504.85937, 90.29801, 118.03541, 407.9023, 59.86977],
    'R-squared': [0.99159, 0.99418, 0.94234, 0.98908, 0.9852, 0.98045, 0.9925, 0.96246, 0.96161, 0.99071, 0.98973, 0.96757, 0.99497],
    'Reduced Chi Sqaure': [0.95836, 0.92000, 6.95000, 2.34173, 1.81618, 2.95460, 1.17988, 3.89000, 4.31504, 0.77178, 1.01000, 3.48634, 0.51171],
    'P Value': [6.10E-01, 7.20E-01, 0.00E+00, 1.29E-14, 1.63E-07, 0.00E+00, 8.95E-02, 0.00E+00, 0.00E+00, 9.68E-01, 4.56E-01, 0.00E+00, 1.00E+00]
})

# Weighted average function
def weighted_average(values, uncertainties):
    weights = 1 / np.array(uncertainties)**2
    weighted_avg = np.sum(values * weights) / np.sum(weights)
    weighted_uncertainty = np.sqrt(1 / np.sum(weights))
    return weighted_avg, weighted_uncertainty

# Step 1: Calculate Mean Reduced Chi-Square
mean_reduced_chi_square = np.mean(data['Reduced Chi Sqaure'])

# Calculate weighted averages and uncertainties
results = {}
for param in ['k', 'D', 'm', 'b']:
    results[param] = weighted_average(data[param], data[f'delta {param}'])

# Direct averages for Chi-Square, R-squared, and P Value
for param in ['Chi-Square', 'R-squared', 'P Value']:
    results[param] = (data[param].mean(), None)

# Print results
for key, value in results.items():
    print(f"{key}: Weighted Average = {value[0]}, Uncertainty = {value[1]}")

print(f"Mean Reduced Chi-Square: {mean_reduced_chi_square}")
