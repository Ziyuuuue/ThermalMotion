import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress, chi2

# -----------------------------
# 1. Constants and Conversion Factors
# -----------------------------
conversion_factor = 0.12048  # microns per pixel
temperature = 296.5  # Temperature in Kelvin
viscosity = 0.94e-3  # Viscosity of water in Pa*s
radius = 0.5 * 1.9e-6  # Radius of bead in meters (1.9 µm diameter)
k_accepted = 1.38e-23  # Accepted Boltzmann constant in J/K

# Uncertainties
position_uncertainty = 0.1  # µm
time_uncertainty = 0.03  # s
viscosity_uncertainty = 0.05e-3  # Pa.s
radius_uncertainty = 0.1e-6 / 2  # meters
T_uncertainty = 0.5  # K


# -----------------------------
# 2. Load and Process Data
# -----------------------------
# Load data
data_path = r"C:\Users\yueyu\OneDrive\ES_2024\PHY293\LAB2\data\7.2.txt"
data = pd.read_csv(data_path, delimiter='\t', skiprows=2, names=['X (pixels)', 'Y (pixels)'])

# Convert pixel positions to micrometers
x_microns = data['X (pixels)'].astype(float) * conversion_factor  # µm
y_microns = data['Y (pixels)'].astype(float) * conversion_factor  # µm

# Calculate displacement at each step
displacement = np.sqrt(np.diff(x_microns)**2 + np.diff(y_microns)**2)  # µm

# Time intervals (assuming 0.5s between frames)
time_intervals = np.arange(1, len(displacement) + 1) * 0.5  # seconds


# -----------------------------
# 3. Calculate Cumulative Mean Squared Displacement (MSD)
# -----------------------------

# Mean Squared Displacement (cumulative sum)
mean_squared_displacement = np.cumsum(displacement**2)  # µm²

# -----------------------------
# 4. Uncertainty Calculations
# -----------------------------


# Displacement uncertainty (assuming independent x and y uncertainties)
displacement_uncertainty = np.sqrt(2) * position_uncertainty  # µm

# Calculate individual MSD uncertainties
individual_msd_uncertainty = 2 * displacement * displacement_uncertainty  # µm²

# Calculate cumulative uncertainty using root-sum-square propagation
msd_uncertainty = np.sqrt(np.cumsum(individual_msd_uncertainty**2))  # µm²

# -----------------------------
# 5. Define Linear Fit Function with Uncertainties
# -----------------------------
def linear_fit_with_uncertainty(x, y, y_uncertainty):
    """
    Perform a weighted linear fit using curve_fit.
    
    Parameters:
    - x: Independent variable data (Time)
    - y: Dependent variable data (MSD)
    - y_uncertainty: Uncertainties in MSD
    
    Returns:
    - m: Slope of the fit
    - sm: Uncertainty in slope
    - b: Intercept of the fit
    - sb: Uncertainty in intercept
    """
    # Define linear function
    def linear_func(x, m, b):
        return m * x + b
    
    # Perform fit with curve_fit, using y_uncertainty as weights
    popt, pcov = curve_fit(linear_func, x, y, sigma=y_uncertainty, absolute_sigma=True)
    m, b = popt
    sm, sb = np.sqrt(np.diag(pcov))  # Uncertainties in m and b
    return m, sm, b, sb

# -----------------------------
# 6. Perform Linear Fit
# -----------------------------
# Perform the fit with uncertainties
m, sm, b, sb = linear_fit_with_uncertainty(time_intervals, mean_squared_displacement, msd_uncertainty)

# -----------------------------
# 7. Plot Fit and Residuals
# -----------------------------
def plot_fit_and_residuals(x, y, y_uncertainty, fit_slope, fit_intercept):
    """
    Plot the MSD vs Time with the linear fit and residuals.
    
    Parameters:
    - x: Time intervals (s)
    - y: Mean Squared Displacement (µm²)
    - y_uncertainty: Uncertainties in MSD (µm²)
    - fit_slope: Slope from linear fit
    - fit_intercept: Intercept from linear fit
    """
    # Calculate fitted values and residuals
    y_fit = fit_slope * x + fit_intercept
    residuals = y - y_fit

    # Create a figure with two subplots: MSD vs Time and Residuals
    plt.figure(figsize=(10, 8))

    # -----------------------------
    # 7A. Plot MSD vs Time with Linear Fit
    # -----------------------------
    plt.subplot(2, 1, 1)
    plt.errorbar(x, y, yerr=y_uncertainty, fmt='o', label='MSD Data', capsize=3)
    plt.plot(x, y_fit, label=f'Fit: Slope = {fit_slope:.5e} µm²/s, Intercept = {fit_intercept:.5f} µm²', color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Squared Displacement (µm²)")
    plt.title("Mean Squared Displacement vs. Time with Linear Fit")
    plt.legend()
    plt.grid(True)

    # -----------------------------
    # 7B. Plot Residuals
    # -----------------------------
    plt.subplot(2, 1, 2)
    plt.errorbar(x, residuals, yerr=y_uncertainty, fmt='o', color='red', capsize=3)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Residuals (µm²)")
    plt.title("Residuals of Linear Fit")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Plot the fit with residuals to verify the linearity of the fit
plot_fit_and_residuals(time_intervals, mean_squared_displacement, msd_uncertainty, m, b)

# -----------------------------
# 8. Calculate Diffusion Coefficient (D) and Boltzmann Constant (k)
# -----------------------------
# Diffusion Coefficient (D) Calculation
# For 2D: <MSD> = 4Dt => D = m / 4
D = m / 4  # µm²/s
D_uncertainty = sm / 4  # µm²/s

# Convert D to SI units (m²/s)
D_SI = D * 1e-12  # m²/s
D_uncertainty_SI = D_uncertainty * 1e-12  # m²/s

# Stokes Drag Calculation
gamma = 6 * np.pi * viscosity * radius  # Pa·s·m

# Calculate Boltzmann Constant (k)
k_calculated = (D_SI * gamma) / temperature  # J/K

# Error Propagation for k Uncertainty
# Given:
# k = (D * gamma) / T
# So, uncertainty in k is:
# sigma_k = sqrt( (gamma / T * sigma_D)^2 + (D / T * sigma_gamma)^2 + (D * sigma_T * gamma / T^2)^2 )

# Calculate uncertainty in gamma
gamma_uncertainty = 6 * np.pi * (viscosity_uncertainty * radius + viscosity * radius_uncertainty)  # Pa·s·m

# Calculate uncertainty in k
k_calculated_uncertainty = np.sqrt(
    (gamma / temperature * D_uncertainty_SI) ** 2 +  # Contribution from D uncertainty
    (D_SI / temperature * gamma_uncertainty) ** 2 +  # Contribution from gamma uncertainty
    (D_SI * gamma / temperature**2 * T_uncertainty) ** 2  # Contribution from T uncertainty
)

# -----------------------------
# 9. Output Calculated Constants and Statistical Metrics
# -----------------------------
print(f"Calculated Boltzmann Constant (k): {k_calculated:.4e} J/K ± {k_calculated_uncertainty:.4e} J/K")
# Compare to accepted value of k
percent_difference = np.abs((k_calculated - k_accepted) / k_accepted) * 100
print(f"Percent Difference compared to accepted value of k: {percent_difference:.2f}%")
print(f"Diffusion Coefficient (D): {D:.4e} µm²/s ± {sm / 4:.4e} µm²/s")
print(f"Slope (m): {m:.5e} µm²/s")
print(f"Slope Uncertainty (sm): {sm:.5e} µm²/s")
print(f"Intercept (b): {b:.5f} µm²")
print(f"Intercept Uncertainty (sb): {sb:.5f} µm²")

# -----------------------------
# 10. Calculate Statistical Metrics
# -----------------------------
def calculate_r_squared(y, y_fit):
    """
    Calculate the R-squared value.
    
    Parameters:
    - y: Observed data points
    - y_fit: Fitted model values
    
    Returns:
    - R-squared value
    """
    ss_res = np.sum((y - y_fit) ** 2)  # Sum of squared residuals
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)

def calculate_chi_square(observed, expected, uncertainties):
    """
    Calculate the chi-square statistic.
    
    Parameters:
    - observed: Observed data points
    - expected: Expected (fitted) data points
    - uncertainties: Uncertainties in observed data points
    
    Returns:
    - Chi-square statistic
    """
    chi_square = np.sum(((observed - expected) / uncertainties) ** 2)
    return chi_square

def calculate_chi_square(observed, expected, uncertainties):
    """
    Calculate the chi-square statistic.
    
    Parameters:
    - observed: Observed data points
    - expected: Expected (fitted) data points
    - uncertainties: Uncertainties in observed data points
    
    Returns:
    - Chi-square statistic
    """
    chi_square = np.sum(((observed - expected) / uncertainties) ** 2)
    return chi_square

# Calculate fitted values
y_fit = m * time_intervals + b
residuals = mean_squared_displacement - y_fit

# Calculate R^2
r_squared = calculate_r_squared(mean_squared_displacement, y_fit)

# Perform linear regression for p-value (alternative method using linregress)
slope_lr, intercept_lr, r_value, p_value_lr, std_err_lr = linregress(time_intervals, mean_squared_displacement)

# Calculate the chi-square value
chi_square = calculate_chi_square(mean_squared_displacement, y_fit, msd_uncertainty)

# Degrees of freedom
degrees_of_freedom = len(mean_squared_displacement) - 2  # Number of data points - 2 parameters (m and b)

# Calculate p-value from chi-square
p_value_chi_square = 1 - chi2.cdf(chi_square, degrees_of_freedom)

# Calculate Reduced Chi-Square
reduced_chi_square = chi_square / degrees_of_freedom

# Output statistical results
print(f"\nStatistical Metrics:")
print(f"Chi-Square Statistic (χ²): {chi_square:.5f}")
print(f"Degrees of Freedom: {degrees_of_freedom}")
print(f"P-Value from χ²: {p_value_chi_square:.5e}")
print(f"Reduced Chi-Square (χ²/ν): {reduced_chi_square:.5f}")
print(f"R-squared: {r_squared:.5f}")
print(f"P-value from linear regression: {p_value_lr:.5e}")