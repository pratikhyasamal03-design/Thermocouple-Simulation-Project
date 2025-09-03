import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
fs = 1000  # Sampling frequency (Hz)
t = np.linspace(0, 1, fs)  # Time vector (1 second)
temp_true = 100 + 50 * np.sin(np.pi * t)  # True temperature (°C); modifiable (e.g., base + amplitude * sin(pi * t))
noise_amplitude = 0.02  # Noise in mV; modifiable
cutoff = 5  # Cutoff frequency (Hz); modifiable
order = 4  # Filter order

# Simulate thermocouple (Type K: ~0.04 mV/°C)
voltage_true = temp_true * 0.04  # Convert temperature to mV
voltage_noisy = voltage_true + noise_amplitude * np.random.randn(len(t))  # Add Gaussian noise

# Design low-pass Butterworth filter
b, a = signal.butter(order, cutoff / (fs / 2), btype='low')  # Filter coefficients

# Apply filter (zero-phase)
voltage_filtered = signal.filtfilt(b, a, voltage_noisy)

# Calibrate back to temperature
temp_filtered = voltage_filtered / 0.04  # Convert mV back to °C

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, voltage_noisy, label='Noisy Voltage (mV)')
plt.plot(t, voltage_true, 'r--', label='True Voltage (mV)')
plt.title('Thermocouple Voltage Output')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, temp_filtered, label='Filtered Temperature (°C)')
plt.plot(t, temp_true, 'r--', label='True Temperature (°C)')
plt.title('Processed Temperature Reading')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Calculate and display error metrics
rmse_before = np.sqrt(np.mean((temp_true - voltage_noisy / 0.04)**2))
rmse_after = np.sqrt(np.mean((temp_true - temp_filtered)**2))
print(f"RMSE Before Filtering: {rmse_before:.2f} °C")
print(f"RMSE After Filtering: {rmse_after:.2f} °C")
