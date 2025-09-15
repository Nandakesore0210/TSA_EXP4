# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 15/09/2025

### AIM:
To implement ARMA model in python.

### ALGORITHM:
1. Import necessary libraries.

2. Set up matplotlib settings for figure size.

3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.

5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```py
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [10, 7.5]

ar1 = np.array([1,0.33])
ma1 = np.array([1,0.9])
ARMA_1 = ArmaProcess(ar1,ma1).generate_sample(nsample = 1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_1)
plot_pacf(ARMA_1)
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_2)
plot_pacf(ARMA_2)
```
OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img width="845" height="646" alt="image" src="https://github.com/user-attachments/assets/67f260f8-3f6c-4da2-a3aa-ee9ca0cc074f" />

Partial Autocorrelation

<img width="863" height="654" alt="image" src="https://github.com/user-attachments/assets/2fe3296f-f103-48b7-89c0-c0be3b18fab7" />

Autocorrelation

<img width="870" height="650" alt="image" src="https://github.com/user-attachments/assets/652833a8-6761-4dca-a7cc-f31ed83e8a36" />

SIMULATED ARMA(2,2) PROCESS:

<img width="879" height="664" alt="image" src="https://github.com/user-attachments/assets/9e385f88-5269-4b80-a7be-abd9b0516bd4" />

Partial Autocorrelation

<img width="858" height="655" alt="image" src="https://github.com/user-attachments/assets/557c9848-2814-465b-923a-a8f3a22a5c11" />

Autocorrelation

<img width="852" height="645" alt="image" src="https://github.com/user-attachments/assets/099d989e-b612-4a1e-9abd-68109f5bb4db" />



RESULT:
Thus, a python program is created to fir ARMA Model successfully.
