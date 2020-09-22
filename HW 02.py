from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np


df = pd.read_csv (r'C:\Users\andre\Documents\GitHub\MSE-307-HW-02\Data.csv')
x = pd.DataFrame(df['x'])
y = pd.DataFrame(df['y'])

lr = LinearRegression().fit(x[0:5],y[0:5])
x_vals = np.linspace(0,4E-4,2)
y_vals = lr.coef_[0,0] * x_vals + lr.intercept_[0]

fig, ax = plt.subplots()

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.xaxis.set_major_formatter(formatter)
ax.scatter(x,y,s=20,color='red')
ax.plot(x_vals,y_vals)
plt.title('Gallium Diffusion')
plt.xlabel('x^2 (cm^2)')
plt.ylabel('ln(Ga %)')
plt.savefig('Plot.png', dpi=300)
plt.show()

t = 24 * 3600
D = - lr.coef_[0,0] ** (-1) / (4 * t)
a = lr.intercept_[0] * np.sqrt(4 * np.pi * D * t)
print(D,'cm^2/s')
print(a,'cm')