import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import sympy as sp

df = pd.read_csv (r'C:\Users\andre\Documents\GitHub\MSE-307-HW-02\Data.csv')
x = pd.DataFrame(df['x'])
y = pd.DataFrame(df['y'])

lr = LinearRegression().fit(x,y)
print(lr.coef_[0,0])
print(lr.intercept_[0])


x_fit = sp.symbols('x_fit')
y_fit = lr.coef_[0,0] * x_fit + lr.intercept_[0]
lam_y = sp.lambdify(x_fit, y_fit, modules=['numpy'])

x_vals = np.linspace(0, 0.0005, 2)
y_vals = lam_y(x_vals)

plt.plot(x_vals, y_vals)

plt.scatter(x,y,s=20,color='red')
plt.savefig('Plot.png', dpi=300)
plt.show()