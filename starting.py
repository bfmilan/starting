import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Test Pandas
data = pd.DataFrame({'A': np.random.rand(10), 'B': np.random.rand(10)})
print("Pandas DataFrame:\n", data)

# Test Matplotlib
plt.plot(data['A'], label='A')
plt.plot(data['B'], label='B')
plt.legend()
plt.title("Test Plot")
plt.show()
