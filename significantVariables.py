import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Bacteria.xlsx'
bacteria_data = pd.read_excel(file_path)

significant_variables = ['x8', 'x7', 'x10', 'x3', 'x5']

plt.figure(figsize=(15, 10))

for i, column in enumerate(significant_variables, 1):
    plt.subplot(3, 2, i)
    plt.scatter(bacteria_data[column], bacteria_data['Yield'], alpha=0.5)
    plt.title(f'{column} vs Yield')
    plt.xlabel(column)
    plt.ylabel('Yield')
    plt.grid(True)

plt.tight_layout()
plt.savefig('Scatter_Significant_Variables_vs_Yield.png')
plt.show()
