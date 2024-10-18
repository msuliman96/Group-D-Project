import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Bacteria.xlsx'
bacteria_data = pd.read_excel(file_path)

plt.figure(figsize=(10, 6))
for column in bacteria_data.columns[:-1]:
    plt.scatter(bacteria_data[column], bacteria_data['Yield'], alpha=0.5, label=column)

plt.title('Scatter Plot of Variables vs Yield')
plt.xlabel('Variables')
plt.ylabel('Yield')
plt.legend(title='Variables')
plt.grid(True)
# plt.savefig('Scatter_All_Variables_vs_Yield.png')
plt.show()