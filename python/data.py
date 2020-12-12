from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

df = pd.read_csv('data.csv', ',')
df['round'] += 1  # because it starts at zero
df_mean = df[df['round'] >= 5].groupby('name').mean()
print(df_mean)

# fig = plt.figure()
# sbn.boxplot(x='name', y='f1', data=df)

fig = plt.figure(figsize=[12, 5])
sbn.lineplot(x='round', y='f1', hue='name', data=df)
plt.legend(['Image & Gradients', 'Image', 'Scaled Gradients'],
           loc="lower right", fontsize='xx-large', title=None)
plt.xlabel('Training Epochs (x10)', fontsize='xx-large')
plt.ylabel('F1 Score', fontsize='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')

# fig = plt.figure()
# sbn.boxplot(x='name', y='time', data=df)

plt.show()
