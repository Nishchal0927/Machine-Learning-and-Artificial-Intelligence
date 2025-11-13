# Class distribution visualization
import matplotlib.pyplot as plt

labels = ['Legitimate\n(284,315)', 'Fraudulent\n(492)']
sizes = [284315, 492]
colors = ['lightgreen', 'red']
explode = (0, 0.1)  # explode the fraud slice

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Extreme Class Imbalance in Our Dataset')
plt.show()