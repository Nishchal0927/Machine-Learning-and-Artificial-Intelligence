# Sample visualization code you can use
import matplotlib.pyplot as plt

# Fraud statistics
years = [2018, 2019, 2020, 2021, 2022]
fraud_losses = [24.2, 25.6, 27.8, 28.7, 30.2]  # in billions

plt.figure(figsize=(8, 4))
plt.plot(years, fraud_losses, marker='o', linewidth=2, color='red')
plt.title('Global Credit Card Fraud Losses (Billions $)')
plt.xlabel('Year')
plt.ylabel('Losses ($ Billions)')
plt.grid(True, alpha=0.3)
plt.show()