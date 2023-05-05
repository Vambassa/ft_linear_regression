import matplotlib.pyplot as plt
import gradient_descent
import csv
import numpy as np


x_data = []
y_data = []
X_scaled = []


# Read Dataset
def read_dataset():
    with open('data.csv') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if not i:
                continue
            x_data.append(float(row[0]))
            y_data.append(float(row[1]))
    min_max_scaler()


def reverse_scaler(t0, t1):
    x_max = max(x_data)
    x_min = min(x_data)
    t0 -= t1 * x_min / (x_max - x_min)
    t1 /= (x_max - x_min)
    return round(t0, 2), round(t1, 2)


def min_max_scaler():
    x_min = min(x_data)
    x_max = max(x_data)
    for elem in x_data.copy():
        X_scaled.append((elem - x_min) / (x_max - x_min))


# Fit model
read_dataset()
t0_scaled, t1_scaled, errors = gradient_descent.GradientDescent(X_scaled, y_data).fit()
t0, t1 = reverse_scaler(t0_scaled, t1_scaled)
with open("weights.txt", "w") as file:
    file.write(str(t0))
    file.write("\n")
    file.write(str(t1))
print("Ready! Now you can predict the cost of your car:)")


# Price & mileage plot
def estimate_price(x):
    return t0 + t1 * x


x_pd_data = np.array(x_data)
y_pd_data = np.array(y_data)
plt.figure(figsize=(10, 5))
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.plot(x_pd_data, estimate_price(x_pd_data), label="predicted", c="r")
plt.scatter(x_data, y_pd_data, label="train", c="orange")
plt.grid(alpha=0.2)
plt.legend()
plt.show()

# Loss plot
errors_np = np.array(errors)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(errors_np)
plt.show()
