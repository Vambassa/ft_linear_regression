import sys


# Ask mileage
mileage = 0
try:
    mileage = float(input('Enter the mileage of the car: '))
except ValueError:
    print('Mileage is a number')
    exit()


# Predict price
try:
    with open('weights.txt') as file:
        t0 = float(file.readline())
        t1 = float(file.readline())
        predicted_price = int(t0 + (t1 * mileage))
        if predicted_price < 0:
            predicted_price = 0
        print(f"Mileage = {mileage}")
        print(f"Price = {predicted_price}")
except IOError:
    print('Parameter values were not found. First run "fit.py"')
    exit()
