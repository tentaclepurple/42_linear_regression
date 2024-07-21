from linear_regression import ft_linear_regression


def prediction():
    try:
        print("Introduce a mileage in km to get the price of your car:")
        mileage = float(input())

        slope, intersec = ft_linear_regression()
        price = slope * mileage + intersec
        print(f"The estimated price for your car for {mileage} kms is: {price:.2f}")
   
    except Exception as e:
        print("An error occured: ", e)


if __name__ == '__main__':
    prediction()