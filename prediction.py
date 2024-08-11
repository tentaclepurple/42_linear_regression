from linear_regression import ft_linear_regression
import pickle
import os


def prediction(slope, intersect):
    try:
        print("Introduce a mileage in km to get the price of your car:")
        mileage = float(input())

        file = "params.pkl"
        if os.path.exists(file):
            if os.access(file, os.R_OK):
                with open(file, 'rb') as file:
                    param = pickle.load(file)

                slope, intersect = param

        price = slope * mileage + intersect

        print(f"Making a prediction theta1 = {slope} and theta0 = {intersect}")
        print(f"The estimated price for your car for {mileage} kms is: {price:.2f}")
   
    except Exception as e:
        print("An error occured: ", e)


if __name__ == '__main__':

    theta1 = 0
    theta0 = 0

    try:
        print("Welcome to the ft_linear_regression")
        print("Select an option:")
        print("1. Train the model")
        print("2. Make a prediction")
        option = int(input())
        
        if option == 1:
            theta1, theta0 = ft_linear_regression()
        elif option == 2:
            prediction(theta1, theta0)
        else:
            raise Exception()

    except Exception as e:
        print("Invalid option")