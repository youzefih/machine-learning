import numpy as np
# X          - vector
# y          - vector
# theta      - vector
# alpha      - scalar
# iterations - scalar

def gradientDescent(X, y, theta, alpha, num_iters):
    '''
    This function returns a tuple (theta, cost_array)
    '''
    m = len(y)

    cost_array =[]

    for i in range(0, num_iters):
        ################ START TODO #################
        # Make predictions
        # Hint: y_hat = theta_0 + (theta_1 * x_1) + (theta_2 * x_2)
        # Shape of y_hat: m by 1
        if i == m:
            break
        y_hat =(1/m)*(theta[0] + (theta[1] * X[i,1]) + (theta[2] * X[i,2]))

        # Compute the difference between predictions and true values
        # Shape of residuals: m by 1
        # print(y_hat)
        residuals = y_hat - y
        # Calculate the current cost
        cost =+ sum([residuals**2])
        cost_array.append(cost)
        # Compute gradients
        # Shape of gradients: 3 by 1, i.e., same as theta
        gradients = residuals.dot(y) * alpha
        # Update theta
        theta = theta - residuals.dot(X) * alpha
        ################ END TODO ##################


    return theta, cost_array
