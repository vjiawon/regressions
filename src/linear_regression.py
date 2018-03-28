# Assume a linear relationship y = mx + b 
# Cost function is Mean Squared Error (mse)

# general algorithm is to 
    # 1.) Calculate the current predicted value (y)
    # 2.) Calculate the cost i.e. mse 
    # 3.) Calculate d/dm and d/db of cost 
    # 4.) Update each tunable parameter (m and b in this case) in their 
    #       negative gradient by a value = the step 

from statistics import mean

'''
Data is a tuple of (x, y)
'''
def linear_regression_step(data, m, b, learning_rate):
    # Calculate the predicted values 
    data_and_estimate = [(x, y_act, x*m+b) for x, y_act in data]

    # Calculate the cost i.e. mse
    cost = mean([(y_act-y_est)**2 for x, y_act, y_est in data_and_estimate])

    # Calculate the partial derivatives 
    d_dm = -2 * mean([ x * (y_act - y_est) for x, y_act, y_est in data_and_estimate])
    d_db = -2 * mean([y_act - y_est for x, y_act, y_est in data_and_estimate])
       
    # Update m and b by the learning rate in the direction of their negative partials 
    adjusted_m = m - learning_rate * d_dm
    adjusted_b = b - learning_rate * d_db

    return adjusted_m, adjusted_b, cost, d_dm, d_db

def linear_regression(data, epochs=30000, inital_m=1, initial_b=0, learning_rate=0.0002):
    # Perform a linear regression per epoch with the data 
    current_m = inital_m
    current_b = initial_b
    for increment in range(epochs):
        
        adjusted_m, adjusted_b, cost, d_dm, d_db = linear_regression_step(data, current_m, current_b, learning_rate)
        print(f"For iteration {increment}: m = {adjusted_m}; b = {adjusted_b}; cost = {cost}; d/dm = {d_dm}; d/db = {d_db}")
        current_m = adjusted_m
        current_b = adjusted_b

def linear_regression_test_harness():
    with open('/Users/Vish/Documents/Repositories/regressions/regressions/src/train.csv', 'r') as f:
        f.readline() # skip the header line
        # data = [ 
        #     [ ((float(s[0]), float(s[1]) for s in line.split(",") if len(s) == 2 ]
        #     for line in f 
        # ]
        data = []
        for line in f:
            split_strings = line.split(",")
            if (len(split_strings) == 2):
                t = float(split_strings[0]), float(split_strings[1])
                data.append(t)
        # todo: parse as one list comprehension
        # data = [ [ (float(l[0]), float(l[1])) for l in line.split(",") if l[0].isdigit()] for line in f ]
        #split_strings = [ [l for l in line.split(",")] for line in f ]
        #data = [t for t in map(lambda x: (float(x[0]), float(x[1])), split_strings[1:])]
    linear_regression(data)
#[int(chunk) for chunk in line.split() if chunk.isdigit() ]


linear_regression_test_harness()
        
