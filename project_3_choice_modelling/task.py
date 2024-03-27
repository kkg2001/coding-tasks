import numpy as np
import matplotlib.pyplot as plt
"""
    Calculate the probability of each alternative in a multinomial choice setting using
    the logistic function.

    Parameters:
        parameters (dict): Dictionary containing the β coefficients.
        data (dict): Dictionary containing the independent variables.
        utilities (list of functions): List of functions defining the deterministic utilities
                                       for each alternative based on the parameters and data.

    Returns:
        probabilities (dict): Dictionary with keys representing each alternative and values
                              as lists containing the calculated probabilities for each data point.
    """

def calculate_probabilities(parameters, data, utilities):
    #Calculate utility for each alternative
    utilities_values = []
    for utility in utilities:
        utilities_values.append(utility(parameters, data))

    # Calculate exponentials of utilities
    exp_utilities=np.exp(utilities_values)

    # Calculate denominators i.e. sum of all exponential values
    denominator=np.sum(utilities_values,axis=0)

    #Calculate probability for each alternative
    probabilities={}
    for i,exp_utility in enumerate(exp_utilities):
        alter_name=f"P{i+1}"
        probabilities[alter_name]=exp_utility/denominator

    return probabilities

# Define the deterministic utilities functions
def utility_1(parameters, data):
    return parameters['beta01']+np.dot(parameters['beta1'],(data['X1']))+np.dot(parameters['betaS113'],(data['S1']))
    
def utility_2(parameters, data):
    return parameters['beta02']+np.dot(parameters['beta2'],(data['X2']))+np.dot(parameters['betaS123'],(data['S1']))

def utility_3(parameters, data):
    return parameters['beta03']+np.dot(parameters['beta1'],data['Sero'])+np.dot(parameters['beta2'],data['Sero'])

#Parameters stored in a dictionary
parameters={
    'beta01': 0.1,
    'beta1': -0.5,
    'beta2': -0.4,
    'beta02': 1,
    'beta03': 0,
    'betaS113': 0.33,
    'betaS123': 0.58

}

#Data stored in a dictionary
data={
    'X1':[2,1,3,4,2,1,8,7,3,2],
    'X2':[8,7,4,1,4,7,2,2,3,1],
    'Sero':[0,0,0,0,0,0,0,0,0,0],
    'S1':[3,8,4,7,1,6,5,9,2,3],
    'AV1':[1,1,1,1,1,0,0,1,1,0],
    'AV2':[1,1,1,0,0,1,1,1,0,1],
    'AV3':[1,1,0,0,1,1,1,1,1,1]

}

#Utility values stored in a list
utilities=[utility_1,utility_2, utility_3]

probabilities=calculate_probabilities(parameters, data, utilities)


#Saving the output in .txt format
with open("probabilities.txt","w") as f:
    for alternative, probs in probabilities.items():
        f.write(f"{alternative}: {probs}\n")
 
