#In the case of second moment. The second-moment operators consist of the following matA, 
#where local random circuits are summation of it and brick-wall circuits are tensor products of it.

# need to wrap like this for it to work?
function __init__()
    py"""
    import numpy as np

    def eh(d):
        '''Calculate eh as a function of d.'''
        return (d**2 - 1) / (d**2 + 1)

    def matA(d, eu, gu):
        '''Calculate the matrix A as a function of d, eu, and gu.'''
        eh_d = eh(d)
        
        # Define the elements of the matrix
        mat = np.array([
            [1, 0, 0, 0],
            [(d / (d**2 + 1)) * (eu / eh_d), 1 - eu / (2 * eh_d) - gu, -eu / (2 * eh_d) + gu, (d / (d**2 + 1)) * (eu / eh_d)],
            [(d / (d**2 + 1)) * (eu / eh_d), -eu / (2 * eh_d) + gu, 1 - eu / (2 * eh_d) - gu, (d / (d**2 + 1)) * (eu / eh_d)],
            [0, 0, 0, 1]
        ])
        
        return mat
    """
end

matA(d, eu, gu) = py"matA"(d, eu, gu)