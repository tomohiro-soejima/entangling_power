using PyCall
using TensorOperations
using LinearAlgebra

#In the case of second moment. The second-moment operators consist of the following matA, 
#where local random circuits are summation of it and brick-wall circuits are tensor products of it.

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

# This function takes obscenely long to compile
# Not using it for that reason
# function contract(small_mat, big_mat, ind1, ind2)
#     index_length = length(size(big_mat))
#     index_list = -1 * collect(1:index_length)
#     index_list[ind1] = 1
#     index_list[ind2] = 2
#     return ncon((small_mat, big_mat), ([-ind1, -ind2, 1, 2], index_list)) 
# end

function contract(small_mat, big_mat, ind1, ind2)
    index_length = length(size(big_mat))
    index_list = -1 * collect(1:index_length)
    index_list[ind1] = 1
    index_list[ind2] = 2
    index_list_destination =  -1 * collect(1:index_length)
    index_list_small = [-ind1, -ind2, 1, 2]
    return einsum(EinCode((string.(index_list_small), string.(index_list)), string.(index_list_destination)), (small_mat, big_mat))
    # return ncon((small_mat, big_mat), ([-ind1, -ind2, 1, 2], index_list)) 
end

function make_brick_wall_dense_matrix(mat, n_qubits, local_hilbert_space_dimension)
    dimension = local_hilbert_space_dimension^n_qubits
    final_matrix = Matrix{ComplexF64}(I, (dimension, dimension))
    final_matrix = reshape(final_matrix, tuple((ones(Int, 2*n_qubits) * 2)...))
    @assert n_qubits % 2 == 0
    for ind in 1:2:n_qubits
        contract(mat, final_matrix, ind, ind+1)
    end

    for ind in 2:2:n_qubits-2
        contract(mat, final_matrix, ind, ind+1)
    end

    contract(mat, final_matrix, n_qubits, 1)
    return final_matrix
end