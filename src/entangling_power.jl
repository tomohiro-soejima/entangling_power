using PyCall
using TensorOperations
using LinearAlgebra
using KrylovKit

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
        final_matrix = contract(mat, final_matrix, ind, ind+1)
    end

    for ind in 2:2:n_qubits-2
        final_matrix = contract(mat, final_matrix, ind, ind+1)
    end

    final_matrix = contract(mat, final_matrix, n_qubits, 1)
    return final_matrix
end


function diagonalize_brick_wall(n_qubits, local_hilbert_space_dimension, eu, gu)
    mat = py"matA"(local_hilbert_space_dimension, eu, gu)
    mat_reshaped = reshape(mat, (2, 2, 2, 2))

    mat_big = make_brick_wall_dense_matrix(mat_reshaped, n_qubits, local_hilbert_space_dimension)
    return eigen(reshape(mat_big, (2^n_qubits, 2^n_qubits))), mat_big
end


#########################################################################
################# Implement sparse version ##############################
#########################################################################

struct NSiteVector{T, N} <: AbstractVector{T}
    tensor :: Array{T, N}
end

import VectorInterface: zerovector, scale, scale!, scale!!, scalartype, add, add!, add!!, inner
Base.:+(vec1::NSiteVector{T, N}, vec2::NSiteVector{T, N}) where {T, N} = NSiteVector{T, N}(vec1.tensor + vec2.tensor)
add(vec1::NSiteVector{T, N}, vec2::NSiteVector{T, N}, α=1::Number, β=1::Number) where {T, N} = α * vec1 + β * vec2
add(vec1::NSiteVector{T, N}, vec2::NSiteVector{T, N}, α::Number, β::Number) where {T, N} = α * vec1 + β * vec2 # why is this necessary at all?
add!!(vec1::NSiteVector{T, N}, vec2::NSiteVector{T, N}, α=1::Number, β=1::Number) where {T, N} = add(vec1, vec2, α, β)
add!!(vec1::NSiteVector{T, N}, vec2::NSiteVector{T, N}, α::Number, β::Number) where {T, N} = add(vec1, vec2, α, β) #somehow necessary
add!!(vec1::NSiteVector{T, N}, vec2::NSiteVector{T, N}, α::Number) where {T, N} = add!!(vec1, vec2, α, 1)

Base.:*(a :: Number, vec::NSiteVector{T, N}) where {T, N} = NSiteVector{T, N}(a * vec.tensor)
scale!!(v :: NSiteVector{T, N}, α::Number) where {T, N} = scale(v, α)
scale(v :: NSiteVector{T, N}, α::Number) where {T, N} = α * v


Base.size(vec::NSiteVector) = prod(size(vec.tensor))
# LinearAlgebra.dot(vec1::NSiteVector, vec2::NSiteVector) = reshape(vec1.tensor, :)' * reshape(vec2.tensor, :)
inner(vec1::NSiteVector, vec2::NSiteVector) = reshape(vec1.tensor, :)' * reshape(vec2.tensor, :)
LinearAlgebra.norm(vec::NSiteVector) = norm(vec.tensor)
zerovector(vec::NSiteVector{T,N}) where {T, N} = NSiteVector{T, N}(zeros(T, size(vec.tensor)))
zerovector(vec::NSiteVector{T,N}, ::Type{S}) where {T, N, S<:Number} = NSiteVector{S, N}(zeros(S, size(vec.tensor)))
zerovector(vec::NSiteVector{T,N}, ::S) where {T, N, S<:Number} = NSiteVector{S, N}(zeros(S, size(vec.tensor))) # don't know which one of the two needed to be implemented
Base.zero(vec::NSiteVector{T, N}) where {T, N} = NSiteVector{T, N}(zero(vec.tensor))

struct BrickWall{T}
    mat :: Array{T, 4}
end

function contract(small_mat, big_mat :: NSiteVector{T, N}, ind1, ind2) where {T, N}
    index_length = N
    index_list = -1 * collect(1:index_length)
    index_list[ind1] = 1
    index_list[ind2] = 2
    index_list_destination =  -1 * collect(1:index_length)
    index_list_small = [-ind1, -ind2, 1, 2]
    return NSiteVector{T, N}(einsum(EinCode((string.(index_list_small), string.(index_list)), string.(index_list_destination)), (small_mat, big_mat.tensor)))
end

function (bw :: BrickWall)(vec::NSiteVector{T, N}) where {T, N}
    n_qubits = N
    final_matrix = vec
    @assert n_qubits % 2 == 0
    for ind in 1:2:n_qubits
        final_matrix = contract(bw.mat, final_matrix, ind, ind+1)
    end

    for ind in 2:2:n_qubits-2
        final_matrix = contract(bw.mat, final_matrix, ind, ind+1)
    end

    final_matrix = contract(bw.mat, final_matrix, n_qubits, 1)
    return final_matrix
end

# function Base.:*(bw :: BrickWall, vec::NSiteVector{T, N}) where {T, N}
#     n_qubits = N
#     final_matrix = vec.tensor
#     @assert n_qubits % 2 == 0
#     for ind in 1:2:n_qubits
#         final_matrix = contract(bw.mat, final_matrix, ind, ind+1)
#     end

#     for ind in 2:2:n_qubits-2
#         final_matrix = contract(bw.mat, final_matrix, ind, ind+1)
#     end

#     final_matrix = contract(bw.mat, final_matrix, n_qubits, 1)
#     return final_matrix
# end