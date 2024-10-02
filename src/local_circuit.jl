function make_local_circuit_dense_matrix(mat, n_qubits, local_hilbert_space_dimension)
    dimension = local_hilbert_space_dimension^n_qubits
    T = eltype(mat)
    identity = Matrix{T}(I, (dimension, dimension))
    identity_reshaped = reshape(final_matrix, tuple((ones(Int, 2*n_qubits) * 2)...))
    final_matrix = zeros(T, tuple((ones(Int, 2*n_qubits) * 2)...))
    @assert n_qubits % 2 == 0
    for ind in 1:n_qubits
        final_matrix += co-1ntract(mat, final_matrix, ind, ind+1)
    end

    final_matrix += contract(mat, final_matrix, n_qubits, 1)
    return final_matrix
end