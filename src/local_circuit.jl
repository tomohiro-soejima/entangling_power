function make_local_circuit_dense_matrix(mat, n_qubits, local_hilbert_space_dimension)
    dimension = local_hilbert_space_dimension^n_qubits
    T = eltype(mat)
    identity = Matrix{T}(I, (dimension, dimension))
    identity_reshaped = reshape(identity, tuple((ones(Int, 2*n_qubits) * 2)...))
    final_matrix = zeros(T, tuple((ones(Int, 2*n_qubits) * 2)...))
    @assert n_qubits % 2 == 0
    for ind in 1:n_qubits
        final_matrix += contract(mat, identity_reshaped, ind, ind+1)
    end

    final_matrix += contract(mat, identity_reshaped, n_qubits, 1)
    return final_matrix
end

function diagonalize_local_circuit(n_qubits, local_hilbert_space_dimension, eu, gu)
    mat = (matA(local_hilbert_space_dimension, eu, gu))
    mat_reshaped = reshape(mat, (2, 2, 2, 2))

    mat_big = make_local_circuit_dense_matrix(mat_reshaped, n_qubits, local_hilbert_space_dimension)
    return eigen(reshape(mat_big, (2^n_qubits, 2^n_qubits))), mat_big
end

function diagonalize_local_circuit_arnoldi(n_qubits, local_hilbert_space_dimension, eu, gu)
    mat = (matA(local_hilbert_space_dimension, eu, gu))
    mat_reshaped = reshape(mat, (2, 2, 2, 2))

    mat_big = reshape(make_local_circuit_dense_matrix(mat_reshaped, n_qubits, local_hilbert_space_dimension), (2^n_qubits, 2^n_qubits))
    return eigsolve(mat_big, 4)
end
