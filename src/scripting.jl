using NPZ

function run_brick_wall_scan(n_qubit, resolution, local_hilbert_space_dimension=2, verbosity=0, save=true)
    if local_hilbert_space_dimension == 2
        eu_list = range(1/2, 2/3, resolution)
        gu_list = range(0.25, 0.75, resolution)
    elseif local_hilbert_space_dimension == 3
        eu_list = range(0.7, 1, resolution)
        gu_list = range(0.35, 0.65, reresolutions)
    end
    iter = Iterators.product(eu_list, gu_list)

    res_list = map(iter) do item
        if verbosity > 0
            println(item)
        end
        eu, gu = item
        res=diagonalize_brick_wall_arnoldi_matrix_free(n_qubit, local_hilbert_space_dimension, eu, gu)
        res[1]
    end

    second_largest_eigenvalue = map(res_list) do res
        first(filter(x->abs(x-1)>1e-8, res))
    end

    if save
        npzwrite("eigvals_n=$(n_qubit)_res_$(resolution)_d_$(local_hilbert_space_dimension).npy", second_largest_eigenvalue)
        npzwrite("eulist_n=$(n_qubit)_res_$(resolution)_$(local_hilbert_space_dimension).npy", collect(eu_list))
        npzwrite("gulist_n=$(n_qubit)_res_$(resolution)_$(local_hilbert_space_dimension).npy", collect(gu_list))
    end
end



# imshow(transpose(abs.(second_largest_eigenvalue)), origin="lower", extent=(1/2, 2/3, 0.25, 0.75), aspect="auto")
# xlabel(L"$e_u$")
# ylabel(L"$g_u$")
# colorbar()
# title("Brick wall, n=$(n_qubit)")
# # aspect("equal")
