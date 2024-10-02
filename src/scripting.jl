using NPZ

function run_brick_wall_scan(n_qubit, res)
    eu_list = range(1/2, 2/3, res)
    gu_list = range(0.25, 0.75, res)
    iter = Iterators.product(eu_list, gu_list)

    res_list = map(iter) do item
        eu, gu = item
        res=diagonalize_brick_wall_arnoldi_matrix_free(n_qubit, 2, eu, gu)
        res[1]
        end

    second_largest_eigenvalue = map(res_list) do res
        first(filter(x->abs(x-1)>1e-8, res))
    end

    npzwrite("eigvals_n=$(n_qubit)_res_$(res).npy", second_largest_eigenvalue)
    npzwrite("eulist_n=$(n_qubit)_res_$(res).npy", eu_list)
    npzwrite("gulist_n=$(n_qubit)_res_$(res).npy", gu_list)
end



# imshow(transpose(abs.(second_largest_eigenvalue)), origin="lower", extent=(1/2, 2/3, 0.25, 0.75), aspect="auto")
# xlabel(L"$e_u$")
# ylabel(L"$g_u$")
# colorbar()
# title("Brick wall, n=$(n_qubit)")
# # aspect("equal")