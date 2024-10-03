using Test
using EntanglingPower
using LinearAlgebra
@testset "basic operations" begin
    matA = reshape(EntanglingPower.matA(2, 2/3, 5/9), (2, 2, 2, 2))
    mat = EntanglingPower.make_brick_wall_dense_matrix(matA, 4, 2);
    mat_square = reshape(mat, (2^4, 2^4))
    vector_linear = rand(ComplexF64, 2^4)
    vec1 = EntanglingPower.NSiteVector{ComplexF64, 4}(reshape(vector_linear, (2, 2, 2, 2)))
    bw = EntanglingPower.BrickWall{ComplexF64}(reshape(matA, (2, 2, 2, 2)));

    @test reshape(bw(vec1).tensor, :) ≈ mat_square * vector_linear

    @test EntanglingPower.inner(vec1, bw(vec1)) ≈ dot(vector_linear, mat_square * vector_linear)

    @test norm(vec1) ≈ norm(vector_linear)
    @test norm(bw(vec1)) ≈ norm(mat_square * vector_linear)

    @test EntanglingPower.scale(vec1, 2).tensor ≈ vec1.tensor * 2
    @test EntanglingPower.add(vec1, vec1).tensor ≈ (vec1 + vec1).tensor

    @test EntanglingPower.add(vec1, vec1, 3, 4).tensor ≈ (3 * vec1 + 4 * vec1).tensor
end

@testset "finding_eigenvalues" begin
    @testset "local circuit" begin
        vals = EntanglingPower.diagonalize_local_circuit(4, 2, 2/3, 5/9)[1].values
        @test isapprox(abs(first(filter(x->abs(x-1)>1e-8, vals))), 0.758714, atol=1e-3)
    end

    @testset "brick wall circuit" begin
        vals=diagonalize_brick_wall(4, 2, 2/3, 5/9)[1].values
        second_largest_eigval = (first(filter(x->abs(x-1)>1e-8, vals)))
        @test isapprox(abs(second_largest_eigval), 0.0605844, atol=1e-5)

        vals=diagonalize_brick_wall_arnoldi(4, 2, 2/3, 5/9)[1]
        second_largest_eigval = (first(filter(x->abs(x-1)>1e-8, vals)))
        @test isapprox(abs(second_largest_eigval), 0.0605844, atol=1e-5)
        
        vals=diagonalize_brick_wall_arnoldi_matrix_free(4, 2, 2/3, 5/9)[1]
        second_largest_eigval = (first(filter(x->abs(x-1)>1e-8, vals)))
        @test isapprox(abs(second_largest_eigval), 0.0605844, atol=1e-5)
    end


end