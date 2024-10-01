using Test
using Revise
includet("../src/entangling_power.jl")

@testset "basic operations" begin
    matA = reshape(py"matA"(2, 2/3, 5/9), (2, 2, 2, 2))
    mat = make_brick_wall_dense_matrix(matA, 4, 2);
    mat_square = reshape(mat, (2^4, 2^4))
    vector_linear = rand(ComplexF64, 2^4)
    vec1 = NSiteVector{ComplexF64, 4}(reshape(vector_linear, (2, 2, 2, 2)))
    bw = BrickWall{ComplexF64}(reshape(py"matA"(2, 2/3, 5/9), (2, 2, 2, 2)));

    @test reshape(bw(vec1).tensor, :) ≈ mat_square * vector_linear

    @test inner(vec1, bw(vec1)) ≈ dot(vector_linear, mat_square * vector_linear)

    @test norm(vec1) ≈ norm(vector_linear)
    @test norm(bw(vec1)) ≈ norm(mat_square * vector_linear)

    @test scale(vec1, 2).tensor ≈ vec1.tensor * 2
    @test add(vec1, vec1).tensor ≈ (vec1 + vec1).tensor

    @test add(vec1, vec1, 3, 4).tensor ≈ (3 * vec1 + 4 * vec1).tensor
end