using LinearAlgebra
using Random
using Printf

"""
    random_range_finder(A::AbstractMatrix, k::Int64; p::Int64=5, q::Int64=2)

Compute an orthonormal basis for the range of matrix `A` using randomized sampling.

This function implements the randomized range finder algorithm, which constructs an
approximate orthonormal basis for the column space of `A` by using random projections
with optional power iterations

# Arguments
- `A::AbstractMatrix`: Input matrix of size m × n
- `k::Int64`: Target rank for the approximation
- `p::Int64=5`: Oversampling parameter (default: 5)
- `q::Int64=2`: Number of power iterations (default: 1)

# Returns
- `Matrix`: Orthonormal matrix Q of size m × l, where l = min(n, k + p)

# Throws
- `ArgumentError`: If k > n (number of columns)

# Algorithm
1. Generate random Gaussian matrix Omega of size n × l
2. Compute Y = A * Omega
3. Perform QR decomposition to get orthonormal basis Q
4. Apply q power iterations to improve accuracy
5. Return the orthonormal basis Q
"""
function random_range_finder(A::AbstractMatrix, k::Int64; p::Int64=5, q::Int64=1)
    m, n = size(A)
    if k > n
        throw(ArgumentError("k should be less than number of columns"))
    end

    l = min(n, k + p)
    Omega = randn(n, l)
    Y = A * Omega
    Q, _ = qr(Y)

    ## Power iterations
    for i in 1:q
        Y = A' * Q
        Q, _ = qr(Y)
        Y = A * Q
        Q, _ = qr(Y)
    end

    return Matrix(Q)
end

"""
    randsvd(A::AbstractMatrix, k::Int64; p::Int64=5, q::Int64=1)

Compute a randomized truncated Singular Value Decomposition (SVD) of matrix `A`.

This function computes an approximate rank-k SVD using randomization, which allows
for significant performance boost compared to deterministic SVD methods for large
matrices while also keeping good accuracy

# Arguments
- `A::AbstractMatrix`: Input matrix of size m × n
- `k::Int64`: Target rank (number of singular values/vectors to compute)
- `p::Int64=5`: Oversampling parameter for range finder (default: 5)
- `q::Int64=1`: Number of power iterations for range finder (default: 1)

# Returns
- `U`: Left singular vectors, matrix of size m × k
- `S`: Singular values, vector of length k
- `Vt`: Right singular vectors (transposed), matrix of size k × n
"""
function randsvd(A::AbstractMatrix, k::Int64; p::Int64=5, q::Int64=1)
    Q = random_range_finder(A, k; p=p, q=q)
    B = Q' * A
    F = svd(B)
    U = Q * F.U
    # Return only the first k components
    return U[:, 1:k], F.S[1:k], F.Vt[1:k, :]
end

"""
    svdk(A, k)

Compute the truncated rank-k Singular Value Decomposition (SVD) using deterministic methods.

This function computes the full SVD and then truncates it to rank k, providing
the best possible rank-k approximation of the input matrix.

# Arguments
- `A`: Input matrix
- `k`: Target rank (number of singular values/vectors to return)

# Returns
- `U`: Left singular vectors, first k columns
- `S`: First k singular values
- `Vt`: Right singular vectors (transposed), first k rows

"""
function svdk(A, k)
    U, S, V = svd(A)  # Name it V, not Vt!
    return U[:, 1:k], S[1:k], V[:, 1:k]'  # Transpose V to get Vt
end

"""
    unit_test_small()

Run a small unit test comparing randomized SVD with deterministic SVD.

This test draws a random 80x60 matrix and computes its rank-10 best approximation
using both randomised SVD and deterministic SVD. To assess results, it compares 
Frobenius norm errors of both approximations.

# Returns
- Named tuple with fields:
  - `err_rand`: Frobenius norm error of randomized SVD approximation
  - `err_det`: Frobenius norm error of deterministic SVD approximation

# Notes
Uses a fixed random seed (1234) for reproducibility.
"""
function unit_test_small()
    Random.seed!(1234)
    m, n = 80, 60
    A = randn(m, n)
    k = 10
    U_r, S_r, Vt_r = randsvd(A, k; p=10, q=2)
    # deterministic SVD best rank-k approximation
    F = svd(A)
    U_d = F.U[:, 1:k]
    S_d = F.S[1:k]
    Vt_d = F.Vt[1:k, :]
    # compare Frobenius error of approximations
    A_r = U_r * Diagonal(S_r) * Vt_r
    A_d = U_d * Diagonal(S_d) * Vt_d
    err_rand = norm(A - A_r)
    err_det  = norm(A - A_d)
    return (err_rand=err_rand, err_det=err_det)
end

"""
    synthetic_experiment(; m=500, n=400, ktrue=20, noise_level=1e-10)

Run a synthetic experiment comparing randomized SVD with deterministic SVD across multiple ranks.

This function generates a synthetic low-rank matrix with specific structure and
adds noise, then compares the accuracy of deterministic and randomised SVD for
various target ranks.

# Keyword Arguments
- `m=500`: Number of rows in the test matrix
- `n=400`: Number of columns in the test matrix
- `ktrue=20`: True rank of the underlying low-rank matrix
- `noise_level=1e-10`: Standard deviation of Gaussian noise added to the matrix

# Returns
- Vector of named tuples, each containing:
  - `k`: Target rank used for approximation
  - `err_rand`: Operator (spectral) norm error of randomized SVD approximation
  - `err_det`: Operator (spectral) norm error of deterministic SVD approximation

# Notes
- Uses fixed random seed (42) for reproducibility
- Tests ranks: [5, 10, 15, 20, 30]
- Uses oversampling parameter p=10 and q=2 power iterations for randomized SVD
"""
function synthetic_experiment(; m=500, n=400, ktrue=20, noise_level=1e-10)
    Random.seed!(42)
    # build low-rank matrix
    U = qr(randn(m, ktrue)).Q
    V = qr(randn(n, ktrue)).Q
    s = exp.(-collect(0:ktrue-1) / 5)  # decaying singular values
    A0 = U[:, 1:ktrue] * Diagonal(s) * (V[:, 1:ktrue])'
    A = A0 + noise_level * randn(m, n)
    # compute errors of randomized SVD vs true best truncation (deterministic SVD)
    ranks = [5, 10, 15, 20, 30]
    results = []
    F = svd(A)
    for k in ranks
        U_r, S_r, Vt_r = randsvd(A, k; p=10, q=2)
        A_r = U_r * Diagonal(S_r) * Vt_r
        A_d = F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.Vt[1:k, :]
        push!(results, (k=k,
                        err_rand = norm(A - A_r, 2),   # operator norm
                        err_det  = norm(A - A_d, 2)))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running small unit test...")
    t = unit_test_small()
    println("Frobenius error randomized: ", t.err_rand)
    println("Frobenius error deterministic: ", t.err_det)
    @printf("Relative error: %.2f%%\n", 100 * (t.err_rand - t.err_det) / t.err_det)

    println("\nRunning synthetic experiment (operator norm errors)...")
    res = synthetic_experiment()
    for r in res
        rel_err = 100 * (r.err_rand - r.err_det) / r.err_det
        @printf("k=%2d | rand_err=%.6f | det_err=%.6f | rel_err=%.2f%%\n",
                r.k, r.err_rand, r.err_det, rel_err)
    end
end
