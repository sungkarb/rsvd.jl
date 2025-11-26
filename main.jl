using LinearAlgebra
using Random
using Printf

function random_range_finder(A::AbstractMatrix, k::Int64; p::Int64=5, q::Int64=0)
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
        Y_tilda = A' * Q
        Q_tilda, _ = qr(Y_tilda)
        Y = A * Q_tilda
        Q, _ = qr(Y)
    end

    return Matrix(Q)
end

function randsvd(A::AbstractMatrix, k::Int64; p::Int64=5, q::Int64=0)
    Q = random_range_finder(A, k; p=p, q=q)
    B = Q' * A
    F = svd(B)
    U = Q * F.U
    # Return only the first k components
    return U[:, 1:k], F.S[1:k], F.Vt[1:k, :]
end

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
