using LinearAlgebra
using Random
using Distributions

"""
    rsvd(A, e)

Compute SVD decomposition of matrix A with given precision of accuracy e

See: Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011). *Finding structure with randomness:
Probabilistic algorithms for matrix decompositions.*
"""
function rsvd(A::AbstractMatrix, e::Float64=0.005)
    m, n = size(A)
    I = Diagonal(ones(m))

    ## Iteration 1 when Q0 is empty
    w = randn(n, 1)
    cur_y = A * w
    cur_qtilda = cur_y
    cur_q = cur_qtilda / norm(cur_qtilda)
    cur_Q = cur_q

    ## Subsequent iterations with updates to Qi with error checking
    error = norm((I - cur_Q * transpose(cur_Q)) * A)
    while error > e
        w = randn(n, 1)
        cur_y = A * w
        cur_qtilda = (I - cur_Q * transpose(cur_Q)) * cur_y
        cur_q = cur_qtilda / norm(cur_qtilda)
        cur_Q = hcat(cur_Q, cur_q) # Concatenate new vector to Q
    end
    

    B = transpose(cur_Q) * A
    U_tilda, s, Vt = svd(B)
    U = cur_Q * U_tilda
    S = Diagonal(s)
    
    error = norm((I - cur_Q * transpose(cur_Q)) * A)
    return U, S, Vt, error
end

A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
U, S, Vt, error = rsvd(A)
println("My U: {$U}")
println("My S: {$S}")
println("My Vt: {$Vt}\n")


F = svd(A)
println("Algo U: {$F.U}")
println("Algo S: {$F.S}")
println("Algo Vt: {$F.Vt")
