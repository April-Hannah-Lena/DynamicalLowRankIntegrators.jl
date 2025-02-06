using LinearAlgebra
using LinearAlgebra: NoPivot, ColumnNorm
using ApproxFun, FastGaussQuadrature
import ClassicalOrthogonalPolynomials as cl



# set up quadrature

x_grid, x_weights = [-1:2/m_x:1-2/m_x;], ones(m_x)/m_x
v_grid, v_weights = gausslegendre(m_v)

x_grid .= (xlims[2]-xlims[1])/2 .* x_grid# .+ (xlims[2]+xlims[1])/2
v_grid .= (vlims[2]-vlims[1])/2 .* v_grid# .+ (vlims[2]+vlims[1])/2

x_weights .*= (xlims[2]-xlims[1])
v_weights .*= (vlims[2]-vlims[1])/2

@assert sum(x_weights) ≈ xlims[2] - xlims[1]
@assert sum(v_weights) ≈ vlims[2] - vlims[1]

f0v = @. exp(-v_grid^2)

x_gram = Diagonal(x_weights)
sqrt_x_gram = sqrt(x_gram)

v_gram = Diagonal(f0v .* v_weights)
sqrt_v_gram = sqrt(v_gram)



# large basis

Mx = 10r + (isodd(10r) ? 0 : 1)
cfourier = cl.Fourier()
x_basis = cfourier[x_grid,1:Mx]

Mv = 5r
chermite = cl.Hermite()
v_basis = chermite[v_grid,1:Mv]

# normalize basis functions
x_basis[:,1] /= √(2π)
x_basis[:,2:end] /= √(π)

#v_basis ./= Float64.(.√( √(π) .* 2.0.^(0:Mv-1) .* factorial.(big.(0:Mv-1)) ))'
# small error in quarature because we cut off the domain
function basic_gram_schmidt(f, sqrt_gram, pivot::Bool=true)
    QR = qr(sqrt_gram * f, pivot ? ColumnNorm() : NoPivot())
    Q = inv(sqrt_gram) * Matrix(QR.Q)
    R = QR.R
    E = Diagonal(ifelse.(diag(R) .< 0, -1, 1))   # want +1's on diagonal of R
    rmul!(Q, E); lmul!(E, R)
    pivot  &&  ( R *= QR.P' )
    return Q, R
end
v_basis, _R = basic_gram_schmidt(v_basis, sqrt(Diagonal(f0v .* v_weights)), false)




# orthonormalization
# not efficient but easy to implement

@views function gram_schmidt(f, gram, basis, TOL=500*eps(); pivot=true)
    full_coeff_matrix = basis' * gram * f
    cutoff = maximum(CartesianIndices(full_coeff_matrix)) do index
        getindex(full_coeff_matrix, index) < TOL  &&  return 1
        i, _ = Tuple(index)
        return i
    end
    cutoff = max(cutoff, size(f,2))
    coeff_matrix = full_coeff_matrix[1:cutoff,:]
    QR = qr(coeff_matrix, pivot ? ColumnNorm() : NoPivot())
    Q, R = QR
    pivot && ( R *= QR.P' )
    return basis[:,1:cutoff] * Matrix(Q), R
end

@views function gram_schmidt(f, gram, basis, rank::Integer, TOL=500*eps(); pivot=true)
    r = size(f, 2)
    @assert rank < r

    full_coeff_matrix = basis' * gram * f
    @assert full_coeff_matrix[1:rank, 1:rank] ≈ I(rank)
    @assert all( abs.(full_coeff_matrix[rank+1:end, 1:rank]) .< 100*eps() )

    cutoff = maximum(CartesianIndices(full_coeff_matrix)) do index
        getindex(full_coeff_matrix, index) < TOL  &&  return 1
        i, _ = Tuple(index)
        return i
    end
    cutoff = max(cutoff, r + rank + 1)
    coeff_matrix = full_coeff_matrix[1:cutoff,:]

    Q = zeros(cutoff, r)
    R = zeros(r, r)

    R[1:rank, :] .= coeff_matrix[1:rank, :]
    for k in 1:rank
        Q[k,k] = 1
    end

    QR = qr(coeff_matrix[rank+1:end, rank+1:end], pivot ? ColumnNorm() : NoPivot())
    Q[rank+1:end, rank+1:end] .= Matrix(QR.Q)
    R[rank+1:end, rank+1:end] .= QR.R
    pivot  &&  ( R[rank+1:end, rank+1:end] *= QR.P' )

    return basis[:,1:cutoff] * Matrix(Q), R
end

#=
# other orthonormalization methods that were too unstable
function gram_schmidt(f, sqrt_gram, pivot::Bool=true)
    QR = qr(sqrt_gram * f, pivot ? ColumnNorm() : NoPivot())
    Q = inv(sqrt_gram) * Matrix(QR.Q)
    R = QR.R
    E = Diagonal(ifelse.(diag(R) .< 0, -1, 1))   # want +1's on diagonal of R
    rmul!(Q, E); lmul!(E, R)
    pivot  &&  ( R *= QR.P' )
    return Q, R
end

function gram_schmidt(f, sqrt_gram, rank::Integer)
    @assert rank < size(f, 2)
    Q, R = gram_schmidt(f, sqrt_gram, false)
    perm = sortperm( vec(sum(abs2, R[:, rank+1:end], dims=1)), rev=true )
    pivot = [1:rank; rank .+ perm]
    f_pivoted = f[:, pivot]
    Q, R = gram_schmidt(f_pivoted, sqrt_gram, false)
    R = R[:, invperm(pivot)]
    return Q, R
end

function smooth_gram_schmidt(f, sqrt_gram, pad)
    Q, R = gram_schmidt(f, sqrt_gram, true)
    good_rows = vec(sum(R, dims=2)) .> 100*eps()

    n_good = sum(good_rows)
    n_total = size(R,1)
    n_good == n_total  &&  return Q, R
    # @info "rows" n_good n_total
    
    Q̃, R̃ = gram_schmidt( [Q[:,good_rows];; pad], sqrt_gram, false )
    perm = sortperm( diag(R̃)[(n_good+1):end], rev=true )
    replacement = n_good .+ perm[1:n_total-n_good]
    
    Q[:, .!good_rows] .= Q̃[:, replacement]

    return Q, R
end
=#
 