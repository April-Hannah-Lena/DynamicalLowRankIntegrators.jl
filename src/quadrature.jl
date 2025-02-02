using LinearAlgebra, FillArrays, StaticArrays
using .LazyTensorProduct
using LinearAlgebra: NoPivot, ColumnNorm
using ApproxFun, FastGaussQuadrature
import ClassicalOrthogonalPolynomials as cl



# set up quadrature

_x_grid = [-1:2/m_x:1-2/m_x;]
_x_grid .= (xlims[2]-xlims[1])/2 .* _x_grid# .+ (xlims[2]+xlims[1])/2

x_grid = vec(TensorProduct(_x_grid, (x1,x2,x3)->SVector{3,Float64}(x1,x2,x3), 3))
x_weights = Fill( ( (xlims[2]-xlims[1]) / m_x )^3, size(x_grid) )

_v_grid, _v_weights = gausshermite(m_v)
#_v_grid .= (vlims[2]-vlims[1])/2 .* _v_grid# .+ (vlims[2]+vlims[1])/2
#_v_weights *= (vlims[2]-vlims[1])/2

v_grid = vec(TensorProduct(_v_grid, (x1,x2,x3)->SVector{3,Float64}(x1,x2,x3), 3))
v_weights = vec(TensorProduct(_v_weights, (w1,w2,w3)->w1*w2*w3, 3))

#@assert sum(x_weights) ≈ xlims[2] - xlims[1]
#@assert sum(v_weights) ≈ vlims[2] - vlims[1]

f0v = @. exp(-norm(v_grid)^2)

x_gram = Diagonal(x_weights)
sqrt_x_gram = sqrt(x_gram)

v_gram = Diagonal(v_weights)
sqrt_v_gram = sqrt(v_gram)



# large basis

Mx = 2r
cfourier = cl.Fourier()[_x_grid, 1:Mx]
cfourier[:,1] /= √(2π)
cfourier[:,2:end] /= √(π)
_x_basis = Array(DoubleTensorProduct(cfourier, (f1,f2,f3)->f1*f2*f3, 3));
x_basis = reshape(_x_basis, (m_x^3,Mx^3))

Mv = 2r
chermite = cl.Hermite()[_v_grid, 1:Mv]
chermite ./= Float64.(.√( √(π) .* 2.0.^(0:Mv-1) .* factorial.(big.(0:Mv-1)) ))'
_v_basis = Array(DoubleTensorProduct(chermite, (f1,f2,f3)->f1*f2*f3, 3));
v_basis = reshape(_v_basis, (m_v^3,Mv^3))

perm = unique([1; 2; Mv + 1; Mv^2 + 1; 3; 2 * Mv + 1; 2 * Mv^2 + 1; axes(v_basis, 2);])
v_basis = v_basis[:,perm]

# v -> v₁  ==  v_basis[:,:,:,2,1,1]  =^=  v_basis[:, 2]
# v -> v₂  ==  v_basis[:,:,:,1,2,1]  =^=  v_basis[:, Mv + 1]
# v -> v₃  ==  v_basis[:,:,:,1,1,2]  =^=  v_basis[:, Mv^2 + 1]
# v -> |v|²  ==  v_basis[:,:,:,3,1,1] + v_basis[:,:,:,1,3,1] + v_basis[:,:,:,1,1,3]
# v -> v₁²  ==  v_basis[:,:,:,3,1,1]  =^=  v_basis[:, 3]
# v -> v₂²  ==  v_basis[:,:,:,1,3,1]  =^=  v_basis[:, 2 * Mv + 1]
# v -> v₃²  ==  v_basis[:,:,:,1,1,3]  =^=  v_basis[:, 2 * Mv^2 + 1]



function basic_gram_schmidt(f, sqrt_gram, pivot::Bool=true)
    QR = qr(sqrt_gram * f, pivot ? ColumnNorm() : NoPivot())
    Q = inv(sqrt_gram) * Matrix(QR.Q)
    R = QR.R
    E = Diagonal(ifelse.(diag(R) .< 0, -1, 1))   # want +1's on diagonal of R
    rmul!(Q, E); lmul!(E, R)
    pivot  &&  ( R *= QR.P' )
    return Q, R
end
#v_basis, _R = basic_gram_schmidt(v_basis, sqrt(Diagonal(f0v .* v_weights)), false)




# orthonormalization
# not efficient but easy to implement

"""
Gram Schmidt process using a (large) basis as 
seed from which orthogonal functions are created. 
"""
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

"""
Gram Schmidt, maintaining the first `rank` columns. 
"""
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
 