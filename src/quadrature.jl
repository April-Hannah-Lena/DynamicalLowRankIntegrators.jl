using LinearAlgebra, FillArrays
using LinearAlgebra: NoPivot, ColumnNorm
using ApproxFun, FastGaussQuadrature
import ClassicalOrthogonalPolynomials as cl



# set up quadrature

_x_grid, _x_weights = -1:2/m_x:1-2/m_x, Fill(1/m_x, m_x)
const v_grid, v_weights = gausslegendre(m_v)

x_stretch = (xlims[2]-xlims[1])/2
v_stretch = (vlims[2]-vlims[1])/2

const x_grid = x_stretch .* _x_grid# .+ (xlims[2]+xlims[1])/2
v_grid .*= v_stretch# .+ (vlims[2]+vlims[1])/2

const x_weights = _x_weights * 2x_stretch
v_weights .*= v_stretch

@assert sum(x_weights) ≈ 2x_stretch
@assert sum(v_weights) ≈ 2v_stretch

const f0v = @. exp(-v_grid^2)

const x_gram = Diagonal(x_weights)
const sqrt_x_gram = sqrt(x_gram)

const v_gram_unweighted = Diagonal(v_weights)
const v_gram = Diagonal(f0v .* v_weights)
const sqrt_v_gram = sqrt(v_gram)



# large basis
# Mx, Mv must be at least 5*r_max

const Mx = 6r_max + 1 + iseven(5r_max+1)
cfourier = cl.Fourier()
const x_basis = cfourier[x_grid,1:Mx]

const Mv = 6r_max + 1
chermite = cl.Hermite()
const v_basis = chermite[v_grid,1:Mv]

const Mlegendre = min(2Mv, m_v÷2 + 1)
clegendre = cl.Legendre()
const legendre_basis = clegendre[v_grid ./ v_stretch, 1:Mlegendre]

cjacobi = cl.jacobi(1, 1, vlims[1]..vlims[2])
const ∂_legendre_basis = cjacobi[v_grid, 1:Mlegendre]

# normalize basis functions
x_basis[:,1] ./= √(2π)
x_basis[:,2:end] ./= √(π)

v_basis ./= Float64.(.√( √(π) .* 2.0.^(0:Mv-1) .* factorial.(big.(0:Mv-1)) ))'

legendre_basis ./= sqrt.(2 .* v_stretch ./ (2 .* (0:Mlegendre-1) .+ 1))'
# don't normalize basis of legendre derivatives


# orthonormalization
# not efficient but easy to implement

function basic_gram_schmidt!(f, gram)
    R = zeros(eltype(f), size(f,2), size(f,2))
    for j in axes(f, 2)
        for k in axes(f, 2)
            R[k,j] = f[:,k]' * gram * f[:,j]
            if k < j 
                f[:,j] .-= R[k,j] * f[:,k]
            elseif k == j
                R[k,j] = sqrt(R[k,j])
                f[:,j] ./= R[k,j]
            end
        end
    end
    return f, R
end

function gram_schmidt!(f, sqrt_gram, pivot::Bool)
    QR = qr(sqrt_gram * f, pivot ? ColumnNorm() : NoPivot())
    Q = inv(sqrt_gram) * Matrix(QR.Q)
    R = QR.R
    E = Diagonal(ifelse.(diag(R) .< 0, -1, 1))   # want +1's on diagonal of R
    rmul!(Q, E); lmul!(E, R)
    pivot  &&  ( R *= QR.P' )
    #return Q, R
    f .= Q
    return Q, R
end

# small error in quadrature because we cut off the domain. 
# For some reason just doing `gram_schmidt!` makes the first quadr. point
# for each function totally weird. But just doing basic gram schmidt 
# doesn't make the basis orthogonal enough. So we do a cheeky double
gram_schmidt!(basic_gram_schmidt!(v_basis, v_gram)[1], sqrt_v_gram, false)




@views function gram_schmidt(f, gram, basis, TOL=50eps(); pivot=true)
    @assert size(f,2) ≤ size(basis,2)
    full_coeff_matrix = basis' * gram * f

    cutoff = maximum(CartesianIndices(full_coeff_matrix)) do index
        getindex(full_coeff_matrix, index) < TOL  &&  return 1
        i, _ = Tuple(index)
        return i
    end
    cutoff = max(cutoff, size(f,2))
    coeff_matrix = full_coeff_matrix[1:cutoff,:]
    #coeff_matrix[coeff_matrix .< TOL] .= 0

    QR = qr(coeff_matrix, pivot ? ColumnNorm() : NoPivot())
    Q, R = QR
    pivot && ( R *= QR.P' )
    return basis[:,1:cutoff] * Matrix(Q), R
end

@views function gram_schmidt(f, gram, basis, rank::Integer, TOL=50eps(); pivot=true)
    r = size(f, 2)
    @assert rank < r ≤ size(basis, 2)

    full_coeff_matrix = basis' * gram * f

    @assert all( abs.(full_coeff_matrix[1:rank, 1:rank] - I(rank)) .< sqrt(TOL) )
    @assert all( abs.(full_coeff_matrix[rank+1:end, 1:rank]) .< sqrt(TOL) )
    full_coeff_matrix[1:rank, 1:rank] .= I(rank)
    full_coeff_matrix[rank+1:end, 1:rank] .= 0

    cutoff = maximum(CartesianIndices(full_coeff_matrix)) do index
        getindex(full_coeff_matrix, index) < TOL  &&  return 1
        i, _ = Tuple(index)
        return i
    end
    cutoff = max(cutoff, r + rank + 1)
    coeff_matrix = full_coeff_matrix[1:cutoff,:]
    #coeff_matrix[coeff_matrix .< TOL] .= 0

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
function gram_schmidt(f, sqrt_gram, pivot::Bool)
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
 