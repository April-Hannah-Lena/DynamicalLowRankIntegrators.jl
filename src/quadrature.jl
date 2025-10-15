using LinearAlgebra, FillArrays
using LinearAlgebra: NoPivot, ColumnNorm
using ApproxFun, FastGaussQuadrature
import ClassicalOrthogonalPolynomials as cl



# set up quadrature

# for x we use fourier => uniform grid and weights
# for v we use Hermite => typical choice would be Hermite quadrature but 
#                         [Trefethen, 2020: Exactness of Quadrature Formulas, Section 5]
#                         concludes it is far more efficient to simply use 
#                         Gauss-Legendre quadrature and reweight by a Gaussian

_x_grid, _x_weights = -1:2/m_x:1-2/m_x, Fill(1/m_x, m_x)
const v_grid, v_weights = gausslegendre(m_v)

x_stretch = (xlims[2]-xlims[1])/2   # length of the domain
v_stretch = (vlims[2]-vlims[1])/2

const x_grid = x_stretch .* _x_grid   # integrate over xlims, not [-1, 1]
v_grid .*= v_stretch                  # same for vlims

perm = sortperm(v_grid, by=abs, rev=true)  # numerically more stable to 
v_grid .= v_grid[perm]                     # sum in order of magnitude
v_weights .= v_weights[perm]
iperm = invperm(perm)

const x_weights = _x_weights * 2x_stretch   # ∫ 1 dx = length of domain
v_weights .*= v_stretch                     # same for v

@assert sum(x_weights) ≈ 2x_stretch
@assert sum(v_weights) ≈ 2v_stretch

const f0v = @. exp(-v_grid^2)       # Gauss weight

const x_gram = Diagonal(x_weights)      # should be renamed since it's not a gram
const sqrt_x_gram = sqrt(x_gram)

const v_gram_unweighted = Diagonal(v_weights)
const v_gram = Diagonal(f0v .* v_weights)
const sqrt_v_gram = sqrt(v_gram)



# construct a large basis for spectral differentiation & QR
# Mx, Mv = basis sizes, must be at least 5*r_max

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

# normalize basis functions
x_basis_norms = √(π) * ones(Mx)
x_basis_norms[1] *= √(2)
@assert diag(x_basis' * x_gram * x_basis) ≈ x_basis_norms
x_basis ./= x_basis_norms'

legendre_basis_norms = sqrt.(2 .* v_stretch ./ (2 .* (0:Mlegendre-1) .+ 1))
@assert diag(legendre_basis' * v_gram_unweighted * legendre_basis) ≈ legendre_basis_norms
legendre_basis ./= legendre_basis_norms'


# orthonormalization
# not efficient but easy to implement
#=
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
=#
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

# orthonormalize v basis
_, R = gram_schmidt!(v_basis, sqrt_v_gram, false)
const v_basis_norms = diag(R)
# In theory this would = √( √(π)  .*  2.0 .^ (0:Mv-1)  .*  factorial.(big.(0:Mv-1)) )
# but the cutoff causes us to lose (up to) 60% of a 
# function's mass (in the high end, lower orders are integrated pretty much exact)

# v basis is now orthonormal, but we need v_basis_norms for 
# the spectral differentiation matrix




@views function gram_schmidt(f, gram, basis, TOL=50eps(); pivot=true)
    @assert size(f,2) ≤ size(basis,2)
    full_coeff_matrix = basis' * gram * f
    
    projection_error = diag( f' * gram * f  -  full_coeff_matrix' * full_coeff_matrix )
    @assert all(projection_error .< TOL)

    cutoff = maximum(CartesianIndices(full_coeff_matrix)) do index
        full_coeff_matrix[index] < TOL  &&  return 1
        i, _ = Tuple(index)
        return i
    end
    cutoff = max(cutoff, size(f,2))
    coeff_matrix = full_coeff_matrix[1:cutoff,:]
    #coeff_matrix[abs.(coeff_matrix) .< TOL] .= 0

    QR = qr(coeff_matrix, pivot ? ColumnNorm() : NoPivot())
    Q, R = QR
    pivot && ( R *= QR.P' )
    return basis[:,1:cutoff] * Matrix(Q), R
end

@views function gram_schmidt(f, gram, basis, rank::Integer, TOL=50eps(); pivot=false)
    @assert size(f,2) ≤ size(basis,2)
    full_coeff_matrix = basis' * gram * f
    
    projection_error = diag( f' * gram * f  -  full_coeff_matrix' * full_coeff_matrix )
    @assert all(projection_error .< TOL)

    should_be_small_1 = full_coeff_matrix[1:rank, 1:rank] - I(rank)
    should_be_small_2 = full_coeff_matrix[rank+1:end, 1:rank]
    @debug "quadrature error" should_be_small_1 should_be_small_2
    
    @assert all( abs.(should_be_small_1) .< TOL )
    @assert all( abs.(should_be_small_2) .< TOL )
    full_coeff_matrix[1:rank, 1:rank] .= I(rank)
    full_coeff_matrix[rank+1:end, 1:rank] .= 0

    cutoff = maximum(CartesianIndices(full_coeff_matrix)) do index
        getindex(full_coeff_matrix, index) < TOL  &&  return 1
        i, _ = Tuple(index)
        return i
    end
    cutoff = max(cutoff, size(f,2))
    coeff_matrix = full_coeff_matrix[1:cutoff,:]
    #coeff_matrix[abs.(coeff_matrix) .< eps()] .= 0

    QR = qr(coeff_matrix, pivot ? ColumnNorm() : NoPivot())
    Q, R = QR
    pivot && ( R *= QR.P' )
    return basis[:,1:cutoff] * Matrix(Q), R
end


#=
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
=#
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
 