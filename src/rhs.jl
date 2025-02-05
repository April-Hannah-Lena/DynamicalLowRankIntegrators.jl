using LinearAlgebra, Einsum, SparseArrays
using ApproxFun
using TupleTools
const del_at, ins_after = TupleTools.deleteat, TupleTools.insertafter

"""
Convert spectral differentiation matrix for 
1-dimensional basis into spectral (partial) 
differentiation matrix for tensor product basis. 
"""
function construct_∂i(basis, ∂_matrix, i)

    og_size = size(basis)[ndims(basis)÷2 + 1 : end]
    og_indices = CartesianIndices(og_size)
    reshaped_indices = LinearIndices(og_size)
    #∂1_matrix = Dict{CartesianIndex{2},Float64}()
    nz_rows, nz_cols, nz_vals = Int64[], Int64[], Float64[]

    for og_ind in og_indices
        linear_row = reshaped_indices[og_ind]
        og_tuple = Tuple(og_ind)
        row_ind = og_tuple[i]
        rest_ind = del_at(og_tuple, i)
        @info "debug outer loop" linear_row og_tuple row_ind rest_ind maxlog=1
        for col_ind in axes(∂_matrix, 2)
            val = ∂_matrix[row_ind, col_ind]
            if val != 0
                full_ind = ins_after(rest_ind, i-1, (col_ind,))
                linear_col = reshaped_indices[CartesianIndex(full_ind...)]
                @info "debug inner loop" full_ind linear_col val maxlog=1
                #∂1_matrix[CartesianIndex(linear_row,linear_col)] = val
                push!(nz_rows, linear_row)
                push!(nz_cols, linear_col)
                push!(nz_vals, val)
            end
        end
    end

    return sparse(nz_rows, nz_cols, nz_vals, Mx^3, Mx^3)

end

# rhs functions

"""
    ∫ f dx

Each column needs to be a function in x. If there are multiple 
columns, return a (row) vector. 
"""
∫dx(f) = vec(sum(f .* x_weights, dims=1))

"""
    ∫ f ⋅ f0v dv 

CAREFUL ABOUT WEIGHT f0v

Each row needs to be a function in v. If there are multiple 
rows, return a (column) vector. 
"""
∫dv(f) = vec(sum(f .* v_weights', dims=2))

_x_domain = PeriodicSegment(xlims...)
x_domain = _x_domain ^ 3
_fourierspace = Fourier(_x_domain)
fourierspace = _fourierspace ^ 3

_hermspace = Hermite()
hermspace = _hermspace ^ 3
_v_domain = domain(_hermspace)
_v_domain = domain(hermspace)

∂_fourier = Derivative(_fourierspace)[1:Mx,1:Mx]
∂²_fourier = (Derivative(_fourierspace)^2)[1:Mx,1:Mx]
∂_hermite = Derivative(_hermspace)[1:Mv,1:Mv]

∂x1 = construct_∂i(_x_basis, ∂_fourier, 1)
∂x2 = construct_∂i(_x_basis, ∂_fourier, 2)
∂x3 = construct_∂i(_x_basis, ∂_fourier, 3)

∂v1 = construct_∂i(_v_basis, ∂_hermite, 1)
∂v2 = construct_∂i(_v_basis, ∂_hermite, 2)
∂v3 = construct_∂i(_v_basis, ∂_hermite, 3)
∂v1 = ∂v1[:,perm]
∂v2 = ∂v2[:,perm]
∂v3 = ∂v3[:,perm]

Δx = construct_∂i(_x_basis, ∂²_fourier, 1)
Δx += construct_∂i(_x_basis, ∂²_fourier, 2)
Δx += construct_∂i(_x_basis, ∂²_fourier, 3)

"""
    ∇ₓ(f::Matrix{Float}) -> Matrix{SVector{3,Float}}

Each column is a function in x
"""
function ∇ₓ(f)  # 3-torus
    coeff_matrix = x_basis' * x_gram * f
    ∂1 = x_basis * ∂x1 * coeff_matrix
    ∂2 = x_basis * ∂x2 * coeff_matrix
    ∂3 = x_basis * ∂x3 * coeff_matrix

    ∇ₓf = [SA_F64[∂1u,∂2u,∂3u] for (∂1u,∂2u,∂3u) in zip(∂1,∂2,∂3)]
end

"""
    ∇ᵥ(f::Matrix{Float}) -> Matrix{SVector{3,Float}}

Each column is a function in v
"""
function ∇ᵥ(f)  # 1-dimensional real domain
    coeff_matrix = v_basis' * v_gram * f 
    ∂1 = v_basis * ∂v1 * coeff_matrix
    ∂2 = v_basis * ∂v2 * coeff_matrix
    ∂3 = v_basis * ∂v3 * coeff_matrix

    ∇ᵥf = [SA_F64[∂1u,∂2u,∂3u] for (∂1u,∂2u,∂3u) in zip(∂1,∂2,∂3)]
end

"""
    E(X*S*V' :: Matrix{Float}) -> Vector{SVector{3,Float}}

Energy as a (vector-valued) function of x
"""
function E(XSV)
    ∫dvf = ∫dv(XSV)
    coeff_vec = x_basis' * x_gram * vec( 1/(2π)^3 .- ∫dvf )
    
    coeff_grid = reshape(coeff_vec, (Mx,Mx,Mx,:))
    coeff_grid[1,:,:,:] .= 0
    coeff_grid[:,1,:,:] .= 0
    coeff_grid[:,:,1,:] .= 0

    ϕ_coeffs = -Δx \ coeff_vec

    ∂1 = x_basis * (-∂x1) * ϕ_coeffs
    ∂2 = x_basis * (-∂x2) * ϕ_coeffs
    ∂3 = x_basis * (-∂x3) * ϕ_coeffs

    minus∇ₓϕ = [SA_F64[∂1u,∂2u,∂3u] for (∂1u,∂2u,∂3u) in zip(∂1,∂2,∂3)] 
end


#∇ᵥf0v = -2 .* v_grid .* f0v
RHS(XSV) = E(XSV) .⋅ ((-2 .* v_grid' .* XSV  +  ∇ᵥ(XSV')') .* f0v')  -  v_grid' .⋅ ∇ₓ(XSV .* f0v')

mass(XSV) = ∫dx(∫dv(XSV))
momentum(XSV) = ∫dx(∫dv(XSV .* v_grid'))
energy(XSV) = ∫dx( ∫dv( XSV .* (map(norm, v_grid) .^ 2)' )  +  map(norm, E(XSV)) .^ 2 ) / 2
Lp(XSV, p) = ( ∫dx(∫dv( abs.(XSV).^p  .*  f0v.^(p-1) )) ) .^ (1/p)
entropy(XSV) = -∫dx(∫dv( XSV .* log.(max.(XSV .* f0v', eps())) ))





