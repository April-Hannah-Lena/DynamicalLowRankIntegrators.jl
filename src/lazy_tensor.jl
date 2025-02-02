module LazyTensorProduct

export TensorProduct, DoubleTensorProduct

import Base: size, eltype, similar, getindex, show

struct TensorProduct{T,N,F,A<:AbstractArray{T,1}} <: AbstractArray{T,N}
    basearray :: A
    reduction :: F
    ndims :: Val{N}
end

TensorProduct(basearray, reduction, ndims) = TensorProduct(basearray, reduction, Val(ndims))

size(A::TensorProduct{T,N}) where {T,N} = ntuple(_->length(A.basearray), N)
eltype(A::TensorProduct{T,N,F}) where {T,N,F} = typeof(A[ones(Int, N)...])
similar(A::TensorProduct{T,N,F}) where {T,N,F} = Array{eltype(A),N}(undef, size(A))

function getindex(A::TensorProduct{T,N,F}, I::Vararg{Int,N}) where {T,N,F}
    base = (A.basearray[i] for i in Tuple(I))
    return A.reduction(base...)
end

function show(io::IO, A::TensorProduct)
    print(io, join(size(A), " × "), " - element ", typeof(A), " with eltype ", eltype(A))
end

function show(io::IO, ::MIME"text/plain", A::TensorProduct)
    print(io, join(size(A), " × "), " - element ", typeof(A), " with eltype ", eltype(A))
end


struct DoubleTensorProduct{T,N,F,A<:AbstractArray{T,2}} <: AbstractArray{T,N}
    basearray :: A
    reduction :: F
    twondims :: Val{N}
end

function DoubleTensorProduct(basearray, reduction, ndims)
    DoubleTensorProduct(basearray, reduction, Val(2*ndims))
end

size(A::DoubleTensorProduct{T,N}) where {T,N} = (ntuple(_->size(A.basearray, 1), N÷2)..., ntuple(_->size(A.basearray, 2), N÷2)...)
eltype(A::DoubleTensorProduct{T,N,F}) where {T,N,F} = typeof(A[ones(Int, N)...])
similar(A::DoubleTensorProduct{T,N,F}) where {T,N,F} = Array{eltype(A),N}(undef, size(A))

function getindex(A::DoubleTensorProduct{T,N,F}, I::Vararg{Int,N}) where {T,N,F}
    base_ind = Tuple(I)[1 : N÷2]
    ext_ind = Tuple(I)[N÷2 + 1 : N]
    base = (A.basearray[i, j] for (i,j) in zip(base_ind, ext_ind))
    return A.reduction(base...)
end

function show(io::IO, A::DoubleTensorProduct)
    print(io, join(size(A), " × "), " - element ", typeof(A), " with eltype ", eltype(A))
end

function show(io::IO, ::MIME"text/plain", A::DoubleTensorProduct)
    print(io, join(size(A), " × "), " - element ", typeof(A), " with eltype ", eltype(A))
end

end
