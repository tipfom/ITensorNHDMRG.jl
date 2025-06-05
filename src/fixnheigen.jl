using ITensors, ITensors.NDTensors, ITensors.NDTensors.Expose

function LinearAlgebra.eigen(
  T::ITensors.DenseTensor{ElT,2,IndsT};
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
) where {ElT<:Union{Real,Complex},IndsT}
  matrixT = matrix(T)
  if any(!isfinite, matrixT)
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
      ),
    )
  end
  DM, VM = eigen(expose(matrixT))

  # Sort by largest to smallest eigenvalues
  p = sortperm(DM; by=abs, rev = true)
  DM = DM[p]
  VM = VM[:,p]

  if any(!isnothing, (maxdim, cutoff))
    DM, truncerr, _ = truncate!!(
      DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
    dD = length(DM)
    truncerr = 0.0
  end
  spec = Spectrum(abs.(DM), truncerr)

  i1, i2 = inds(T)

  # Make the new indices to go onto D and V
  l = typeof(i1)(dD)
  r = dag(sim(l))
  Dinds = (l, r)
  Vinds = (dag(i2), r)
  D = complex(tensor(Diag(DM), Dinds))
  V = complex(tensor(Dense(vec(VM)), Vinds))
  return D, V, spec
end

