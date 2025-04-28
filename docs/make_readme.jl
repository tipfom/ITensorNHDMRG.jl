using Literate: Literate
using ITensorNHDMRG: ITensorNHDMRG

Literate.markdown(
  joinpath(pkgdir(ITensorNHDMRG), "examples", "README.jl"),
  joinpath(pkgdir(ITensorNHDMRG));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
