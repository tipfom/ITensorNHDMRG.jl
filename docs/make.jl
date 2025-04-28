using ITensorNHDMRG: ITensorNHDMRG
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  ITensorNHDMRG, :DocTestSetup, :(using ITensorNHDMRG); recursive=true
)

include("make_index.jl")

makedocs(;
  modules=[ITensorNHDMRG],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="ITensorNHDMRG.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/ITensorNHDMRG.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(;
  repo="github.com/ITensor/ITensorNHDMRG.jl", devbranch="main", push_preview=true
)
