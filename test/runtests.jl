using SafeTestsets: @safetestset
using Suppressor: @suppress

# check for filtered groups
# either via `--group=ALL` or through ENV["GROUP"]
const pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = uppercase(
  if isnothing(arg_id)
    get(ENV, "GROUP", "ALL")
  else
    only(match(pat, ARGS[arg_id]).captures)
  end,
)

function istestfile(filename)
  return isfile(filename) &&
         endswith(filename, ".jl") &&
         startswith(basename(filename), "test")
end

@time begin
  # tests in groups based on folder structure
  for testgroup in filter(isdir, readdir(@__DIR__))
    if GROUP == "ALL" || GROUP == uppercase(testgroup)
      for file in filter(istestfile, readdir(joinpath(@__DIR__, testgroup); join=true))
        @eval @safetestset $file begin
          include($file)
        end
      end
    end
  end

  # single files in top folder
  for file in filter(istestfile, readdir(@__DIR__))
    (file == basename(@__FILE__)) && continue
    @eval @safetestset $file begin
      include($file)
    end
  end

  # test examples
  examplepath = joinpath(@__DIR__, "..", "examples")
  for file in filter(endswith(".jl"), readdir(examplepath; join=true))
    @suppress @eval @safetestset $file begin
      include($file)
    end
  end
end
