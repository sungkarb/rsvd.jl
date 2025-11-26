using Images
using ImageIO
using LinearAlgebra
using ColorVectorSpace
include("main.jl")  

function svd_compress_channel(A, k, solver)
    U, S, Vt = solver(A, k)
    return U * Diagonal(S) * Vt
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 2
        println("Usage input_image rank")
        exit(1)
    end

    # load image
    filename = ARGS[1]
    k = parse(Int, ARGS[2])
    img = load(filename) |> float

    for method in [svdk, randsvd]
        if eltype(img) <: AbstractGray
            A = float.(img)
            Arec = svd_compress_channel(A, k, method)

        elseif eltype(img) <: Colorant
            # extract channels as Float64 matrices
            R = float.(channelview(img)[1, :, :])
            G = float.(channelview(img)[2, :, :])
            B = float.(channelview(img)[3, :, :])

            Rc = svd_compress_channel(R, k, method)
            Gc = svd_compress_channel(G, k, method)
            Bc = svd_compress_channel(B, k, method)

            # combine channels to 3×m×n
            channels = stack((Rc, Gc, Bc); dims=1)

            # convert to RGB image
            Arec = colorview(RGB, channels)

        else
            # numeric array
            Arec = svd_compress_channel(img, k, method)
        end

        Arec = clamp01.(Arec)
        base_name = splitext(basename(filename))[1]
        method_name = string(nameof(method))
        output_filename = "$(base_name)_$(method_name)_$(k).png"
        save(output_filename, Arec)
    end
end
