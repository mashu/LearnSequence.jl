using Flux, Flux.Zygote, Flux.Optimise
using LinearAlgebra
using Flux: @functor
using StatsBase
struct Embedding{T}
    W::T
end

Embedding(vocab_size::Integer, embedding_size::Integer) = Embedding(randn(Float32, embedding_size, vocab_size))

@functor Embedding

(m::Embedding)(x) = m.W[:, x]

struct DotProduct{T}
    fᵤ::T
    fᵥ::T
end

@functor DotProduct

(m::DotProduct)(x::Tuple{Integer,Integer}) = m.fᵤ(x[1]) ⋅ m.fᵥ(x[2])

(m::DotProduct)(x,y) = sum(m.fᵤ(x) .* m.fᵥ(y))

function load_names(path="dinos.txt")
    lines = []
    unique = Set([' '])
    open(path,"r") do f
        for line in eachline(f)
            line = '<'*line*'>'
            push!(lines,line)
            for c in line
                push!(unique,c)
            end
        end
    end
    vocab = collect(unique)
    # Return
    return lines, vocab
end

data, vocab = load_names()
vocab_length = length(vocab)
indices = Dict(zip(vocab,1:vocab_length))
batch_size = 100
loader = Flux.Data.DataLoader(data, batchsize=batch_size, partial=false,shuffle=true)
embedding_size = 16

encodder = Embedding(vocab_length, embedding_size)
decodder = Embedding(vocab_length, embedding_size)
model = DotProduct(encodder, decodder)
model_zip(x::Integer, y::Integer) = model((x, y))

opt = Flux.Optimiser(Flux.Optimise.ClipNorm(1000), Flux.Optimise.ADAM(1e-3))

function loss(target_idx,
    context_idx,
    neg_idx)
    l1 = - sum(log.(sigmoid.(model(target_idx, context_idx))))
    l2 = - sum(log.(sigmoid.(-model(target_idx, neg_idx))))
    l1 + l2
end

function sample_negative(t, k)
    """
    Sampels k negative indicies without replacement from the vocab_legth
    Excluded indices is t token itself
    """
    sample([i for i in 1:vocab_length if (i != t)],k,replace=false)
end

k = 25 # Half are randoms
for epoch in 1:100
    losses = []
    for example in loader
        for s in example
            for t in 1:(length(s)-1)
                # context target pair
                context_idx,target_idx = map(c -> indices[c], collect(s[t:t+1]))
                neg_idx = Flux.batch(sample_negative(t, k))

                ps = params(model)
                l = 0
                gs = gradient(ps) do
                  l = loss(target_idx, context_idx, neg_idx)
                end
                push!(losses, l)
                Flux.Optimise.update!(opt, ps, gs)
            end
        end
    end
    @show mean(losses)
end

M = fit(PCA, encodder.W; maxoutdim=2)
Y = transform(M, encodder.W)
using Plots
scatter!(Y[1,:], Y[2,:])
