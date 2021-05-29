module learnsequence
end # module

using Flux
using Zygote
using Random
using StatsBase
Random.seed!(1234)
#using CUDA
using BSON: @save, @load

# Define cell parameters
struct MyRNNCell
    Wax
    Waa
    Wya
    ba
    by
end

# Constructor with random initialization
MyRNNCell(nx::Integer, na::Integer) = MyRNNCell(
    Flux.glorot_normal(na, nx), # Wax
    Flux.glorot_normal(na, na), # Waa
    Flux.glorot_normal(nx, na), # Wya
    Flux.glorot_normal(na),  # ba
    Flux.glorot_normal(nx)  # by
    )

# Implement forward pass
function (m::MyRNNCell)(a_prev, xt)
    a_next = tanh.(m.Waa * a_prev + m.Wax * xt .+ m.ba)
    yt_pred = softmax(m.Wya * a_next .+ m.by)
    return a_next, yt_pred
end

# Exposes for collecting parameters by call params(...) as well as moving them to GPU
Flux.@functor MyRNNCell
# or directly marking fields which are trainable
#Flux.trainable(m::MyRNNCell) = (m.Wax, m.Waa, m.Wya, m.ba, m.by)

# Embedding layer to help model generalize
struct EmbeddingLayer
   W
end
EmbeddingLayer(mf, vs) = EmbeddingLayer(Flux.glorot_normal(mf, vs))
Flux.@functor EmbeddingLayer
(m::EmbeddingLayer)(x) = (m.W * x)

#
# Below define LSTM variant
#
struct MyLSTMCell
    Wf
    Wc
    Wi
    Wo
    Wy
    bf
    bc
    bi
    bo
    by
end

# Constructor with random initialization
MyLSTMCell(nx::Integer, na::Integer) = MyLSTMCell(
    Flux.glorot_normal(na, na+nx),
    Flux.glorot_normal(na, na+nx),
    Flux.glorot_normal(na, na+nx),
    Flux.glorot_normal(na, na+nx),
    Flux.glorot_normal(nx, na),
    Flux.glorot_normal(na),
    Flux.glorot_normal(na),
    Flux.glorot_normal(na),
    Flux.glorot_normal(na),
    Flux.glorot_normal(nx)
    )

# Implement forward pass
function (m::MyLSTMCell)(state, xt)
    a_prev, c_prev = state
    concat = vcat([a_prev,xt]...)
    Γf = σ.(m.Wf*concat.+m.bf)  # Gammaf forget gate
    c_next = tanh.(m.Wc*concat.+m.bc)
    Γi = σ.(m.Wi*concat.+m.bi)  # Gammai update gate
    c = Γf.*c_prev+Γi.*c_next
    Γo = σ.(m.Wo*concat.+m.bo)
    a_next = Γo .* tanh.(c)
    y_pred = softmax(m.Wy*a_next.+m.by)
    return (a_next, c_next), y_pred
end
Flux.@functor MyLSTMCell

function shift(seq)
    return seq[1:end-1], seq[2:end]
end

function shift_batch(example)
    vec = shift.(example)
    return first.(vec), last.(vec)
end

# Add more challanging dataset
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

    # Pad the lines to maximum length
    data = rpad.(lines,maximum(length.(lines)),' ')
    #data = lines
    vocab = collect(unique)
    # Return
    return data, vocab
end

lr = 0.01
batch_size = 100
# opt = Flux.Optimise.Optimiser(Flux.Optimise.ClipValue(1000),Flux.Optimise.ADAM(lr))
opt = Flux.Optimise.ADAM(lr)

data, vocab = load_names()
loader = Flux.Data.DataLoader(data, batchsize=batch_size, partial=false)
nhidden = 64

a_prev_tmp = zeros(nhidden, batch_size)  # hidden x number of examples
c_prev_tmp = zeros(nhidden, batch_size)  # for LSTM variant we also need memory cells

model = Flux.Recur(MyRNNCell(length(vocab), nhidden), a_prev_tmp)
embedding = EmbeddingLayer(length(vocab),length(vocab))
#model = Flux.Recur(MyLSTMCell(length(vocab), nhidden), (a_prev_tmp, c_prev_tmp)) #|> gpu
default_state = copy.(model.state)

ps = params(model)

function loss(x,y)
    #yhat = model.(embedding.(x))
    yhat = model.(x)
    l = sum(Flux.Losses.crossentropy.(yhat,y,agg=sum))
    return l
end

function accuracy(x,y; ignore=' ')
    default_state = copy.(model.state)
    #ŷ = Flux.onecold(Flux.batch(model.(embedding.(x))),vocab)
    ŷ = Flux.onecold(Flux.batch(model.(x)),vocab)
    y  = Flux.onecold(Flux.batch(y),vocab)
    match = (ŷ .== y) .| (y .== ignore)
    return sum(match)/length(y)
end

function wloss(x,y,w)
    """
    Weighted loss which can be used to exclude padded elements
    """
    yhat = model.(x)
    l = 0
    for i in 1:length(x)
        if length(y[i][:,w[i]]) != 0
            l+= Flux.Losses.crossentropy(yhat[i][:,w[i]],y[i][:,w[i]], agg=sum)
        end
    end
    return l
end

for epoch in 1:2000
    losses = []
    accuracies = []
    for example in loader
        xexample, yexample = shift_batch(example)

        # Get weights and into correct dimenions
        #weights = map(s -> [c != '-' for c in s], Flux.batchseq(yexample,' '))

        model.state = copy.(default_state)

        xbatch = Flux.unstack(Flux.batch(map(seq -> float.(Flux.onehotbatch(seq,vocab)),xexample)),2) #|> gpu
        ybatch = Flux.unstack(Flux.batch(map(seq -> float.(Flux.onehotbatch(seq,vocab)),yexample)),2) #|> gpu

        l = 0
        gs = Zygote.gradient(ps) do
            # Foreach timepoint compute enitre batch of activations
            #l = wloss(xbatch, ybatch, weights)
            l = loss(xbatch, ybatch)
            return l
        end
        Flux.Optimise.update!(opt, ps, gs)
        acc = accuracy(xbatch, ybatch)
        push!(losses, l)
        push!(accuracies, acc)
    end
    @show mean(losses), mean(accuracies)
    #break
end
# @save "embedding-acc72.bson" embedding

function sample_rnnseq(ch_init = "<", max_len=50)
    """
    For each time-step sample indices of vocab using probability distribution from softmax
    """
    x = Flux.onehotbatch(ch_init,vocab)
    a_prev = zeros(nhidden)
    seq = []
    for i in 1:max_len
        a = tanh.(model.cell.Wax*x + model.cell.Waa*a_prev .+ model.cell.ba)
        z = model.cell.Wya * a .+ model.cell.by
        y_pred = softmax(z)
        ch = sample(vocab, Weights(y_pred[:]))
        x = Flux.onehot(ch, vocab)
        a_prev = a
        if ch == '>'
            break
        end
        push!(seq, ch)
    end
    return join(seq)
end

sample_rnnseq("S")

function sample_lstmseq(ch_init = "<", max_len=50)
    #x = Flux.squeezebatch(embedding(Flux.onehotbatch(ch_init,vocab)))
    x = Flux.onehotbatch(ch_init,vocab)
    a_prev = zeros(nhidden)
    c_prev = zeros(nhidden)
    seq = []
    for i in 1:max_len
        concat = vcat([a_prev,x]...)
        Γf = σ.(model.cell.Wf*concat.+model.cell.bf)
        c_next = tanh.(model.cell.Wc*concat.+model.cell.bc)
        Γi = σ.(model.cell.Wi*concat.+model.cell.bi)
        c = Γf.*c_prev+Γi.*c_next
        Γo = σ.(model.cell.Wo*concat.+model.cell.bo)
        a_next = Γo .* tanh.(c)
        y_pred = softmax(model.cell.Wy*a_next.+model.cell.by)
        ch = sample(vocab, Weights(y_pred[:]))
        x = Flux.onehot(ch, vocab)
        a_prev = a_next
        if ch == '>'
            break
        end
        push!(seq, ch)
    end
    return join(seq)
end
sample_lstmseq("<")
