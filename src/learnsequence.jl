module learnsequence
end # module

using Flux
using Zygote
using Random
using StatsBase
Random.seed!(1234)

# Define cell parameters
struct MyRNNCell{tWax<:AbstractArray,tWaa<:AbstractArray,tWya<:AbstractArray,
                 tba<:AbstractVector,tby<:AbstractVector}
    Wax::tWax
    Waa::tWaa
    Wya::tWya
    ba::tba
    by::tby
end

# Constructor with random initialization
MyRNNCell(nx::Integer, na::Integer) = MyRNNCell(
    randn(na, nx), # Wax
    randn(na, na), # Waa
    randn(nx, na), # Wya
    randn(na),  # ba
    randn(nx)  # by
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

#
# Below define LSTM variant
#

# Define cell parameters
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
    randn(na, na+nx),
    randn(na, na+nx),
    randn(na, na+nx),
    randn(na, na+nx),
    randn(nx, na),
    randn(na),
    randn(na),
    randn(na),
    randn(na),
    randn(nx)
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

# Simulate data
vocab = [c for c in "<ABCDEFGHIJKLMNOPQRSTUVWXYZ>"]
function kmers(k=5)
    return [vocab[i:(i+k)-1] for i in 1:length(vocab)-k+1]
end

function sample_kmers(k=6, m=100)
    examples = []
    for i in 1:m
        push!(examples,Random.shuffle(kmers(k)))
    end
    return vcat(examples...)
end

function shift(seq)
    return seq[1:end-1], seq[2:end]
end

function shift_batch(example)
    vec = shift.(example)
    return first.(vec), last.(vec)
end

function accuracy(x,y)
    ŷ = Flux.onecold(Flux.batch(model.(x)),vocab)
    y  = Flux.onecold(Flux.batch(y),vocab)
    return sum(ŷ .== y)/length(y)
end

lr = 0.01
batch_size = 100
opt = Flux.Optimise.Optimiser(Flux.Optimise.ClipValue(1000),Flux.Optimise.ADAM(lr))

data = sample_kmers(6, batch_size)
loader = Flux.Data.DataLoader(data, batchsize=batch_size,partial=false)

nhidden = 32
a_prev_tmp = randn(nhidden, batch_size)  # hidden x number of examples
c_prev_tmp = randn(nhidden, batch_size)  # for LSTM variant we also need memory cells

#model = Flux.Recur(MyRNNCell(length(vocab), nhidden), a_prev_tmp)
model = Flux.Recur(MyLSTMCell(length(vocab), nhidden), (a_prev_tmp, c_prev_tmp))
default_state = copy.(model.state)

ps = params(model)

function loss(x,y)
    l = sum(Flux.Losses.crossentropy.(model.(x),y))
    return l
end

for epoch in 1:2000
    losses = []
    accuracies = []
    for example in loader
        xexample, yexample = shift_batch(example)

        xbatch = Flux.unstack(Flux.batch(map(seq -> float.(Flux.onehotbatch(seq,vocab)),xexample)),2)
        ybatch = Flux.unstack(Flux.batch(map(seq -> float.(Flux.onehotbatch(seq,vocab)),yexample)),2)
        model.state = copy.(default_state)

        l = 0
        gs = Zygote.gradient(ps) do
            # Foreach timepoint compute enitre batch of activations
            l = loss(xbatch, ybatch)
            return l
        end
        Flux.Optimise.update!(opt, ps, gs)
        acc = accuracy(xbatch, ybatch)
        push!(losses, l)
        push!(accuracies, acc)
    end
    @show mean(losses), mean(accuracies)
end

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

function sample_lstmseq(ch_init = "<", max_len=50)
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

sample_lstmseq()
