module learnsequence
end # module

using Flux
using Zygote
using Random
using Statistics
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
opt = Flux.Optimise.Descent(lr)

data = sample_kmers(6, batch_size)
loader = Flux.Data.DataLoader(data, batchsize=batch_size,partial=false)

nhidden = 32
a_prev_tmp = randn(nhidden, batch_size)  # hidden x number of examples
c_prev_tmp = randn(nhidden, batch_size)  # for LSTM variant we also need memory cells

model = Flux.Recur(MyRNNCell(length(vocab), nhidden), a_prev_tmp)
#model = Flux.Recur(MyLSTMCell(length(vocab), nhidden), (a_prev_tmp, c_prev_tmp))
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
        #if acc > 0.99
        #    epoch = 100
        #end
    end
    @show mean(losses), mean(accuracies)
end

function test_model()
    test =  # We expect mostly B
    test_batch = Flux.unstack(Flux.batch(map(seq -> float.(Flux.onehotbatch(seq,vocab)),test)),2)
    pred = Flux.onecold(Flux.batch(model.(test_batch)),vocab)
    return sum(pred .== ['C'])/batch_size
end
test_model()


function predict(c)
    x = [[c] for i in 1:batch_size]
    x_batch = Flux.unstack(Flux.batch(map(seq -> float.(Flux.onehotbatch(seq,vocab)),x)),2)
    # Sample random element
    return Random.rand(Flux.onecold(Flux.batch(model.(x_batch)),vocab))
end

function predict_seq(c,n)
    seq = []
    for i in 1:n
        c = predict(c)
        push!(seq,c)
    end
    return join(seq)
end

predict_seq('<',50) # "ABCDEFGHIJKLMNOPQRSTUVWXYZ>OPQRSTUVWXYZ>OPQRSTUVWX"
predict_seq('<',50) # "CDEFGHIJKLMNOPQRSTUVWXYZ>OPQRSTUVWXYZ>OPQRSTUVWXYZ"
predict_seq('<',50) # ">OPQRSTUVWXYZ>UVWXYZ>OPQRSTUVWXYZ>UVWXYZ>OPQRSTUVW"
predict_seq('<',50) # "ABCDEFGHIJKLMNOPQRSTUVWXYZ>OPQRSTUVWXYZ>>VWXYZ>OPQ"
predict_seq('<',50) # "ABCDEFGHIJKLMNOPQRSTUVWXYZ>OPQRSTUVWXYZ>OPQRSTUVWX"
predict_seq('C',50) # "DEFGHIJKLMNOPQRSTUVWXYZ>OPQRSTUVWXYZ>OPQRSTUVWXYZ>"
predict_seq('D',50) # "EFGHIJKLMNOPQRSTUVWXYZ>OPQRSTUVWXYZ>UVWXYZ>OPQRSTU"
predict_seq('P',50) # "QRSTUVWXYZ>OPQRSTUVWXYZ>>VWXYZ>OPQRSTUVWXYZ>OPQRST"
predict_seq('U',50) # "VWXYZ>STUVWXYZ>OPQRSTUVWXYZ>OPQRSTUVWXYZ>>VWXYZ>HI"

