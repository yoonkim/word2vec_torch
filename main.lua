--[[
Config file for skipgram with negative sampling 
--]]

require("io")
require("os")
require("paths")
require("torch")
dofile("word2vec.lua")

-- Default configuration
config = {}
config.corpus = "corpus.txt" -- input data
config.window = 5 -- (maximum) window size
config.dim = 100 -- dimensionality of word embeddings
config.alpha = 0.75 -- smooth out unigram frequencies
config.table_size = 1e8 -- table size from which to sample neg samples
config.neg_samples = 5 -- number of negative samples for each positive sample
config.minfreq = 10 --threshold for vocab frequency
config.lr = 0.025 -- initial learning rate
config.min_lr = 0.001 -- min learning rate
config.epochs = 3 -- number of epochs to train
config.gpu = 0 -- 1 = use gpu, 0 = use cpu
config.stream = 1 -- 1 = stream from hard drive 0 = copy to memory first

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus", config.corpus)
cmd:option("-window", config.window)
cmd:option("-minfreq", config.minfreq)
cmd:option("-dim", config.dim)
cmd:option("-lr", config.lr)
cmd:option("-min_lr", config.min_lr)
cmd:option("-neg_samples", config.neg_samples)
cmd:option("-table_size", config.table_size)
cmd:option("-epochs", config.epochs)
cmd:option("-gpu", config.gpu)
cmd:option("-stream", config.stream)
params = cmd:parse(arg)

for param, value in pairs(params) do
    config[param] = value
end

for i,j in pairs(config) do
    print(i..": "..j)
end
-- Train model
m = Word2Vec(config)
m:build_vocab(config.corpus)
m:build_table()

for k = 1, config.epochs do
    m.lr = config.lr -- reset learning rate at each epoch
    m:train_model(config.corpus)
end
m:print_sim_words({"the","he","can"},5)
