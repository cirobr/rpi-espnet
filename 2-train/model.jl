"""
LibFluxML.Learn!()
Author: cirobr@GitHub
Date: 20-Oct-2024
"""

@info "Project start"
cd(@__DIR__)

### arguments
cudadevice = parse(Int64, ARGS[1])
nepochs    = parse(Int64, ARGS[2])
debugflag  = parse(Bool,  ARGS[3])

# cudadevice = 0
# nepochs    = 400
# debugflag  = true

script_name = basename(@__FILE__)
@info "script_name: $script_name"
@info "cudadevice: $cudadevice"
@info "nepochs: $nepochs"
@info "debugflag: $debugflag"

minibatchsize = 4
epochs        = nepochs


### libs
using Pkg
envpath = expanduser("~/envs/dev/")
Pkg.activate(envpath)

using CUDA
CUDA.device!(cudadevice)
CUDA.versioninfo()

using Flux
import Flux: relu, leakyrelu
using Images
using DataFrames
using CSV
using JLD2
using FLoops
using Random
using Statistics: mean, minimum, maximum, norm
using StatsBase: sample
using MLUtils: splitobs, kfolds, randobs
using Mmap
using HyperTuning
using Dates


# private libs
using TinyMachines; const tm=TinyMachines
using LibMetalhead
using PreprocessingImages; const p=PreprocessingImages
using CocoTools; const c=CocoTools
using LibFluxML
import LibFluxML: IoU_loss, ce1_loss, ce3_loss, cosine_loss, softloss,
                  AccScore, F1Score, IoUScore
using LibCUDA

LibCUDA.cleangpu()
@info "environment OK"


### generate a random string based on the current date and time
random_string() = string(Dates.format(now(), "yyyymmdd_HHMMSSsss"))


### constants
const KB = 1024
const MB = KB * KB
const GB = KB * MB


### folders
outputfolder = script_name[1:end-3] * "/"

# pwd(), homedir()
workpath = pwd() * "/"
workpath = replace(workpath, homedir() => "~")
datasetpath = "~/projects/kd-coco-dataset/"
# mkpath(expanduser(datasetpath))   # it should already exist

modelspath  = workpath * "models/" * outputfolder
mkpath(expanduser(modelspath))

tblogspath  = workpath * "tblogs/" * outputfolder
rm(tblogspath; force=true, recursive=true)
mkpath(expanduser(tblogspath))

tmp_path = "/scratch/cirobr/tmp/"
# tmp_path = "/tmp/"
# mkpath(tmp_path)     # it should already exist
@info "folders OK"


### datasets
@info "creating datasets..."
classnames   = ["cow"]   #["cat", "cow", "dog", "horse", "sheep"]
classnumbers = [c.coco_classnames[classname] for classname in classnames]
C = length(classnumbers) + 1

fpfn = expanduser(datasetpath) * "dftrain-cow-resized.csv"
dftrain = CSV.read(fpfn, DataFrame)
dftrain = dftrain[dftrain.cow,:]
# too large dataset: get % of random observations
# dftrain = randobs(dftrain, trunc(Int, size(dftrain, 1) * 0.8))

fpfn = expanduser(datasetpath) * "dfvalid-cow-resized.csv"
dfvalid = CSV.read(fpfn, DataFrame)
dfvalid = dfvalid[dfvalid.cow,:]


########### debug ############
if debugflag
      dftrain = first(dftrain, 5)
      dfvalid = first(dfvalid, 2)
      minibatchsize = 1
      epochs  = 2
end
##############################


# check memory requirements
Xtr = Images.load(expanduser(dftrain.X[1]))
ytr = Images.load(expanduser(dftrain.y[1]))
Ntrain = size(dftrain, 1)
Nvalid = size(dfvalid, 1)

dpsize   = sizeof(Xtr) + sizeof(ytr)
dpsizeGB = dpsize / GB
dbsizeGB = dpsizeGB * (Ntrain + Nvalid)

@info "dataset points = $(Ntrain + Nvalid)"
@info "dataset size = $(dbsizeGB) GB"
# @assert dbsizeGB < 2.0 || error("dataset is too large")
# @info "dataset size OK"

# create tensors
@info "creating tensors..."
dims = size(Xtr)
Ntrain = size(dftrain, 1)
Nvalid = size(dfvalid, 1)

# Xtrain = Array{Float32, 4}(undef, (dims...,3,Ntrain))
Xtrain_fpfn = tmp_path * "xtrain_" * random_string() * ".bin"
io_xtrain   = open(Xtrain_fpfn, "w+", lock=true)
Xtrain      = mmap(io_xtrain, Array{Float32, 4}, (dims...,3,Ntrain))

# ytrain = Array{Bool, 4}(undef, (dims...,C,Ntrain))
ytrain_fpfn = tmp_path * "ytrain_" * random_string() * ".bin"
io_ytrain   = open(ytrain_fpfn, "w+", lock=true)
ytrain      = mmap(io_ytrain, Array{Bool, 4}, (dims...,C,Ntrain))

# Xvalid = Array{Float32, 4}(undef, (dims...,3,Nvalid))
Xvalid_fpfn = tmp_path * "xvalid_" * random_string() * ".bin"
io_xvalid   = open(Xvalid_fpfn, "w+", lock=true)
Xvalid      = mmap(io_xvalid, Array{Float32, 4}, (dims...,3,Nvalid))

# yvalid = Array{Bool, 4}(undef, (dims...,C,Nvalid))
yvalid_fpfn = tmp_path * "yvalid_" * random_string() * ".bin"
io_yvalid   = open(yvalid_fpfn, "w+", lock=true)
yvalid      = mmap(io_yvalid, Array{Bool, 4}, (dims...,C,Nvalid))

dfs = [dftrain, dfvalid]
Xouts = [Xtrain, Xvalid]
youts = [ytrain, yvalid]

for (df, Xout, yout) in zip(dfs, Xouts, youts)   # no @floop here
      N = size(df, 1)
      @floop for i in 1:N
            local fpfn = expanduser(df.X[i])
            img = Images.load(fpfn)
            img = p.color2Float32(img)
            Xout[:,:,:,i] = img

            local fpfn = expanduser(df.y[i])
            mask = Images.load(fpfn)
            mask = p.gray2Int32(mask)
            mask = Flux.onehotbatch(mask, [0,classnumbers...],0)
            mask = permutedims(mask, (2,3,1)) .|> Bool
            yout[:,:,:,i] = mask
      end
      Mmap.sync!(Xout)
      Mmap.sync!(yout)
end
@info "tensors OK"

# dataloaders
Random.seed!(1234)   # to enforce reproducibility
trainset = Flux.DataLoader((Xtrain, ytrain),
                            batchsize=minibatchsize,
                            shuffle=true) |> gpu
validset = Flux.DataLoader((Xvalid, yvalid),
                            batchsize=1,
                            shuffle=false) |> gpu
@info "dataloader OK"


LibCUDA.cleangpu()


### model
Random.seed!(1234)   # to enforce reproducibility
α2 = 5
α3 = 3
modelcpu = ESPnet(3,C; activation=leakyrelu, alpha2=α2, alpha3=α3, verbose=false)
# fpfn = expanduser("")
# LibFluxML.loadModelState!(fpfn, modelcpu)
model    = modelcpu |> gpu;
@info "model OK"


# check for matching between model and data
Xtr = Xtrain[:,:,:,1:1] |> gpu
ytr = ytrain[:,:,:,1:1] |> gpu
@assert size(model(Xtr)) == size(ytr) || error("model/data features do not match")
@info "model/data matching OK"


# loss functions
lossFunction(yhat, y) = IoU_loss(yhat, y)
lossfns = [lossFunction]
@info "loss functions OK"


# optimizer
optimizerFunction = Flux.RMSProp
η = 5e-4
λ = 0.0
modelOptimizer = λ > 0 ? Flux.Optimiser(WeightDecay(λ), optimizerFunction(η)) : optimizerFunction(η)


optimizerState = Flux.setup(modelOptimizer, model)
# Flux.freeze!(optimizerState.enc)
@info "optimizer OK"


### training
@info "start training ..."

number_since_best = 20
patience = 5
metrics = [
      AccScore,
      F1Score,
      IoUScore,
]

Random.seed!(1234)   # to enforce reproducibility
Learn!(epochs, model, (trainset, validset), optimizerState, lossfns;
      metrics=metrics,
      earlystops=(number_since_best, patience),
      modelspath=modelspath * "train/",
      tblogspath=tblogspath * "train/"
)

fpfn = expanduser(modelspath) * "train/model.jld2"
mv(fpfn, expanduser(modelspath) * "train/bestmodel.jld2", force=true)
@info "training OK"


### tuning
@info "start tuning ..."
fpfn = expanduser(modelspath) * "train/bestmodel.jld2"
LibFluxML.loadModelState!(fpfn, modelcpu)
model = modelcpu |> gpu

Flux.thaw!(optimizerState)
Flux.adjust!(optimizerState, η/10)
@info "optimizer adjusted"

Random.seed!(1234)   # to enforce reproducibility
Learn!(epochs, model, (trainset, validset), optimizerState, lossfns;
      metrics=metrics,
      earlystops=(number_since_best, patience),
      modelspath=modelspath * "tune/",
      tblogspath=tblogspath * "tune/"
)

fpfn = expanduser(modelspath) * "tune/model.jld2"
mv(fpfn, expanduser(modelspath) * "tune/bestmodel.jld2", force=true)
@info "tuning OK"


### clean memory
close(io_xtrain)
close(io_ytrain)
close(io_xvalid)
close(io_yvalid)
rm(Xtrain_fpfn; force=true)
rm(ytrain_fpfn; force=true)
rm(Xvalid_fpfn; force=true)
rm(yvalid_fpfn; force=true)

LibCUDA.cleangpu()
@info "project finished"
