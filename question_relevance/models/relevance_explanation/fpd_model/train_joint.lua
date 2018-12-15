-------------------------------------------------------------------------------
-- This script trains false premise detection model as descibed in 
-- The Promise of Premise: Harnessing Question Premises in Visual Question 
-- Answering (EMNLP 2017)
-- author - Aroma Mahendru (maroma@vt.edu)
-- based on HieCoAttenVQA model script train.lua from:
-- https://github.com/jiasenlu/HieCoAttenVQA
-------------------------------------------------------------------------------
require 'nn'
require 'torch'
require 'nngraph'
require 'optim'
require 'misc.netdef'
require 'cutorch'
require 'cunn'
require 'hdf5'
cjson=require('cjson') 
LSTM=require 'misc.LSTM'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train binary classifier for grounding tuples')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_coco','../../img_data/train_fc7.t7','path to the h5file containing the image feature')
cmd:option('-input_img_coco_val','../../img_data/val_fc7.t7','path to the h5file containing the image feature')
cmd:option('-input_img_vg','../../img_data/vg_fc7.t7','path to the h5file containing the image features')

cmd:option('-input_json_coco','../../img_data/coco_train_dict.json','path to the h5file containing the image feature')
cmd:option('-input_json_coco_val','../../img_data/coco_val_dict.json','path to the h5file containing the image feature')
cmd:option('-input_json_vg','../../img_data/vg_dict.json','path to the h5file containing the image feature')

cmd:option('-input_tuples_h5','data/joint.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/joint.json','path to the json file containing additional info and vocab')

-- Model parameter settings
cmd:option('-learning_rate',0.1,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 1000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-batch_size',64,'batch_size for each iterations')
cmd:option('-max_iters', 50000, 'max number of iterations to run for ')
cmd:option('-input_encoding_size', 1000, 'he encoding size of each token in the vocabulary')
cmd:option('-rnn_size', 64,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-rnn_layer',1,'number of the rnn layer')
cmd:option('-common_embedding_size', 128, 'size of the common embedding vector')
cmd:option('-num_output', 2, 'number of output answers')
cmd:option('-img_norm', 0, 'normalize the image feature. 1 = normalize, 0 = not normalize')
cmd:option('-img_scale', 0, 'normalize the image feature. 1 = normalize, 0 = not normalize')
cmd:option('-lambda',0.00018,'lambda parameter for L1 Penalty')

--check point
cmd:option('-save_checkpoint_every', 500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

opt = cmd:parse(arg)
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------


local model_path = opt.checkpoint_path
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local lstm_size_q=opt.rnn_size
local nlstm_layers_q=opt.rnn_layer
local nhimage=4096
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dummy_output_size=1
local decay_factor = 1
local l1_mult = opt.lambda
local wt_init = 'xavier'
--local decay_factor = 0.99997592083 -- 50000
paths.mkdir(model_path)

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local file = io.open(opt.input_json_coco, 'r')
local text = file:read()
file:close()
coco_train_dict = cjson.decode(text)

local file = io.open(opt.input_json_coco_val, 'r')
local text = file:read()
file:close()
coco_val_dict = cjson.decode(text)

local file = io.open(opt.input_json_vg, 'r')
local text = file:read()
file:close()
vg_dict = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_tuples_h5)
local dataset = {}
local h5_file = hdf5.open(opt.input_tuples_h5, 'r')

dataset['tuples_train'] = h5_file:read('/tuples_train'):all()
dataset['labels_train'] = h5_file:read('/labels_train'):all()
dataset['tuples_val'] = h5_file:read('/tuples_val'):all()
dataset['labels_val'] = h5_file:read('/labels_val'):all()
dataset['ttype_train'] = h5_file:read('/ttype_train'):all()
dataset['ttype_val'] = h5_file:read('/ttype_val'):all()
h5_file:close()

dataset['imid_train'] = json_file['imid_train']
dataset['imid_val'] = json_file['imid_val']

print('DataLoader loading t7 file: ', opt.input_img_coco)
dataset['fv_coco'] = torch.load(opt.input_img_coco)


print('DataLoader loading t7 file: ', opt.input_img_coco_val)
dataset['fv_coco_val'] = torch.load(opt.input_img_coco_val)

print('DataLoader loading t7 file: ', opt.input_img_vg)
dataset['fv_vg'] = torch.load(opt.input_img_vg)

dataset['lengths_q_train'] = torch.LongTensor(dataset['tuples_train']:size()[1]):fill(dataset['tuples_train']:size()[2])
dataset['lengths_q_val'] = torch.LongTensor(dataset['tuples_val']:size()[1]):fill(dataset['tuples_val']:size()[2])

--dataset['question'] = right_align(dataset['question'],dataset['lengths_q']) #??

-- Normalize the image feature
if opt.img_norm == 1 then
	local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_coco'],dataset['fv_coco']),2)) 
	dataset['fv_coco']=torch.cdiv(dataset['fv_coco'],torch.repeatTensor(nm,1,nhimage)):float()
	local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_coco_val'],dataset['fv_coco_val']),2)) 
	dataset['fv_coco_val']=torch.cdiv(dataset['fv_coco_val'],torch.repeatTensor(nm,1,nhimage)):float() 
	local nm=torch.sqrt(torch.sum(torch.cmul(dataset['fv_vg'],dataset['fv_vg']),2)) 
	dataset['fv_vg']=torch.cdiv(dataset['fv_vg'],torch.repeatTensor(nm,1,nhimage)):float()
elseif opt.img_scale == 1 then
	local nm=torch.mean(dataset['fv_coco'],1)
	local std=torch.mean(dataset['fv_coco'],1)
	local dsize = dataset['fv_coco']:size()[1]
	dataset['fv_coco']=torch.cdiv(torch.add(dataset['fv_coco'],-1,torch.repeatTensor(nm,dsize,1)) ,torch.repeatTensor(std,dsize,1)):float() 
	local nm=torch.mean(dataset['fv_coco_val'],1)
	local std=torch.mean(dataset['fv_coco_val'],1)
	local dsize = dataset['fv_coco_val']:size()[1]
	dataset['fv_coco_val']=torch.cdiv(torch.add(dataset['fv_coco_val'],-1,torch.repeatTensor(nm,dsize,1)) ,torch.repeatTensor(std,dsize,1)):float() 
	local nm=torch.mean(dataset['fv_vg'],1)
	local std=torch.mean(dataset['fv_vg'],1)
	local dsize = dataset['fv_vg']:size()[1]
	dataset['fv_vg']=torch.cdiv(torch.add(dataset['fv_vg'],-1,torch.repeatTensor(nm,dsize,1)) ,torch.repeatTensor(std,dsize,1)):float() 

else
	dataset['fv_coco'] = dataset['fv_coco']:float()
	dataset['fv_coco_val'] = dataset['fv_coco_val']:float()
	dataset['fv_vg'] = dataset['fv_vg']:float()
end

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count

collectgarbage() 

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

buffer_size_q=dataset['tuples_train']:size()[2]

--Network definitions
--VQA
--embedding: word-embedding
embedding_net_q=nn.Sequential()
				:add(nn.Linear(2*vocabulary_size_q,embedding_size_q)) --todo make 2 a parameter or size of a variable
				:add(nn.Dropout(0.5))
				:add(nn.Tanh())
				:add(nn.L1Penalty(l1_mult))

embedding_net_i=nn.Sequential()
				:add(nn.Linear(nhimage,1000))
				:add(nn.Dropout(0.5))
				:add(nn.Tanh())
				:add(nn.Linear(1000,embedding_size_q))
				:add(nn.L1Penalty(l1_mult))			

--MULTIMODAL
--multimodal way of combining different spaces
multimodal_net=nn.Sequential()
				:add(nn.JoinTable(2))
				:add(nn.Linear(2*embedding_size_q, 64))
				:add(nn.Dropout(0.5))
				:add(nn.Tanh())
				:add(nn.Linear(64,noutput))
				:add(nn.L1Penalty(l1_mult))

--criterion
criterion=nn.CrossEntropyCriterion()

--Optimization parameters
dummy_state_q=torch.Tensor(lstm_size_q*nlstm_layers_q*2):fill(0)
dummy_output_q=torch.Tensor(dummy_output_size):fill(0)

if opt.gpuid >= 0 then
	print('shipped data function to cuda...')
	embedding_net_q = embedding_net_q:cuda()
	embedding_net_i = embedding_net_i:cuda()
	multimodal_net = multimodal_net:cuda()
	criterion = criterion:cuda()
end

--Processings
embedding_w_q,embedding_dw_q=embedding_net_q:getParameters() 
embedding_w_q:uniform(-0.08,0.08)

embedding_w_i,embedding_dw_i=embedding_net_i:getParameters() 
embedding_w_i:uniform(-0.08,0.08)

multimodal_w,multimodal_dw=multimodal_net:getParameters() 
multimodal_w:uniform(-0.08,0.08)


sizes={embedding_w_q:size(1),embedding_w_i:size(1),multimodal_w:size(1)} 


-- optimization parameter
local optimize={} 
optimize.maxIter=opt.max_iters 
optimize.learningRate=opt.learning_rate
optimize.update_grad_per_n_batches=1 

optimize.winit=join_vector({embedding_w_q,embedding_w_i,multimodal_w})

------------------------------------------------------------------------
-- Next batch for train
------------------------------------------------------------------------
function dataset:next_batch()

	local qinds=torch.LongTensor(batch_size):fill(0) 
	local iminds=torch.LongTensor(batch_size):fill(0) 	
	
	local nqs=dataset['tuples_train']:size(1) 
	-- we use the last val_num data for validation (the data already randomlized when created)
	local fv_im = torch.LongTensor(batch_size, dataset['fv_coco']:size()[2]):fill(0)
	for i=1,batch_size do
		qinds[i]=torch.random(nqs) 
		if dataset['labels_train'][qinds[i]] == 1 then
			fv_im[i] = dataset['fv_coco'][coco_train_dict[dataset['imid_train'][qinds[i]]]+1]
		elseif dataset['ttype_train'][qinds[i]] == 0 then
			fv_im[i] = dataset['fv_coco'][coco_train_dict[dataset['imid_train'][qinds[i]]]+1]
		else
			fv_im[i] = dataset['fv_vg'][vg_dict[dataset['imid_train'][qinds[i]]]+1]
		end
	end

	local fv_sorted_q = dataset['tuples_train']:index(1,qinds)
	local labels = dataset['labels_train']:index(1,qinds)
	
	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_sorted_q=fv_sorted_q:cuda() 
		fv_im = fv_im:cuda()
		labels = labels:cuda()
	end

	return fv_sorted_q,fv_im, labels ,batch_size 
end

function dataset:construct_val()
		
	local nqs=dataset['labels_val']:size(1) 
	local fv_im = torch.LongTensor(nqs, dataset['fv_coco_val']:size()[2]):fill(0)
	for i=1,nqs do
		if dataset['labels_val'][i] == 1 then
			fv_im[i] = dataset['fv_coco_val'][coco_val_dict[dataset['imid_val'][i]]+1]
		elseif dataset['ttype_val'][i] == 0 then
			fv_im[i] = dataset['fv_coco_val'][coco_val_dict[dataset['imid_val'][i]]+1]
		else
			fv_im[i] = dataset['fv_vg'][vg_dict[dataset['imid_val'][i]]+1]
		end
	end

	local fv_sorted_q = dataset['tuples_val']
	local labels = dataset['labels_val']
	
	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_sorted_q=fv_sorted_q:cuda() 
		fv_im = fv_im:cuda()
		labels = labels:cuda()
	end

	return fv_sorted_q,fv_im, labels, nqs 
end

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------

-- Objective function
function JdJ(x)
	local params=split_vector(x,sizes) 

	if embedding_w_q~=params[1] then
		embedding_w_q:copy(params[1]) 
	end
	if embedding_w_i~=params[2] then
		embedding_w_i:copy(params[2]) 
	end
	if multimodal_w~=params[3] then
		multimodal_w:copy(params[3]) 
	end

	--clear gradients--

	embedding_dw_q:zero()
	embedding_dw_i:zero() 
	multimodal_dw:zero() 

	--grab a batch--
	local fv_sorted_q,fv_im,labels,batch_size=dataset:next_batch() 
	local question_max_length=fv_sorted_q[2]:size(1) 

	--embedding forward--
	local word_embedding_q=embedding_net_q:forward(fv_sorted_q)
	local im_embedding_q=embedding_net_i:forward(fv_im)
	
	--multimodal/criterion forward--
	local scores=multimodal_net:forward({word_embedding_q,im_embedding_q}) 
	local f=criterion:forward(scores,labels) 
	--multimodal/criterion backward--
	local dscores=criterion:backward(scores,labels) 

	local tmp=multimodal_net:backward({word_embedding_q, im_embedding_q},dscores) 
	dword_embedding_q = tmp[1]
	dword_embedding_i = tmp[2]
	
	--embedding backward--
	embedding_net_q:backward(fv_sorted_q,dword_embedding_q) 
	embedding_net_i:backward(fv_im,dword_embedding_i) 
		
	--summarize f and gradient
	gradients=join_vector({embedding_dw_q,embedding_dw_i,multimodal_dw}) 
	gradients:clamp(-10,10) 
	if running_avg == nil then
		running_avg = f
	end
	running_avg=running_avg*0.95+f*0.05 
	return f,gradients 
end

function compute_val_loss(fv_sorted_q,fv_im,labels,batch_size)
 
	local question_max_length=fv_sorted_q[2]:size(1) 

	--embedding forward--
	local word_embedding_q=embedding_net_q:forward(fv_sorted_q)
	local im_embedding_q=embedding_net_i:forward(fv_im)

	--encoder forward--	
	--multimodal/criterion forward--
	local scores=multimodal_net:forward({word_embedding_q,im_embedding_q}) 
	local f=criterion:forward(scores,labels)
	return f, scores

end
----------------------------------------------------------------------------------------------
-- Training
----------------------------------------------------------------------------------------------
-- With current setting, the network seems never overfitting, so we just use all the data to train

local state={}
v_sorted_q_val,fv_im_val,labels_val, val_size = dataset:construct_val()
for iter = 1, opt.max_iters do
	if iter%opt.save_checkpoint_every == 0 then
		paths.mkdir(model_path..'save')
		torch.save(string.format(model_path..'save/lstm_save_iter%d.t7',iter),
			{encoder_w_q=encoder_w_q,embedding_w_q=embedding_w_q,multimodal_w=multimodal_w}) 
		val_loss, val_scores = compute_val_loss(v_sorted_q_val,fv_im_val,labels_val, val_size)
		print('val loss: ' .. val_loss, 'on iter: ' .. iter .. '/' .. opt.max_iters)
		torch.save(string.format(model_path..'save/preds/preds%d.t7',iter),
			{scores = val_scores, labels = labels_val}) 
	end
	if iter%100 == 0 then
		print('training loss: ' .. running_avg, 'on iter: ' .. iter .. '/' .. opt.max_iters)
	end
	optim.sgd(JdJ, optimize.winit, optimize, state)
	
	optimize.learningRate=optimize.learningRate*decay_factor 
	if iter%50 == 0 then -- change this to smaller value if out of the memory
		collectgarbage()
	end
end

