#importing libraries
from loguru import logger
import sys
import pandas as pd
import csv
import itertools
import hydra
import torch
import os
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
from transformers import XLMConfig,XLMTokenizer,XLMModel
from allennlp.modules.elmo_lstm import ElmoLstm
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from global_vars import *

# The model architecture definition 
# Its static now 
# Todo can replace the bilm with BiLstm
class Model(nn.Module):
    """ 
        The LM Model to be used in case of the Streamed Input
        Inherits from the base class nn.Module
        init and forward are defined
    """
    def __init__(self,options):
        """
            Options is a dictionary initialized using the configuration parameters like 
            hid_size: The hidden size returned by the input module (Here we are using XLM-17-1280 so  it has a value of 1280)
            dropout: The dropout value to be used in the BiLM and optionally in the linear layer
            num_lstm_layers: The number of stacked layers of bilm
            vocab_size: To initialize the final fully connected layer size
        """
        super(Model, self).__init__()
        logger.info(options)
        self.bilm=BiLMEncoder(options['hid_size'],options['hid_size'],options['hid_size'],options['num_lstm_layers'],recurrent_dropout_probability=options['dropout'])
        self.lin=torch.nn.Linear(options['hid_size'],options['vocab_size'])
        self.dropout=nn.Dropout(p=options['dropout'])
    def forward(self,enc_embedding):
        """
            This takes as input the output from the xlm module
            of shape (batch_size,max_seq_len,hid_size)
            Returns a tensor of shape (2*hid_size*batch_size)*vocab_size
            (2 is due to the bidirectionalism of bilm)
        """
        # This is ideally a mask of ones and used as it is with streamed input
        # Mask of ones of shape batch_size*max_seq_len
        mask=torch.ones((enc_embedding[0].shape[0],max_seq_len)).to(device)
        
        enc=self.bilm(enc_embedding[0],mask) 
        # returns tensor of size 1*batch_size*max_seq_len*(2*hid_size)
        # Here 2 is due to bidirectionalism
        fwd,bwd=enc[:,:,:,:hid_size],enc[:,:,:,hid_size:]
        # fwd and bwd of size 1*batch_size*max_seq_len*hid_size
        logits_fwd=self.lin(fwd).view(enc_embedding[0].shape[0] * max_seq_len, -1)
        logits_bwd=self.lin(bwd).view(enc_embedding[0].shape[0] * max_seq_len, -1)
        # logits fwd and logits bwd each of sizes (batch_size*max_seq_len)*vocab_size
        logits=torch.cat((logits_fwd,logits_bwd),dim=0)
        # logits of sizes (2*batch_size*max_seq_len)*vocab_size
        return logits


class Model_pad(nn.Module):
    """ 
        The LM Model to be used in case of the padded Input
        Inherits from the base class nn.Module
        init and forward are defined
    """
    def __init__(self,options):
        """
            Options is a dictionary initialized using the configuration parameters like 
            hid_size: The hidden size returned by the input module (Here we are using XLM-17-1280 so  it has a value of 1280)
            dropout: The dropout value to be used in the BiLM and optionally in the linear layer
            num_lstm_layers: The number of stacked layers of bilm
            vocab_size: To initialize the final fully connected layer size
        """
        super(Model_pad, self).__init__()
        logger.info(options)
        self.bilm=BiLMEncoder(options['hid_size'],options['hid_size'],options['hid_size'],options['num_lstm_layers'],recurrent_dropout_probability=options['dropout'])
        self.lin=torch.nn.Linear(options['hid_size'],options['vocab_size'])
        self.dropout=nn.Dropout(p=options['dropout'])
    def forward(self,enc_embedding,lt):
        """
            This takes as input the output from the xlm module
            of shape (batch_size,max_seq_len,hid_size)
            Returns a tensor of shape (2*hid_size*batch_size)*vocab_size
            (2 is due to the bidirectionalism of bilm)
        """
        # This is ideally a mask of ones. Will have to set this to zeros using sequence length if padding load is used 
        mask=torch.zeros((enc_embedding[0].shape[0],max_seq_len))
        for j,val in enumerate(lt):
            mask[j,:val]=torch.ones((val))
        mask=mask.to(device)
        enc=self.bilm(enc_embedding[0],mask)
        # returns tensor of size 1*batch_size*max_seq_len*(2*hid_size)
        # Here 2 is due to bidirectionalism        
        fwd,bwd=enc[:,:,:,:hid_size],enc[:,:,:,hid_size:]
        # fwd and bwd of size 1*batch_size*max_seq_len*hid_size
        logits_fwd=self.lin(fwd).view(enc_embedding[0].shape[0] * max_seq_len, -1)
        logits_bwd=self.lin(bwd).view(enc_embedding[0].shape[0] * max_seq_len, -1)
        # logits fwd and logits bwd each of sizes (batch_size*max_seq_len)*vocab_size
        logits=torch.cat((logits_fwd,logits_bwd),dim=0)
        # logits of sizes (2*batch_size*max_seq_len)*vocab_size
        return logits


class BiLMEncoder(ElmoLstm):
    """Wrapper around BiLM to give it an interface 
       Basically an lstm cell with a projection (reduced size than standard lstm if multiple layers are used)
    """

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.hidden_size * 2

@hydra.main(config_path="config.yaml")
def configsetters(cfg):
    """
        The function which takes cfg which is the configuration parameters object retrieved using hydra
        It is being used to set the global variables using the configuration file and loads the datafiles if preprocessed already 
        making use of the preprocessed flag and if not then calls the preprocess function which genrates the different dataloader objects.
        Input:
        configuration object from hydra
    """
    global max_seq_len,batch_size,files_dir,train_file,test_file,val_file,hid_size,vocab_size,epochs,learning_rate,pretrained_model_name,xlm,run_name,load_model,load_model_file,val_factor
    global min_learning_rate,weight_decay_factor,steps_for_validation,load_optim_file,load_tsv
    global data_as_stream,dropout
    global do_training,do_eval,cuda,vocab_file,num_lstm_layers
    #setting the global variables using the hydra config file
    max_seq_len=cfg.main.max_seq_len
    batch_size=cfg.main.batch_size
    files_dir=cfg.main.files_dir
    train_file=cfg.main.train_file
    test_file=cfg.main.test_file
    val_file=cfg.main.val_file
    hid_size=cfg.main.hid_size
    vocab_size=cfg.main.vocab_size
    epochs=cfg.main.epochs
    learning_rate=cfg.main.learning_rate
    min_learning_rate=cfg.main.min_learning_rate
    weight_decay_factor=cfg.main.weight_decay_factor
    pretrained_model_name=cfg.main.pretrained_model
    run_name=cfg.main.run_name
    load_model=cfg.main.load_model
    load_model_file=cfg.main.load_model_file
    load_optim_file=cfg.main.load_optim_file
    val_factor=cfg.main.val_factor
    steps_for_validation=cfg.main.steps_for_validation
    load_tsv=cfg.main.load_tsv
    data_as_stream=cfg.main.data_as_stream
    dropout=cfg.main.dropout
    do_training=cfg.main.do_training
    do_eval=cfg.main.do_eval
    cuda=cfg.main.cuda
    vocab_file=cfg.main.vocab_file
    num_lstm_layers=cfg.main.num_lstm_layers
    global train_sents,test_sents,val_sents
    
    # File names for the preprocessed data 
    train_loaded_file=files_dir+run_name+"/"+train_file+"_preprocessed.pt"
    test_loaded_file=files_dir+run_name+"/"+test_file+"_preprocessed.pt"
    val_loaded_file=files_dir+run_name+"/"+val_file+"_preprocessed.pt"
    
    # Creating the log file in the run directory 
    if not os.path.exists(files_dir+run_name):
        os.makedirs(files_dir+run_name)
    logger.add(files_dir+run_name+"/logs.log")
    logger.info(cfg.pretty())

    # Flag to use the preprocessed data 
    if cfg.main.preprocessed==1:
        logger.info("Loading the existing data files ")
        if do_training:
            train_sents=torch.load(train_loaded_file)
            val_sents=torch.load(val_loaded_file)
        if do_eval:
            test_sents=torch.load(test_loaded_file)
        
        # vocab_file parameter used with load_model 
        # vocab_file=files_dir+run_name+"/"+"vocab_"+train_file

        #if preprocessing is done or model is pretrained it should have a vocabulary
        # overriding the sequence length and vocabsize parameter
        with open(vocab_file,'r',encoding="utf-8") as fp:
            data=fp.readlines()
            vocab_size=int(data[0].split("\n")[0])
            max_seq_len=int(data[1].split("\n")[0])
    else:
        preprocess()


def preprocess():
    """
        Function to process the datasets and generate the TensorDataset objects for them based on the type of runs
        If do_train==1 generates train_sents,val_sents TensorDataset objects
        If do_eval==1 generates test_sents TensorDataset object
        It calls the stream_load or padding_load function based on the data_as_stream config parameter
    """
    txtfiles=[]
    global max_seq_len,batch_size,files_dir,train_file,test_file,val_file,hid_size,vocab_size,epochs,learning_rate,pretrained_model_name,xlm,run_name,load_model,load_model_file
    global vocab_file
    train_filepath=files_dir+train_file
    test_filepath=files_dir+test_file
    val_filepath=files_dir+val_file
    
    # The tokenizer to be used 
    # Currently supporting only XLM
    # todo
    tokenizer=XLMTokenizer.from_pretrained(pretrained_model_name)
    
    # List containing the filenames
    if do_training:
        txtfiles.append(train_filepath)
        txtfiles.append(val_filepath)
    if do_eval:
        txtfiles.append(test_filepath)
    
    
    # The global variables storing the TensorDataset for validation,train and test datasets
    global val_sents,train_sents,test_sents
    
    for ind,file in enumerate(txtfiles):
        filepath=file
        # To load a tsv format file need to set the flag in config file
        # Todo experimental currently only supports two fields 
        # need to add support for jsonl files and include labels too
        if load_tsv==1:
            delimiter="\t"
            rows = pd.read_csv(
            filepath,
            sep=delimiter,
            error_bad_lines=True,
            quoting=csv.QUOTE_NONE,
            keep_default_na=False,
            encoding="utf-8",
             )
            label1=rows.columns[0]
            label2=rows.columns[1]
            sents=list(rows[label1])
            for j in list(rows[label2]):
                sents.append(j)
        else:        
            with open(filepath,'r',encoding='utf-8') as fp:
                data=fp.read()
            sents=data.split("\n")
        tokenized_sents=[i for i in range(len(sents))]
        # Do this only for the train file
        # building a vocab from the train_file  only
        # Todo need an override flag for this
        if ind==0 and do_training==1:
            # Storing the indices of the tokenized sentences to restrict the vocab 
            token_inds=set()
            for i,s in enumerate(sents):

                # Added the check for the datalength to be passed into xlm tokenizer should be less than 512
                # setting it to 510 to take account of the start and end token also
                token_sent=tokenizer.tokenize(s)
                if len(token_sent)>=512:
                    token_sent=token_sent[:510]

                temp=tokenizer.convert_tokens_to_ids(token_sent)
                [token_inds.add(ind) for ind in temp]

            # This is also XLM specific to include the <s> </s> <pad> <unk> tokens 
            spl_tokens=[tokenizer.special_tokens_map[k] for (k) in tokenizer.special_tokens_map.keys() if k!='additional_special_tokens']
            spl_tokens_inds=tokenizer.convert_tokens_to_ids(spl_tokens)
            spl_tokens_inds=set(spl_tokens_inds)
            [token_inds.add(i) for i in spl_tokens_inds]
            
            # the dictionary which maps the xlm indices to restricted vocab indices
            restricted_vocab={}
            # Key is tokenizers index and value is restricted vocabs index 
            token_inds=list(token_inds)

            # sorting is done to maintain the order of special tokens in the starting only
            token_inds=sorted(token_inds)

            # populating the restricted vocab
            for i,key in enumerate(token_inds):
                restricted_vocab[key]=i

            # Hyperprameter is setup and this overrides the parameter in the config file    
            vocab_size=len(restricted_vocab)

            # Saving the vocabulary
            # In the first line is the number of tokens
            # in the second line is the seequence length used
            # Then the tokens from tokenizer as a separate one per line

            vocab_file=files_dir+run_name+"/"+"vocab_"+train_file
            with open(vocab_file,'w',encoding="utf-8") as fp:
                fp.write(str(len(restricted_vocab))+"\n")
                fp.write(str(max_seq_len))
                fp.write("\n")
                kys=list(restricted_vocab.keys())
                vals=tokenizer.convert_ids_to_tokens(kys)
                for k in vals:
                    fp.write(k+"\n")
        
        # Only testing has to be done then also loading the vocabulary
        #todo
        #Vocab file name is coupled with the training file currently
        # vocab_file=files_dir+run_name+"/"+"vocab_"+train_file
        if do_training==0 and do_eval==1:
            with open(vocab_file,'r',encoding='utf-8') as fp:
                data=fp.readlines()
            # First two lines are vocab size and seq_len 
            vocab_size=int(data[0].split("\n")[0])
            data=data[2:]
            restricted_vocab={}
            for i,w in enumerate(data):
                restricted_vocab[tokenizer.convert_tokens_to_ids(w.split("\n")[0])]=i
        # If the data is streamed do not use a padding token 
        
        if  data_as_stream==1:
            arr_sents=stream_load(sents,tokenizer,restricted_vocab,max_seq_len)
        else:
            arr_sents=padding_load(sents,tokenizer,restricted_vocab,max_seq_len)
        # arr_sents is the tensordataset object 

        # Saving the preprocessed data files to be used for running the experiments again
        train_loaded_file=files_dir+run_name+"/"+train_file+"_preprocessed.pt"
        test_loaded_file=files_dir+run_name+"/"+test_file+"_preprocessed.pt"
        val_loaded_file=files_dir+run_name+"/"+val_file+"_preprocessed.pt"

        #todo check if both do_training and do_eval are nont set raise error
        if ind==0:
            # if the training has to be done
            if do_training:
                train_sents=arr_sents
                torch.save(train_sents,train_loaded_file)
                trainloader=DataLoader(arr_sents,batch_size=batch_size,shuffle=True)
            # if only  the testing has to be done
            else:
                test_sents=arr_sents
                torch.save(test_sents,test_loaded_file)
                testloader=DataLoader(arr_sents,batch_size=batch_size,shuffle=True)    
        # always used with the validation dataaset                  
        elif ind==1:
            val_sents=arr_sents
            torch.save(val_sents,val_loaded_file)
            valloader=DataLoader(arr_sents,batch_size=batch_size,shuffle=True)
        # used if all training  testing validation has to be done in current run
        else:
            test_sents=arr_sents
            torch.save(test_sents,test_loaded_file)
            testloader=DataLoader(arr_sents,batch_size=batch_size,shuffle=True)



def padding_load(sents,tokenizer,restricted_vocab,max_seq_len):
    """
        Function to be used for padding the sentences and not loading as a stream
        Inputs:
        sents: The list of raw language input sentences
        tokenizer: The tokenizer to be used for tokenizing the words of sentence (Here makes use of the XLM Tokenizer)
        restricted_vocab: The dictionary  which maps the xlm word token indices to restricted vocab indices
        max_seq_len: The configuration parameter max_seq_len
        returns a tensordataset
    """
    # list of input sentences to be fed into the model :uses xlms vocabulary 
    tokenized_sents=[i for i in range(len(sents))]
    for i,sent in enumerate(sents):
        tokenized_sents[i]=tokenizer.encode(sent)
    
    # list of fwd tokens references restricted vocab
    fwd_sents=[]
    for i,sent in enumerate(tokenized_sents):
        temp=[]
        for j in range(1,len(sent)):
            temp.append(restricted_vocab[sent[j]] if sent[j] in restricted_vocab.keys() else tokenizer.unk_token_id)
        temp.append(tokenizer.pad_token_id)
        fwd_sents.append(temp)
    # list of bwd tokens references restricted vocab
    bwd_sents=[]
    for i,sent in enumerate(tokenized_sents):
        temp=[]
        temp.append(tokenizer.pad_token_id)
        for j in range(0,len(sent)-1):
            temp.append(restricted_vocab[sent[j]] if sent[j] in restricted_vocab.keys() else tokenizer.unk_token_id)
        bwd_sents.append(temp)
        #    To implement bidirectionalism
        #Sentence: He is a boy
        #inp: <s> He is a boy </s>
        #fwd: He is a boy </s> <pad>
        #bwd: <pad> <s> He is a boy
    input_sent_lens=[]
    for sent in tokenized_sents:
        input_sent_lens.append((len(sent)) if (len(sent))<max_seq_len else max_seq_len)
    # IDeally keep the padding token id same as in the original namespace of restircted vocabulary
    for i in range(len(tokenized_sents)):
        if len(tokenized_sents[i])<max_seq_len:
            tokenized_sents[i]=tokenized_sents[i]+[tokenizer.pad_token_id]*(max_seq_len-len(tokenized_sents[i]))
        else:
            tokenized_sents[i]=tokenized_sents[i][:max_seq_len-1]+[1]
    
    for i in range(len(fwd_sents)):
        if len(fwd_sents[i])<max_seq_len:
            fwd_sents[i]=fwd_sents[i]+[tokenizer.pad_token_id]*(max_seq_len-len(fwd_sents[i]))
        else:
            fwd_sents[i]=fwd_sents[i][:max_seq_len]
    
    for i in range(len(bwd_sents)):
        if len(bwd_sents[i])<max_seq_len:
            bwd_sents[i]=bwd_sents[i]+[tokenizer.pad_token_id]*(max_seq_len-len(bwd_sents[i]))
        else:
            bwd_sents[i]=bwd_sents[i][:max_seq_len]

    
    input_sent_lens=np.array(input_sent_lens)
    arr_tokenized_sents=np.array(tokenized_sents,dtype=int)
    arr_bwd_sents=np.array(bwd_sents,dtype=int)
    arr_fwd_sents=np.array(fwd_sents,dtype=int)

    arr_sents=TensorDataset(torch.from_numpy(arr_tokenized_sents),torch.from_numpy(arr_fwd_sents),torch.from_numpy(arr_bwd_sents),torch.from_numpy(input_sent_lens))
    return arr_sents


def stream_load(sents,tokenizer,restricted_vocab,max_seq_len):
    """
        The function to be used for loading the entire data as a stream of text and not using  the padding token.
        Inputs:
        sents: The list of raw language input sentences
        tokenizer: The tokenizer to be used for tokenizing the words of sentence (Here makes use of the XLM Tokenizer)
        restricted_vocab: The dictionary  which maps the xlm word token indices to restricted vocab indices
        max_seq_len: The configuration parameter max_seq_len
        returns a tensordataset
    """
    tokenized_sents=[i for i in range(len(sents))]
    for i,sent in enumerate(sents):
        # For bidrectionalism making use of only </s> token 
        sent=sent+"</s>"
        tokenized_sents[i]=tokenizer.encode(sent,add_special_tokens=False) 
    
    # The list of all the sentence tokens conatenated as a one list 
    merged_sents=list(itertools.chain.from_iterable(tokenized_sents))
    
    # As the vocabulary is restricted the inp_sents will have the token id of the tokenizer and fwd_sents and bwd_sents (targets) will have the token id of the restricted vocab
    inp_sents=[]
    fwd_sents=[]
    bwd_sents=[]
    n=len(merged_sents)

    # The main code to implement the bidrectionalism 
    # The data stream looks like this
    # Data Stream: This is a boy</s> He is a good king.</s> There is a cat ...
    # seq_len:11
    # inp: This is a boy </s> He is a good king</s>
    # fwd: is a boy </s> He is a good king </s> There
    # bwd: There This is a boy </s> He is a good king
    for i in range(0,n,max_seq_len):
        temp=merged_sents[i:i+max_seq_len]
        inp_sents.append(temp)
        temp=[]
        # Start the index from i+1 token 
        for j in range(i+1,i+max_seq_len+1 if n>i+max_seq_len+1 else n):
            temp.append(restricted_vocab[merged_sents[j]] if merged_sents[j] in restricted_vocab.keys() else tokenizer.unk_token_id)
        fwd_sents.append(temp)
        temp=[]
        # Append the last token first
        temp.append(restricted_vocab[merged_sents[i+max_seq_len-1 if n>i+max_seq_len else n-1]] if merged_sents[i+max_seq_len-1 if n>i+max_seq_len else n-1] in restricted_vocab.keys() else tokenizer.unk_token_id)
        for j in range(i,i+max_seq_len-1 if n>i+max_seq_len else n-1):
            temp.append(restricted_vocab[merged_sents[j]] if merged_sents[j] in restricted_vocab.keys() else tokenizer.unk_token_id)
        bwd_sents.append(temp)


    # To finally store the list as an array the last sentence split may not have max_seq_len tokens thus using the padding token 
    if len(inp_sents[-1])<max_seq_len:
        inp_sents[-1]=inp_sents[-1]+[tokenizer.pad_token_id]*(max_seq_len-len(inp_sents[-1]))
    if len(fwd_sents[-1])<max_seq_len:
        fwd_sents[-1]=fwd_sents[-1]+[tokenizer.pad_token_id]*(max_seq_len-len(fwd_sents[-1]))
    if len(bwd_sents[-1])<max_seq_len:
        bwd_sents[-1]=bwd_sents[-1]+[tokenizer.pad_token_id]*(max_seq_len-len(bwd_sents[-1]))
    
    # Converting the lists to an array to be used with TensorDataset
    arr_inp_sents=np.array(inp_sents,dtype=int)
    arr_fwd_sents=np.array(fwd_sents,dtype=int)
    arr_bwd_sents=np.array(bwd_sents,dtype=int)

    #TensorDataset returned to the preprocess function
    arr_sents=TensorDataset(torch.from_numpy(arr_inp_sents),torch.from_numpy(arr_fwd_sents),torch.from_numpy(arr_bwd_sents))
    return arr_sents

          

 
def model_setup():
    """
        Main function to load and define the model
        Does the job of creating the model object and loads exisiting model if load_model flag is set
        If do_pretrain is set then calls the appropriate train function
        If do_eval is set then calls the appropriate test function
    """
    options={}
    options['max_seq_len']=max_seq_len
    options['batch_size']=batch_size
    options['vocab_size']=vocab_size
    options['epochs']=epochs
    options['hid_size']=hid_size
    options['learning_rate']=learning_rate
    options['dropout']=dropout
    options['num_lstm_layers']=num_lstm_layers
    global net,device,xlm

    # check whether the current device has the cuda support
    is_cuda = torch.cuda.is_available()

    # initialize device to cpu and override if the cuda is set and available 
    device= torch.device("cpu")

    # is_cuda checks for availability of GPU on the user machine
    # cuda is a config parameter whether the user wants to use the GPU or not.
    if is_cuda and cuda:
        device = torch.device("cuda")
    # Todo 
    # Using the fixed xlm model only now(non trainable)
    xlm=XLMModel.from_pretrained(pretrained_model_name)
    logger.info("Successfully loaded the XLM model")
    xlm=xlm.to(device)

    # using the data_as_stream option choosing which model to use
    if data_as_stream==0:
        net=Model_pad(options)
    else:
        net=Model(options)
    
    # Moving the model to device 
    net=net.to(device)
    
    # The optimizer object currently hardcoded to Adam
    opt=optim.Adam(net.parameters(),lr=learning_rate)
    
    global criterion

    # Here the index should be tokenizer.pad_index 
    # For XLMTokenizer it is 2 and is not included in the Cross Entropy Loss calculation
    criterion=nn.CrossEntropyLoss(ignore_index=2)

    #initializing the minimum validation loss to a higher value
    min_val_loss=100

    #if the existing model is to be loaded then loading the model using this config params
    if load_model==1:
        # added map location so that model trained on gpu can be loaded on cpu too
        net.load_state_dict(torch.load(load_model_file,map_location=device))
        # Load optimizer only if training has to be done again not to be used if doing only eval 
        logger.info("Model successfully Loaded")
        if do_training==1:
            opt.load_state_dict(torch.load(load_optim_file))
            if data_as_stream==0:
                min_val_loss=val_func_pad()
            else:
                min_val_loss=val_func()
            logger.info("Initial Validation loss "+str(min_val_loss))
            # Initialing the minimum validation loss also so as when trianing again the best takes the loaded model as a baseline
        
        
    logger.info("Trainable Parameters")
    logger.info(count_parameters(net))
    print_model(net)

    # Weight_decay scheduler
    # set to min mode as loss is decreasing
    # Weight_decayfactor min_lr are the config parameters can be set there 
    scheduler = ReduceLROnPlateau(opt, mode='min',min_lr=min_learning_rate,factor=weight_decay_factor, patience=0, verbose=True,threshold=1e-4)
    if do_training==1:
        if data_as_stream==0:
            train_func_pad(scheduler,opt,min_val_loss)
        else:
            train_func(scheduler,opt,min_val_loss)
    if do_eval==1:
        if data_as_stream==0:
            test_func_pad()
        else:
            test_func()

def count_parameters(model):
    """
        Helper function to print the count of trainable parameters 
        Inputs:
        Model: the model object
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model(model):
    """
        Helper function to print the model architecture
        Inputs:
        model: The model object
    """
    #Todo
    #currently Hardcoded XLM
    logger.info(xlm.parameters)
    logger.info(model.parameters)
def norm_calc(model):
    """
        The helper function to calculate the norm of the gradients of the model's parameters. To be used in conjunction with gradient clipping
        Inputs:
        model:The Model object
        returns the norm as a floating point value
    """
    total_norm=0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print("Gradient norm "+str(total_norm))

def train_func_pad(scheduler,opt,min_val_loss):
    """
        The training function loop to be used when data_as_stream flag is not set
        Inputs:
        scheduler: The Learning Rate scheduler object initialized using the configuration parameters
        opt: The optimizer object (Here we are using Adam)
        min_val_loss: This is set and used when we are reusing the model for training and we initiazie it to the loss using the loaded model in model_setup function
    """
    logger.info("Started Training on "+run_name)

    #writer object for tensorboard
    writer = SummaryWriter()
    
    #starting the model training
    net.train()

    #main training loop
    for epc in range(epochs):
        trainloader=DataLoader(train_sents,batch_size=batch_size,shuffle=True)
        n_totals=0

        #variable to store the running sum of losses
        n_loss=0
        # To check whether the weight decay has been done at least once in this epoch
        weight_decay_flag=0
        for batch_idx,(inp,fwd,bwd,lt) in enumerate(trainloader):
            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
            lt=lt.to(device)
            xlm_inp=xlm(inp)
            out=net(xlm_inp,lt)
            bwd_sent=bwd.view(-1)
            fwd_sent=fwd.view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            opt.zero_grad()
            loss=criterion(out,targs)
            loss.backward()
            n_loss+=loss.item()
            n_totals+=inp.shape[0]
            opt.step()
            avg_loss=n_loss/(batch_idx+1)

            if (batch_idx+1) %steps_for_validation==0:
                   
                val_loss=val_func_pad()
                net.train()
                logger.info("After "+str(batch_idx+1)+" steps Training Avg_loss "+str(n_loss/(batch_idx+1))+"Training Avg_perplexity "+str(math.exp(n_loss/(batch_idx+1)))+" "+ "Validation Avg_loss "+str(val_loss)+"Validation Avg_perplexity "+str(math.exp(val_loss)))
                #setting weight decay for the larger datasets
                weight_decay_flag=1
                scheduler.step(val_loss)
                if val_loss<min_val_loss:
                    logger.info("Saved the model state best validation loss ")
                    min_val_loss=val_loss
                    model_path=files_dir+run_name+"/"+"model_best.pt"
                    optim_path=files_dir+run_name+"/"+"optim_best.pth"
                    torch.save(opt.state_dict(),optim_path)
                    torch.save(net.state_dict(),model_path)
                

        avg_loss=n_loss/(batch_idx+1)
        logger.info("Epoch "+str(epc+1))
        val_loss=val_func_pad()
        net.train()
        norm_calc(net)

        # If the weight decay has not been done even once in the epoch then do weight decay (meant for small datasets)
        if weight_decay_flag==0:
            scheduler.step(val_loss)
            if val_loss<min_val_loss:
                logger.info("Saved the model state best validation loss ")
                min_val_loss=val_loss
                model_path=files_dir+run_name+"/model_best.pt"
                optim_path=files_dir+run_name+"/optim_best.pth"
                torch.save(opt.state_dict(),optim_path)
                torch.save(net.state_dict(),model_path)            
        
        logger.info("Training Avg_loss "+str(avg_loss)+"Training Avg_perplexity "+str(math.exp(avg_loss))+" "+ "Validation Avg_loss "+str(val_loss)+"Validation Avg_perplexity "+str(math.exp(val_loss)))
        writer.add_scalar("Perplexity/Val",math.exp(val_loss),epc+1)
        writer.add_scalar('Perplexity/Train',math.exp(avg_loss), epc+1)
        
        # Saving the state after 10 epochs 
        if (epc+1) %10==1:
            logger.info("Saved the model state after "+str(epc+1)+" epochs")
            model_path=files_dir+run_name+"/"+"model_"+str(epc+1)+".pt"
            optim_path=files_dir+run_name+"/"+"optim_"+str(epc+1)+".pth"
            torch.save(opt.state_dict(),optim_path)
            torch.save(net.state_dict(),model_path)
              


def train_func(scheduler,opt,min_val_loss):

    """
        The training function loop to be used when data_as_stream flag is  set
        Inputs:
        scheduler: The Learning Rate scheduler object initialized using the configuration parameters
        opt: The optimizer object (Here we are using Adam)
        min_val_loss: This is set and used when we are reusing the model for training and we initiazie it to the loss using the loaded model in model_setup function
    """
    logger.info("Started Training on "+run_name)

    #writer object for tensorboard
    writer = SummaryWriter()
    
    #starting the model training
    net.train()

    #main training loop
    for epc in range(epochs):
        trainloader=DataLoader(train_sents,batch_size=batch_size,shuffle=True)
        n_totals=0
        n_loss=0
        # To check whether the weight decay has been done at least once in this epoch
        weight_decay_flag=0
        for batch_idx,(inp,fwd,bwd) in enumerate(trainloader):
            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
            xlm_inp=xlm(inp)
            out=net(xlm_inp)
            bwd_sent=bwd.view(-1)
            fwd_sent=fwd.view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            opt.zero_grad()
            loss=criterion(out,targs)
            loss.backward()
            n_loss+=loss.item()
            n_totals+=inp.shape[0]
            opt.step()
            avg_loss=n_loss/(batch_idx+1)

            if (batch_idx+1) %steps_for_validation==0:
                   
                val_loss=val_func()
                net.train()
                logger.info("After "+str(batch_idx+1)+" steps Training Avg_loss "+str(n_loss/(batch_idx+1))+"Training Avg_perplexity "+str(math.exp(n_loss/(batch_idx+1)))+" "+ "Validation Avg_loss "+str(val_loss)+"Validation Avg_perplexity "+str(math.exp(val_loss)))
                #setting weight decay for the larger datasets
                weight_decay_flag=1
                scheduler.step(val_loss)
                if val_loss<min_val_loss:
                    logger.info("Saved the model state best validation loss ")
                    min_val_loss=val_loss
                    model_path=files_dir+run_name+"/"+"model_best.pt"
                    optim_path=files_dir+run_name+"/"+"optim_best.pth"
                    torch.save(opt.state_dict(),optim_path)
                    torch.save(net.state_dict(),model_path)
                

        avg_loss=n_loss/(batch_idx+1)
        logger.info("Epoch "+str(epc+1))
        val_loss=val_func()
        net.train()
        norm_calc(net)

        # If the weight decay has not been done even once in the epoch then do weight decay (meant for small datasets)
        if weight_decay_flag==0:
            scheduler.step(val_loss)
            if val_loss<min_val_loss:
                logger.info("Saved the model state best validation loss ")
                min_val_loss=val_loss
                model_path=files_dir+run_name+"/model_best.pt"
                optim_path=files_dir+run_name+"/optim_best.pth"
                torch.save(opt.state_dict(),optim_path)
                torch.save(net.state_dict(),model_path)            
        
        logger.info("Training Avg_loss "+str(avg_loss)+"Training Avg_perplexity "+str(math.exp(avg_loss))+" "+ "Validation Avg_loss "+str(val_loss)+"Validation Avg_perplexity "+str(math.exp(val_loss)))
        writer.add_scalar("Perplexity/Val",math.exp(val_loss),epc+1)
        writer.add_scalar('Perplexity/Train',math.exp(avg_loss), epc+1)
        
        # Saving the state after 10 epochs 
        if (epc+1) %10==1:
            logger.info("Saved the model state after "+str(epc+1)+" epochs")
            model_path=files_dir+run_name+"/"+"model_"+str(epc+1)+".pt"
            optim_path=files_dir+run_name+"/"+"optim_"+str(epc+1)+".pth"
            torch.save(opt.state_dict(),optim_path)
            torch.save(net.state_dict(),model_path)
    

def val_func_pad():
    """
        Function for doing the validation when data_as_stream flag is not set
    """
    n_total=0
    n_loss=0
    # Can modify to do on a restricted set of the sentences
    global val_sents,net
    global val_factor

    # to make the training faster if a large validation set is there then restricting the datalength to be used
    # val_factor is a config parameter (0 to 1)
    n=int(len(val_sents)*val_factor)
    # setting the model in eval mode
    net.eval()
    
    valloader=DataLoader(val_sents,batch_size=batch_size,shuffle=True)
    with torch.no_grad():
        for batch_idx,(inp,fwd,bwd,lt) in enumerate(itertools.islice(valloader,int(n/batch_size))):
            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
            lt=lt.to(device)
            xlm_inp=xlm(inp)
            out=net(xlm_inp,lt)
            bwd_sent=bwd.view(-1)
            fwd_sent=fwd.view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            loss=criterion(out,targs)
            n_loss+=(loss.item())
        avg_loss=n_loss/(batch_idx+1)
        return avg_loss

def val_func():
    """
        Function for doing the validation when data_as_stream flag is  set
    """
    n_total=0
    n_loss=0
    # Can modify to do on a restricted set of the sentences
    global val_sents,net
    global val_factor

    # to make the training faster if a large validation set is there then restricting the datalength to be used
    # val_factor is a config parameter (0 to 1)
    n=int(len(val_sents)*val_factor)
    # setting the model in eval mode
    net.eval()
    
    valloader=DataLoader(val_sents,batch_size=batch_size,shuffle=True)
    with torch.no_grad():
        for batch_idx,(inp,fwd,bwd) in enumerate(itertools.islice(valloader,int(n/batch_size))):
            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
            xlm_inp=xlm(inp)
            out=net(xlm_inp)
            bwd_sent=bwd.view(-1)
            fwd_sent=fwd.view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            loss=criterion(out,targs)
            n_loss+=(loss.item())
        avg_loss=n_loss/(batch_idx+1)
        return avg_loss


def test_func_pad():
    """
        Function for doing the testing  when data_as_stream flag is not set
    """
    n_total=0
    n_loss=0
    testloader=DataLoader(test_sents,batch_size=batch_size)
    logger.info("Started Testing on "+test_file)
    # setting the model in eval mode for testing 
    net.eval()
    with torch.no_grad():
        for batch_idx,(inp,fwd,bwd,lt) in enumerate(testloader):
            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
            lt=lt.to(device)
            xlm_inp=xlm(inp)

            out=net(xlm_inp,lt)
            bwd_sent=bwd.view(-1)
            fwd_sent=fwd.view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            loss=criterion(out,targs)
            n_loss+=(loss.item())

        avg_loss=n_loss/(batch_idx+1)
    

    logger.info(" Test Loss"+str(avg_loss)+" Test Perplexity "+str(math.exp(avg_loss)))    

def test_func():
    """
        Function for doing the testing when data_as_stream flag is  set
    """
    n_total=0
    n_loss=0
    testloader=DataLoader(test_sents,batch_size=batch_size,shuffle=True)
    net.eval()
    logger.info("Started Testing on "+test_file)
    # setting the model in eval mode for testing 
    net.eval()
    with torch.no_grad():
        for batch_idx,(inp,fwd,bwd) in enumerate(testloader):
            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
            xlm_inp=xlm(inp)
            out=net(xlm_inp)
            bwd_sent=bwd.view(-1)
            fwd_sent=fwd.view(-1)
            targs=torch.cat([fwd_sent,bwd_sent],dim=0)
            loss=criterion(out,targs)
            n_loss+=(loss.item())
        avg_loss=n_loss/(batch_idx+1)
    

    logger.info(" Test Loss"+str(avg_loss)+" Test Perplexity "+str(math.exp(avg_loss)))    


if __name__=="__main__":

    configsetters()
    model_setup()
