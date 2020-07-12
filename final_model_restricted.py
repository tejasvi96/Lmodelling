import logging
import itertools
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='run.log',format='%(asctime)s %(message)s',filemode='a',level=logging.INFO)
# logger = logging.logging("logs")  # get the root logging
import hydra
import torch
import os
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
from transformers import XLMConfig,XLMTokenizer,XLMModel
from allennlp.modules.elmo_lstm import ElmoLstm
from torch.utils.data import TensorDataset,DataLoader
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
batch_size=0
max_seq_len=0
files_dir=""
train_file=""
test_file=""
val_file=""
hid_size=0
vocab_size=0
epochs=0
learning_rate=0
train_sents=[]
val_sents=[]
test_sents=[]
net=""
criterion=""
xlm=""
run_name=""
pretrained_model_name=""
load_model=0
load_model_file=""
device=""
val_factor=0.1
class BiLMEncoder(ElmoLstm):
    """Wrapper around BiLM to give it an interface to comply with SentEncoder
    See base class: ElmoLstm
    """

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.hidden_size * 2

@hydra.main(config_path="config.yaml")
def configsetters(cfg):
    print(cfg.pretty())
    global max_seq_len,batch_size,files_dir,train_file,test_file,val_file,hid_size,vocab_size,epochs,learning_rate,pretrained_model_name,xlm,run_name,load_model,load_model_file,val_factor

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
    pretrained_model_name=cfg.main.pretrained_model
    run_name=cfg.main.run_name
    load_model=cfg.main.load_model
    load_model_file=cfg.main.load_model_file
    val_factor=cfg.main.val_factor

    global train_sents,test_sents,val_sents

    train_loaded_file=files_dir+train_file+"_preprocessed.pt"
    test_loaded_file=files_dir+test_file+"_preprocessed.pt"
    val_loaded_file=files_dir+val_file+"_preprocessed.pt"

    if cfg.main.preprocessed==1:
        logging.info("Loading the existing data files ")
        train_sents=torch.load(train_loaded_file)
        test_sents=torch.load(test_loaded_file)
        val_sents=torch.load(val_loaded_file)
        vocab_file=files_dir+run_name+"_vocab_"+train_file
        with open(vocab_file,'r',encoding="utf-8") as fp:
            data=fp.readlines()
            vocab_size=int(data[0].split("\n")[0])
    else:
        preprocess()

def preprocess():
    
    cwd=os.getcwd()+"/"
    txtfiles=[]
    global max_seq_len,batch_size,files_dir,train_file,test_file,val_file,hid_size,vocab_size,epochs,learning_rate,pretrained_model_name,xlm,run_name,load_model,load_model_file

    # max_seq_len=cfg.main.max_seq_len
    # batch_size=cfg.main.batch_size
    # files_dir=cfg.main.files_dir
    # train_file=cfg.main.train_file
    # test_file=cfg.main.test_file
    # val_file=cfg.main.val_file
    # hid_size=cfg.main.hid_size
    # vocab_size=cfg.main.vocab_size
    # epochs=cfg.main.epochs
    # learning_rate=cfg.main.learning_rate
    # pretrained_model_name=cfg.main.pretrained_model
    # run_name=cfg.main.run_name
    # load_model=cfg.main.load_model
    # load_model_file=cfg.main.load_model_file

    # Loading the pretrianined XLM model 

    train_filepath=files_dir+train_file
    test_filepath=files_dir+test_file
    val_filepath=files_dir+val_file
    
    tokenizer=XLMTokenizer.from_pretrained(pretrained_model_name)
    txtfiles.append(train_filepath)
    txtfiles.append(test_filepath)
    txtfiles.append(val_filepath)
    global val_sents,train_sents,test_sents
    for ind,file in enumerate(txtfiles):
        filepath=file
        with open(filepath,'r',encoding='utf-8') as fp:
            data=fp.read()
        sents=data.split("\n")

        tokenized_sents=[i for i in range(len(sents))]
        # Do this only for the train file
        if ind==0:
            token_inds=set()
            for i,s in enumerate(sents):
                temp=tokenizer.encode(s,add_special_tokens=False)
                [token_inds.add(ind) for ind in temp]

            spl_tokens=[tokenizer.special_tokens_map[k] for (k) in tokenizer.special_tokens_map.keys() if k!='additional_special_tokens']
            spl_tokens_inds=tokenizer.convert_tokens_to_ids(spl_tokens)
            spl_tokens_inds=set(spl_tokens_inds)
            [token_inds.add(i) for i in spl_tokens_inds]
            restricted_vocab={}
            token_inds=list(token_inds)
            token_inds=sorted(token_inds)
            for i,key in enumerate(token_inds):
                restricted_vocab[key]=i

            # Hyperpprameter is setup    
            vocab_size=len(restricted_vocab)
            vocab_file=files_dir+run_name+"_vocab_"+train_file
            with open(vocab_file,'w',encoding="utf-8") as fp:
                fp.write(str(len(restricted_vocab)))
                fp.write("\n")
                kys=list(restricted_vocab.keys())
                vals=tokenizer.convert_ids_to_tokens(kys)
                for k in vals:
                    fp.write(k+"\n")

    
            # with open("")
            # print(restricted_vocab)
        for i,sent in enumerate(sents):
            tokenized_sents[i]=tokenizer.encode(sent)
        # arr_tokenized_sents
        arr_retokenized_sents=[]
        for i,sent in enumerate(tokenized_sents):
            temp=[]
            for j in sent:
                temp.append(restricted_vocab[j] if j in restricted_vocab.keys() else tokenizer.unk_token_id)
            arr_retokenized_sents.append(temp)


        backward_sents=[i for i in range(len(sents))]
        for i in range(len(sents)):
            backward_sents[i]=[arr_retokenized_sents[i][-1]]+arr_retokenized_sents[i][1:-1]+[arr_retokenized_sents[i][0]]
        input_sent_lens=[]
        for i,sent in enumerate(tokenized_sents):
        #     print(len(sent))
        #     Added +2 to accomodate for the start and ending token
            # input_sent_lens.append(len(sent))
            input_sent_lens.append(len(sent) if len(sent)<max_seq_len else max_seq_len)

    #     max_seq_len=max(input_sent_lens)
        # print(max_seq_len)
        # Here 1 is the </s>

        # IDeally keep the padding token id same as in the original namespace of restircted vocabulary
        for i in range(len(tokenized_sents)):
        #     print(len(tokenized_sents[i]))
            if len(tokenized_sents[i])<max_seq_len:
                tokenized_sents[i]=tokenized_sents[i]+[tokenizer.pad_token_id]*(max_seq_len-len(tokenized_sents[i]))
            else:
                tokenized_sents[i]=tokenized_sents[i][:max_seq_len-1]+[1]
        for i in range(len(arr_retokenized_sents)):
        #     print(len(tokenized_sents[i]))
            if len(arr_retokenized_sents[i])<max_seq_len:
                arr_retokenized_sents[i]=arr_retokenized_sents[i]+[tokenizer.pad_token_id]*(max_seq_len-len(arr_retokenized_sents[i]))
            else:
                arr_retokenized_sents[i]=arr_retokenized_sents[i][:max_seq_len-1]+[1]
        for i in range(len(backward_sents)):
        #     print(len(tokenized_sents[i]))
            if len(backward_sents[i])<max_seq_len:
                backward_sents[i]=backward_sents[i]+[tokenizer.pad_token_id]*(max_seq_len-len(backward_sents[i]))
            else:
                backward_sents[i]=backward_sents[i][:max_seq_len-1]+[0]

        input_sent_lens=np.array(input_sent_lens)
        arr_tokenized_sents=np.array(tokenized_sents,dtype=int)
        arr_backward_sents=np.array(backward_sents,dtype=int)
        arr_retokenized_sents=np.array(arr_retokenized_sents,dtype=int)
        arr_sents=TensorDataset(torch.from_numpy(arr_tokenized_sents),torch.from_numpy(arr_retokenized_sents),torch.from_numpy(arr_backward_sents),torch.from_numpy(input_sent_lens))
    #     Hyperpprameter
        # batch_size=cfg.main.batch_size

        train_loaded_file=files_dir+train_file+"_preprocessed.pt"
        test_loaded_file=files_dir+test_file+"_preprocessed.pt"
        val_loaded_file=files_dir+val_file+"_preprocessed.pt"

        if ind==0:
            train_sents=arr_sents
            torch.save(train_sents,train_loaded_file)
            trainloader=DataLoader(arr_sents,batch_size=batch_size,shuffle=True)
        elif ind==1:
            test_sents=arr_sents
            torch.save(test_sents,test_loaded_file)
            testloader=DataLoader(arr_sents,batch_size=batch_size,shuffle=True)
        else:
            val_sents=arr_sents
            torch.save(val_sents,val_loaded_file)
            valloader=DataLoader(arr_sents,batch_size=batch_size,shuffle=True)


class Model(nn.Module):
    def __init__(self,options):
        super(Model, self).__init__()
        logging.info(options)
        # print(options)
        self.bilm=BiLMEncoder(options['hid_size'],options['hid_size'],options['hid_size'],1)
        self.lin=torch.nn.Linear(options['hid_size'],vocab_size)
    def forward(self,enc_embedding,lt):
#         print(enc_embedding[0].shape)
        # mask=torch.zeros((lt.shape[0],max_seq_len))
        # for j,l in enumerate(lt):
        #     mask[j,:l]=torch.ones(l) 
        mask=torch.zeros((lt.shape[0],max_seq_len)).to(device)
        for j,l in enumerate(lt):
            for k in range(l):
                mask[j,k]=1
        enc=self.bilm(enc_embedding[0],mask)
        
        fwd,bwd=enc[:,:,:,:hid_size],enc[:,:,:,hid_size:]
        logits_fwd=self.lin(fwd).view(lt.shape[0] * max_seq_len, -1)
        logits_bwd=self.lin(bwd).view(lt.shape[0] * max_seq_len, -1)

        logits=torch.cat((logits_fwd,logits_bwd),dim=0)

        return logits

# @hydra.main(config_path="config.yaml")
def mod():
    options={}
    # global max_seq_len,batch_size,files_dir,train_file,test_file,val_file,hid_size,vocab_size,epochs,learning_rate
    options['max_seq_len']=max_seq_len
    options['batch_size']=batch_size
    options['vocab_size']=vocab_size
    options['epochs']=epochs
    options['hid_size']=hid_size
    options['learning_rate']=learning_rate
    global net,device,xlm
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #add the support of the Loading existing model and existing optimizer
    #  
    xlm=XLMModel.from_pretrained(pretrained_model_name)
    logging.info("Successfully loaded the XLM model")
    xlm=xlm.to(device)
    net=Model(options)
    net=net.to(device)
    if load_model==1:
        net.load_state_dict(torch.load(files_dir+load_model_file))
        logging.info("Model successfully Loaded")
    logging.info("Trainable Parameters")
    # print("Trainable Params")
    logging.info(count_parameters(net))
    # print(count_parameters(net))
    print_model(net)
    
    writer = SummaryWriter()

    net.train()
    opt=optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)
    global criterion
    criterion=nn.CrossEntropyLoss(ignore_index=2)
    logging.info("Started Training")
    # print("Started Training")
    for epc in range(epochs):
        trainloader=DataLoader(train_sents,batch_size=batch_size)
        n_totals=0
        n_loss=0
        min_val_loss=100
        for batch_idx,(inp,fwd,bwd,lt) in enumerate(trainloader):
    #         print(lt)
            # print(inp[0],fwd[0],bwd[0])
            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
            lt=lt.to(device)
            xlm_inp=xlm(inp)
            out=net(xlm_inp,lt)
            bwd_sent=bwd
            fwd_sent=fwd
            targs=torch.cat((fwd_sent,bwd_sent),dim=1)
            targs=targs.view(lt.shape[0]*max_seq_len*2,-1)
            targs=targs.squeeze()
            opt.zero_grad()
            loss=criterion(out,targs)
            loss.backward()
            n_loss+=loss.item()
            n_totals+=lt.shape[0]
            opt.step()
            # print(loss.item())
        # print(batch_idx)
        avg_loss=n_loss/batch_idx
        logging.info("Epoch "+str(epc))
        
        # print("Epoch "+str(epc))
        # print("Training Avg_loss "+str(avg_loss)+"Training Avg_perplexity "+str(math.exp(avg_loss)))
        # print(math.exp(avg_loss))
        val_loss=val_func()
        if val_loss<min_val_loss:
            logging.info("Saved the model state best validation loss ")
            min_val_loss=val_loss
            model_path=files_dir+run_name+"_model_best.pt"
            optim_path=files_dir+run_name+"_optim_best.pth"
            torch.save(opt.state_dict,optim_path)
            torch.save(net.state_dict(),model_path)
        # print("Validation Avg_loss "+str(val_loss)+"Training Avg_perplexity "+str(math.exp(val_loss)))
        logging.info("Training Avg_loss "+str(avg_loss)+"Training Avg_perplexity "+str(math.exp(avg_loss))+" "+ "Validation Avg_loss "+str(val_loss)+"Training Avg_perplexity "+str(math.exp(val_loss)))
        writer.add_scalar("Perplexity/Val",math.exp(val_loss),epc)
        writer.add_scalar('Perplexity/Train',math.exp(avg_loss), epc)
        # Saving the state after 10 epochs 
        if epc %10==0:
            logging.info("Saved the model state after 10 epochs")
            model_path=files_dir+run_name+"_model_"+str(epc)+".pt"
            optim_path=files_dir+run_name+"_optim_"+str(epc)+".pth"
            torch.save(opt.state_dict,optim_path)
            torch.save(net.state_dict(),model_path)           
    test_func()
#         print(loss.item())

def count_parameters(model):
    # l=[p for p in model.parameters() if p.requires_grad]
    # print(l)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model(model):
    logging.info(xlm.parameters)
    logging.info(model.parameters)
    # print(xlm.parameters)
    # print(model.parameters)


def val_func():
    n_total=0
    n_loss=0
    # Can modify to do on a restricted set of the sentences
    global val_sents,net
    global val_factor
    n=int(len(val_sents)*val_factor)

    # print(val_sents)
    valloader=DataLoader(val_sents,batch_size=batch_size,shuffle=True)
    with torch.no_grad():
        for batch_idx,(inp,fwd,bwd,lt) in enumerate(itertools.islice(valloader,int(n/batch_size))):

            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
            lt=lt.to(device)
            xlm_inp=xlm(inp)
            out=net(xlm_inp,lt)
            bwd_sent=bwd
            fwd_sent=fwd

            targs=torch.cat((fwd_sent,bwd_sent),dim=1)
            targs=targs.view(lt.shape[0]*max_seq_len*2,-1)
            targs=targs.squeeze()
            loss=criterion(out,targs)
            n_loss+=(loss.item())
        avg_loss=n_loss/batch_idx
        return avg_loss


def test_func():
    n_total=0
    n_loss=0
    testloader=DataLoader(test_sents,batch_size=batch_size,shuffle=True)
    with torch.no_grad():
        for batch_idx,(inp,fwd,bwd,lt) in enumerate(testloader):

            inp=inp.to(device)
            bwd=bwd.to(device)
            fwd=fwd.to(device)
            lt=lt.to(device)
            xlm_inp=xlm(inp)
            out=net(xlm_inp,lt)
            bwd_sent=bwd
            fwd_sent=fwd
            targs=torch.cat((fwd_sent,bwd_sent),dim=1)
            targs=targs.view(lt.shape[0]*max_seq_len*2,-1)
            targs=targs.squeeze()
            loss=criterion(out,targs)
            n_loss+=(loss.item())
        avg_loss=n_loss/batch_idx
    logging.info("Started Testing")
    logging.info(" Test Loss"+str(avg_loss)+"Test Perplexity "+str(math.exp(avg_loss)))    
    # print(" Test Loss",str(avg_loss),"Test Perplexity ",str(math.exp(avg_loss)))

if __name__=="__main__":
    
    configsetters()
    mod()