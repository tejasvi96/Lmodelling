# global variables
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
load_tsv=0
load_optim_file=""
load_model_file=""
device=""
val_factor=0.1
min_learning_rate=0.000001
weight_decay_factor=0
steps_for_validation=0
data_as_stream=0
dropout=0
do_training=0
do_eval=0
cuda=0
vocab_file=""
num_lstm_layers=1