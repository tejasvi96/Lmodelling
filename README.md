# Quality Estimation in Machine Translation

# Background

In this project we consider the problem of unsupervised quality estimation in subtitles where we evaluate the sentences on the fluency aspect. The fluency aspect basically means that how likely a human is to write such a sentence. This project takes motivation from [here](https://www.aclweb.org/anthology/2020.tacl-1.35.pdf)

e.g 
1) He is eating cake. 
2) He cake eating is.

Score(1) > Score(2) because its composed of more meaningful units and is grammatrically correct in the order of words.

We aim to relate fluency with perplexity which is the exponential of the language modelling loss. Ideally higher perplexity scores mean lesser fluency.



We realize that the language modelling aims to produce more statistically significant sentences and the words which are more common in corpus have higher probability of occurence. Thus this is different from the fluency as it involves statistical significance of the words. 

e.g
1) He loves to eat chocolate 
2) He loves to eat cheese.



Perplexity(1)<Perplexity(2) , This is because the word chocolate is more statistically significant and more likely to occur in the training corpus. Thus sentence 1 has a higher probability of occurence which in turn means lesser perplexity scores. But on the fluency aspect both these sentences are similar and thus to arrive at a common score across different words we need to normalize the perplexity score by decreasing the unigram probabilities of each word to eliminate the effect of statistical significance. This motivation is adapted from [here](https://arxiv.org/abs/1809.08731).

# Model 

We make use of the pretrained  XLM models and try to build an encoder on top of it for the efficient translation using the context based transformer models A raw input sentence is taken and fed to the XLM module which uses a positional embedding which maps the entire sequence , specifically its corresponding tokens to an embedding
space. The embedding space is pretrained and if a token is present then it assigns the corresponding index and if not then assigns it the unk token. The embedder is basically a
table which stores all the words present in the corpus and their corresponding high dimensional representations. This is passed through the Encoder module which is composed
of different attention heads. Each encoder applies self attention to each of the word and combines the input with the self attention and pass this to the feed forward network on which again a normalization layer is applied which passes the context to the next encoder. Proceeding in this way all the encoder states are stored and concatenated and passed to the initial state of the decoder module as the context. The decoder also applies the similar operations and makes use of the context and positional encoding to generate a sequence or final high dimensional output which can be used based on the task at hand. On top of the XLM we add a Bi LSTM layer which uses bidirectionalism through a forward
and a backward layer to produce the final output of size same as the input hidden dimension. This is fed through a linear fully connected layer which produces the outputs as probabilities over the entire vocabulary.

Architecture of a standard Transformer Model.

![Images.](https://github.com/tejasvi96/Lmodelling/blob/master/Transformer_architecture.png?raw=True)


Architecture of the language model that we use

![Images.](https://github.com/tejasvi96/Lmodelling/blob/master/Language_model.png?raw=True)

# Experiment

This problem was considered in the domain of movie subtitles where a bad movie subtitle can hamper the quality of the user viewing. Thus this unsupervised score can be used to identify the low quality subtitles without much human effort and can be manually rectified or passed to another model to suggest the changes.

The standard OpenSubtitles corpus is considered for this problem.

The major configurations can be done in the config.yaml file.

```
main:
  run_name: "Modelling_open_combined"
  max_seq_len: 30
  batch_size: 1
  hid_size: 1280
  vocab_size: 200000
  preprocessed: 0
  files_dir: "/Users/stejasv/pyenv/"
  train_file: "opensub_de.txt"
  test_file: "opensub_de.txt"
  val_file: "opensub_de.txt"
  load_tsv: 0
  epochs: 10
  steps_for_validation: 10000
  learning_rate: 0.0003
  min_learning_rate: 0.000001
  weight_decay_factor: 0.5
  pretrained_model: "xlm-mlm-17-1280"
  load_model: 0
  load_model_file: "/Users/stejasv/pyenv/Modelling_open_combined/model_best.pt"
  load_optim_file: "/Users/stejasv/pyenv/Modelling_source_data/optim_1.pth"
  val_factor: 0.2
  data_as_stream: 1
  dropout: 0.2
  do_training: 1
  do_eval: 1
  cuda: 1
  vocab_file: "/Users/stejasv/pyenv/Modelling_open_combined/vocab_opensub_de.txt"
  num_lstm_layers: 1
```

The base model that is being used is XLM model and the corresponding language specific model variant can be set in the config file.

To run the expriment use the code

```
python final_model_restricted.py config.yaml
```

