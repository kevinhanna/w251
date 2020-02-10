
Now we will continue on the [Conversation AI](https://conversationai.github.io/) dataset seen in [week 4 homework and lab](https://github.com/MIDS-scaling-up/v2/tree/master/week04). 
We shall use a version of pytorch BERT for classifying comments found at [https://github.com/huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).  

The original implementation of BERT is optimised for TPU. Google released some amazing performance improvements on TPU over GPU, for example, see [here](https://medium.com/@ranko.mosic/googles-bert-nlp-5b2bb1236d78) - *BERT relies on massive compute for pre-training ( 4 days on 4 to 16 Cloud TPUs; pre-training on 8 GPUs would take 40–70 days).*. In response, Nvidia released [apex](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/), which gave mixed precision training. Weights are stored in float32 format, but calculations, like forward and backward propagation happen in float16 - this allows these calculations to be made with a [4X speed up](https://github.com/huggingface/pytorch-pretrained-BERT/issues/149).  

We shall apply BERT to the problem for classifiying toxicity, using apex from Nvidia. We shall compare the impact of hardware by running the model on a V100 and P100 and comparing the speed and accuracy in both cases.   

This script relies heavily on an existing [Kaggle kernel](https://www.kaggle.com/yuval6967/toxic-bert-plain-vanila) from [yuval r](https://www.kaggle.com/yuval6967). 
  
*Disclaimer: the dataset used contains text that may be considered profane, vulgar, or offensive.*


```

```


```
import sys, os
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
%load_ext autoreload
%autoreload 2
%matplotlib inline
from tqdm import tqdm, tqdm_notebook
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings(action='once')
import pickle
from apex import amp
import shutil
```


```
# Let's activate CUDA for GPU based operations
device=torch.device('cuda')
```

Change the PATH variable to whereever your `week06/hw` directory is located.  
**For the final run we would like you to have a train_size of at least 1 Million rows, and a valid size of at least 500K rows. When you first run the script, feel free to work with a reduced train and valid size for speed.** 


```
# In bert we need all inputs to have the same length, we will use the first 220 characters. 
MAX_SEQUENCE_LENGTH = 220
SEED = 1234
# We shall run a single epoch (ie. one pass over the data)
EPOCHS = 1
PATH = '/root/v2/week06/hw' # /root/v2/week06/hw"
DATA_DIR = os.path.join(PATH, "data")
WORK_DIR = os.path.join(PATH, "workingdir")

# Validation and training sizes are here. 
#train_size= 10000 # 1000000 
#valid_size= 5000  # 500000

train_size= 1000000 
valid_size= 500000
```

This should be the files you downloaded earlier when you ran `download.sh`


```
os.listdir(DATA_DIR)
```




    ['download.sh',
     'cased_L-12_H-768_A-12',
     'test.csv',
     'train.csv',
     'uncased_L-12_H-768_A-12']



We shall install pytorch BERT implementation.   
If you would like to experiment with or view any code (purely optional, and not graded :) ), you can copy the files from the repo https://github.com/huggingface/pytorch-pretrained-BERT  


```
%%capture
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert import BertConfig
```

We shall now load the model. When you run this, comment out the `capture` command to understand the archecture.


```
%%capture
# Translate model from tensorflow to pytorch
BERT_MODEL_PATH = os.path.join(DATA_DIR, 'uncased_L-12_H-768_A-12')
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
                            os.path.join(BERT_MODEL_PATH, 'bert_model.ckpt'),
                            os.path.join(BERT_MODEL_PATH, 'bert_config.json'), 
                            os.path.join(WORK_DIR, 'pytorch_model.bin'))

shutil.copyfile(os.path.join(BERT_MODEL_PATH, 'bert_config.json'), \
                os.path.join(WORK_DIR, 'bert_config.json'))
# This is the Bert configuration file
bert_config = BertConfig(os.path.join(WORK_DIR, 'bert_config.json'))
```

Bert needs a special formatting of sentences, so we have a sentence start and end token, as well as separators.   
Thanks to this [script](https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming) for a fast convertor of the sentences.


```
def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)
```

Now we load the BERT tokenizer and convert the sentences.


```
%%time
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
train_all = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).sample(train_size+valid_size,random_state=SEED)
print('loaded %d records' % len(train_all))

# Make sure all comment_text values are strings
train_all['comment_text'] = train_all['comment_text'].astype(str) 

sequences = convert_lines(train_all["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)
train_all=train_all.fillna(0)
```

    loaded 1500000 records



    HBox(children=(IntProgress(value=0, max=1500000), HTML(value='')))


    
    33724
    CPU times: user 33min 34s, sys: 9.55 s, total: 33min 43s
    Wall time: 33min 34s


Let us look at how the tokenising works in BERT, see below how it recongizes misspellings - words the model never saw. 


```
train_all[["comment_text", 'target']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment_text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>458232</th>
      <td>It's difficult for many old people to keep up ...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>272766</th>
      <td>She recognized that her tiny-handed husband is...</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>339129</th>
      <td>HPHY76,\nGood for you for thinking out loud, w...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>773565</th>
      <td>And I bet that in the day you expected your Je...</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>476233</th>
      <td>Kennedy will add a much needed and scientifica...</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Lets tokenize some text (I intentionally mispelled some words to check berts subword information handling)


```
text = 'Hi, I am learning new things in w251 about deep learning the cloud and teh edge.'
tokens = tokenizer.tokenize(text)
' '.join(tokens)
```




    'hi , i am learning new things in w ##25 ##1 about deep learning the cloud and te ##h edge .'



Added start and end token and convert to ids. This is how it is fed into BERT.


```
tokens = ["[CLS]"] + tokens + ["[SEP]"]
input_ids = tokenizer.convert_tokens_to_ids(tokens)
' '.join(map(str, input_ids))
```




    '101 7632 1010 1045 2572 4083 2047 2477 1999 1059 17788 2487 2055 2784 4083 1996 6112 1998 8915 2232 3341 1012 102'



When BERT converts this sentence to a torch tensor below is shape of the stored tensors.  
We have 12 input tensors, while the sentence tokens has length 23; where are can you see the 23 tokens in the tensors ?... **Feel free to post in slack or discuss in class**


```
# put input on gpu and make prediction
bert = BertModel.from_pretrained(WORK_DIR).cuda()
bert_output = bert(torch.tensor([input_ids]).cuda())

print('Sentence tokens {}'.format(tokens))
print('Number of tokens {}'.format(len(tokens)))
print('Tensor shapes : {}'.format([b.cpu().detach().numpy().shape for b in bert_output[0]]))
print('Number of torch tensors : {}'.format(len(bert_output[0])))
```

    Sentence tokens ['[CLS]', 'hi', ',', 'i', 'am', 'learning', 'new', 'things', 'in', 'w', '##25', '##1', 'about', 'deep', 'learning', 'the', 'cloud', 'and', 'te', '##h', 'edge', '.', '[SEP]']
    Number of tokens 23
    Tensor shapes : [(1, 23, 768), (1, 23, 768), (1, 23, 768), (1, 23, 768), (1, 23, 768), (1, 23, 768), (1, 23, 768), (1, 23, 768), (1, 23, 768), (1, 23, 768), (1, 23, 768), (1, 23, 768)]
    Number of torch tensors : 12


As it is a binary problem, we change our target to [0,1], instead of float.   
We also split the dataset into a training and validation set, 


```
train_all['target']=(train_all['target']>=0.5).astype(float)
# Training data - sentences
X = sequences[:train_size] 
# Target - the toxicity. 
y = train_all[['target']].values[:train_size]
X_val = sequences[train_size:]                
y_val = train_all[['target']].values[train_size:]
```


```
test_df=train_all.tail(valid_size).copy()
train_df=train_all.head(train_size)
```

**From here on in we would like you to run BERT.**   
**Please do rely on the script available -  [Kaggle kernel](https://www.kaggle.com/yuval6967/toxic-bert-plain-vanila) from [yuval r](https://www.kaggle.com/yuval6967) - for at least the first few steps up to training and prediction.**


```
import time
start_time = time.time()

#Set the seeds for my code
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def time_print():
    display("** Notebook running %4.2f seconds **" % (time.time() - start_time))
    
# Starting time of HW code
time_print()
```




    <torch._C.Generator at 0x7fe5a4852f90>




    '** Notebook running 0.00 seconds **'



**1)**   
**Load the training set to a training dataset. For this you need to load the X sequences and y objects to torch tensors**   
**You can use `torch.utils.data.TensorDataset` to input these into a train_dataset.**


```
# Training data creations
#training_dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
training_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))
```

**2)**  
**Set your learning rate and batch size; and optionally random seeds if you want reproducable results**   
**Load your pretrained BERT using `BertForSequenceClassification`**   
**Initialise the gradients and place the model on cuda, set up your optimiser and decay parameters**
**Initialise the model with `apex` (we imprted this as `amp`) for mixed precision training**


```
%%time
#model = BertForSequenceClassification.from_pretrained('uncased_L-12_H-768_A-12', num_labels = 2,
#config = BertConfig.from_json_file('/root/v2/week06/hw/workingdir/bert_config.json')

model = BertForSequenceClassification.from_pretrained(WORK_DIR, num_labels=1)
model.zero_grad()
#model.cuda()
model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
```

    CPU times: user 6.6 s, sys: 696 ms, total: 7.3 s
    Wall time: 3.25 s


**3)**  
**Start training your model by iterating through batches in a single epoch of the data**


```
%%time
EPOCHS = 1

lr=2e-5
batch_size = 32
accumulation_steps=2
torch.backends.cudnn.deterministic = True

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

train = training_dataset

num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
model=model.train()

tq = tqdm_notebook(range(EPOCHS))
for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
    optimizer.zero_grad()   # Bug fix - thanks to @chinhuic
    for i,(x_batch, y_batch) in tk0:
#        optimizer.zero_grad()
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)
```


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=31250), HTML(value='')))


    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    
    CPU times: user 3h 47min 30s, sys: 2h 16min 58s, total: 6h 4min 28s
    Wall time: 6h 4min 21s


**4)**  
**Store your trained model to disk, you will need it if you choose section 8C.**


```
output_model_file = "bert_pytorch.bin"
torch.save(model.state_dict(), output_model_file)
```

**5)**   
**Now make a prediction for your validation set.**  


```
# Leave commented out unless trying to skip above code
#model = BertForSequenceClassification(bert_config,num_labels=len(y_columns))
#model.load_state_dict(torch.load(output_model_file ))
#model.to(device)
```


```
%%time

for param in model.parameters():
    param.requires_grad=False
    
model.eval()

valid_preds = np.zeros((len(X_val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

tk0 = tqdm_notebook(valid_loader)
for i,(x_batch,)  in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    valid_preds[i*32:(i+1)*32]=pred[:,0].detach().cpu().squeeze().numpy()
    
```


    HBox(children=(IntProgress(value=0, max=15625), HTML(value='')))


    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    
    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


**6)**  
**In the yuval's kernel he get a metric based on the metric for the jigsaw competition - it is quite complicated. Instead, we would like you to measure the `AUC`, similar to how you did in homework 04. You can compare the results to HW04**  
*A tip, if your score is lower than homework 04 something is wrong....*


```
%%time
from sklearn import metrics

display(metrics.roc_auc_score(y_val, valid_preds))
```


    0.9701356732714134


    CPU times: user 412 ms, sys: 828 ms, total: 1.24 s
    Wall time: 157 ms


**7)**  
**Can you show/print the validation sentences predicted with the highest and lowest toxicity ?**


```
top = valid_preds.argsort()[-10:]
bottom = np.flip(valid_preds.argsort()[:10])

val_comments = train_all[["comment_text"]].values[train_size:]

print("Highest Toxicity\n")
print(val_comments[top])

print("\n\nLowest Toxicity\n")
print(np.flip(val_comments[bottom]))


```

    Highest Toxicity
    
    [["Wow. No compassion here, just stupid jokes?  These two people are dead. Our 'society' is sick."]
     ["Absolutely.  You're an idiot.\nI would consider it a subhuman being killed.\nzazzle.com/FirstPrinciples/products?rf=238518351914519699"]
     ['Just line the border with military and start\nshooting them. If there are dissenters, shoot\nthem too. Chuch stupid should be thrown\nin with Pelosi and do the same. Destroy all\nlefties and nuke NK and Iran. Anyone one\nelse, the same.']
     ["He's as dumb as his mother. Puts\npassion before reason. He is a \ntraitor to Canada. And all these\nrefugees will be killing you and\nyour children as trucastro is all\nabout taking the country over\nthe way his lover screwdrove\ncuba."]
     ['save the bs,these two idiot leaders are killing jobs down the line pal. but at least the usa now has a real leader coming in Trump who will kill this nonsense']
     ['exactly. wait until these worthless punks start burning up Cleveland. they will cinch the election for trumpster after real americans watch these idiots on tv after a long days work.']
     ['Since the Right Honourable Stephen Harper is now gone and you have that idiot Trudeau I think the Jews should be worried here too.']
     ['Look at how well things are working out in Europe with the refugees. Now those idiots want that here. Kick them all out of office']
     ['you are the latest example of an idiot. trump is no fascist in any way, the left is with its endless calls to stop free speech etc, trump is controlling nothing, he is freeing the usa from regressive government.  YOU must find a way to get some help with your sickness.']
     ['So trudeau just told all of Canada that Canadian laws only apply to non muslims!\n\nmuslims are free to ignore Canadian law!\n\nOnce again trudeau proves he is just a useful idiot for islam!']]
    
    
    Lowest Toxicity
    
    [['"some of our biggest trading partners are allowed to continue polluting the atmosphere on a massive scale, while receiving billions to help them do it from western nations"\n\nAnd where does the Paris Agreement state anything like this, Jeff? Link to a reputable source, please. Thanks in advance!\n\n(Here\'s a starting point for you, if you\'d like to actually read the agreement: http://unfccc.int/paris_agreement/items/9485.php)']
     ["I can relate to this. A few years ago on one of my trips to the UK my mother-in-law asked me to review her investments as she was now dealing with a new advisor following the death of my father-in-law. She was about 80 at the time.\n\nShe had a couple of dividend paying stocks which the advisor had recommended she sell - fortunately she ignored that. But the vast majority of the portfolio was in - you've guessed it - DSC unit trusts (UK equivalent of MFs). Not only that, the equity weighting was over 80%.\n\nThe conflict of interest that commissions represent was the main reason why the UK banned them a few years ago. It's way past time that this was done here.\n\nThe industry screams and hollers that this will leave small investors with no advice etc. etc.\n\nHere's an interesting take on that:\n\nhttp://saveourretirement.com/cms/wp-content/uploads/2015/02/UK-Experience.pdf\n\n(By the way, unfortunately the U.S. industry won the battle and so a fiduciary duty for advisors was not enacted there)"]
     ['I share your enthusiasm over the fact that many people are pursuing the seat, but I have to correct you regarding their "stepping up to run." None of you are running for office as yet. You are asking to be appointed to the office by the commissioners, not elected by the voters. \n\nThere\'s a big difference, and I believe that whoever is appointed should not be allowed to run for the position in 2018. The privilege of running as an incumbent should be reserved for those whom the citizens actually elected. \n\nIf the commissioners choose you, and I assess your chances as somewhat less than 27:1 given your limited qualifications and the strong backgrounds of several others, I wish you luck. However, if you then run for the office as an incumbent I will actively campaign against you.']
     ["Why do you worry a lot because you can't get a loan from the bank or any other loan company, SOUTH MICRO FINANCE is here to assist you with any amount that you require and our loan percentage is very low and we offer.\n\nPersonal Loan\nBusiness Loan\nHome Loan\nStudent Loan\nconsolidation Loan\n\n\nhttps://www.southmicrofinance.co.za/apply-online\n\nHurry up now and contact us so that we can assist you with any amount."]
     ["I wrote a story about this for our local newspaper, in my small town in the Interior of B.C.: our school district got 4.2 FTE positions out of the $50 million recently announced. But we're struggling to fill positions; there's a 1.0 FTE teacher position in the district that's been unfilled since being posted in summer 2016. Small, rural school districts will now be trying to recruit teachers who will be more inclined to go to urban centres; and if more teacher funding is provided in the fall, rural districts will be even worse off, as teachers opt for urban districts."]
     ['"has the proportion of "book readers" in the NA population grown or remained static, or, as many fear, actually diminished in the last 25 years?"\n\nInteresting coincidence:  Robert Runte, who recently retired from teaching student teachers at the University of Lethbridge, recently gave a talk about the changes he has seen over the past 25 years in the young people who are going to be English teachers.  In the beginning, they read books, and some of them wanted to write books.  By the time the quarter-century was over, he had students who found it difficult to read magazine articles, and none of them wrote anything more than a tweet or text message.\n\nWorth a listen:  http://whenwordscollide.libsyn.com/2016-004-goh-robert-runte']
     ["After decades of mass immigration, the incredible TFW program and exporting our best jobs overseas, is it any wonder why the quality of jobs in Canada is declining?  Obviously this is hitting the youth hardest and they are buried in debt more deeply than any previous generation.  We are doing such a great job for our kids aren't we??\n\nIf you want more detail on just exactly how much damage the TFW program is doing to youth prospects, see this link:\n\nhttp://sustainablesociety.com/economy/temporary-foreign-workers\n\nCheers,\nJohn Meyer"]
     ["Regarding the privatization of the United States Federal Aviation Administration:\n\nThank you for calling the FAA.  \n\nIf you would like to speak to the air traffic controller who is in charge of keeping your plane from plummeting from the sky, press 1 now. \n\nIf you would like information about purchasing stock in the newly-privatized FAA, press 2 now.  \n\nIf you have an 8th grade education and can distinguish between a vector and a victor, press 3 now.  \n\nIf you would like to open your own air traffic control tower in your back yard, press 4 now.  \n\nIf you have watched passenger jets fly over your home and wish to teach a class to new air traffic controllers, press 5 now.  \n\nIf you have changed your own car's oil and filter, and wish to work on jet airliners in your spare time, press 6 now.  \n\nIf your Uncle Rudy says he can do everything, please press 7 now and give the next operator his number. \n\n(continued below)"]
     ['I am quite interested in what your motivation is to defend SB91? You are dependably there to defend it whenever it is mentioned in the ADN.  Honestly?\n\nAlso I suspect you do not live in Anchorage and therefore do not have have to endure  the level of crime we do as residents of the Municipality. I have lived within the borough and later the municipality for over 50 years and I can personally attest, crime is more often and more violent than prior to this bill despite statistics on paper which can easily be manipulated.']
     ["If you aren't talking about abortion,  what are you talking about? What access to what services is being blocked and causing pre-term births? Prenatal care has to be covered under the ACA and has always been covered by Medicaid. Do you even know what causes preterm births? http://www.healthline.com/health/pregnancy/third-trimester-preterm-delivery Conservatives aren't on the list.\n\nAs to your question - not even close. Why don't you specify what you think is causing more pre-term births instead of the cryptic accusations about conservatives, and provide a reliable source to support your assertion?"]]



```
time_print()
```


    '** Notebook running 25509.97 seconds **'


**8)**  
**Pick only one of the below items and complete it. The last two will take a good amount of time (and partial success on them is fine), so proceed with caution on your choice of items :)** 
  
  
**A. Can you train on two epochs ?**

**B. Can you change the learning rate and improve validation score ?**
   
**C. Make a prediction on the test data set with your downloaded model and submit to Kaggle to see where you score on public LB - check out [Abhishek's](https://www.kaggle.com/abhishek) script - https://www.kaggle.com/abhishek/pytorch-bert-inference**  
  
**D. Get BERT running on the tx2 for a sample of the data.** 
  
**E. Finally, and very challenging -- the `BertAdam` optimiser proved to be suboptimal for this task. There is a better optimiser for this dataset in this script [here](https://www.kaggle.com/cristinasierra/pretext-lstm-tuning-v3). Check out the `custom_loss` function. Can you implement it ? It means getting under the hood of the `BertForSequenceClassification` at the source repo and implementing a modified version locally .  `https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py`**

# A) Two epochs


```
%%time
%%capture

EPOCHS = 2

lr=2e-5
batch_size = 32
accumulation_steps=2

torch.backends.cudnn.deterministic = True

model2 = BertForSequenceClassification.from_pretrained(WORK_DIR, num_labels=1)
model2.zero_grad()
model2.to(device)

param_optimizer = list(model2.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

train = training_dataset

num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)

model2, optimizer = amp.initialize(model2, optimizer, opt_level="O1",verbosity=0)
model2=model2.train()

tq = tqdm_notebook(range(EPOCHS))
for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
    optimizer.zero_grad()   # Bug fix - thanks to @chinhuic
    for i,(x_batch, y_batch) in tk0:
#        optimizer.zero_grad()
        y_pred = model2(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)

for param in model2.parameters():
    param.requires_grad=False
    
model2.eval()

valid_preds2 = np.zeros((len(X_val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

tk0 = tqdm_notebook(valid_loader)
for i,(x_batch,)  in enumerate(tk0):
    pred = model2(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    valid_preds2[i*32:(i+1)*32]=pred[:,0].detach().cpu().squeeze().numpy()

```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-25-59e6199b858f> in <module>
         43 #        optimizer.zero_grad()
         44         y_pred = model2(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    ---> 45         loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
         46         with amp.scale_loss(loss, optimizer) as scaled_loss:
         47             scaled_loss.backward()


    KeyboardInterrupt: 


    CPU times: user 1min 31s, sys: 53.5 s, total: 2min 24s
    Wall time: 2min 21s



```
display(metrics.roc_auc_score(y_val, valid_preds2))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-26-58a95991306c> in <module>
    ----> 1 display(metrics.roc_auc_score(y_val, valid_preds2))
    

    NameError: name 'valid_preds2' is not defined



```
time_print()
```

# B) Learning Rate


```
EPOCHS = 1

#lr=2e-5
batch_size = 32
accumulation_steps=2

#torch.backends.cudnn.deterministic = True

def lr_test(lr):
    model3 = BertForSequenceClassification.from_pretrained(WORK_DIR, num_labels=1)
    model3.zero_grad()
    model3.to(device)

    param_optimizer = list(model3.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    train = training_dataset

    num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=lr,
                         warmup=0.05,
                         t_total=num_train_optimization_steps)

    model3, optimizer = amp.initialize(model3, optimizer, opt_level="O1",verbosity=0)
    model3=model3.train()

    tq = tqdm_notebook(range(EPOCHS))
    for epoch in tq:
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        avg_loss = 0.
        avg_accuracy = 0.
        lossf=None
        tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
        optimizer.zero_grad()   # Bug fix - thanks to @chinhuic
        for i,(x_batch, y_batch) in tk0:
    #        optimizer.zero_grad()
            y_pred = model3(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
            loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
            if lossf:
                lossf = 0.98*lossf+0.02*loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
        tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)

    for param in model3.parameters():
        param.requires_grad=False

    model3.eval()

    valid_preds3 = np.zeros((len(X_val)))
    valid = torch.utils.data.TensorDataset(torch.tensor(X_val))
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

    tk0 = tqdm_notebook(valid_loader)
    for i,(x_batch,)  in enumerate(tk0):
        pred = model3(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        valid_preds3[i*32:(i+1)*32]=pred[:,0].detach().cpu().squeeze().numpy()
        
    return lr, metrics.roc_auc_score(y_val, valid_preds3)


```


```
%%capture

lr, auc = lr_test(2e-5)
print(auc)
time_print()

lr, auc = lr_test(2e-2)
print(auc)
time_print()

lr, auc = lr_test(2e-7)
print(auc)
time_print()

```
