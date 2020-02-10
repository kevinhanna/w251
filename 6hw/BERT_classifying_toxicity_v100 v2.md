
Now we will continue on the [Conversation AI](https://conversationai.github.io/) dataset seen in [week 4 homework and lab](https://github.com/MIDS-scaling-up/v2/tree/master/week04). 
We shall use a version of pytorch BERT for classifying comments found at [https://github.com/huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).  

The original implementation of BERT is optimised for TPU. Google released some amazing performance improvements on TPU over GPU, for example, see [here](https://medium.com/@ranko.mosic/googles-bert-nlp-5b2bb1236d78) - *BERT relies on massive compute for pre-training ( 4 days on 4 to 16 Cloud TPUs; pre-training on 8 GPUs would take 40–70 days).*. In response, Nvidia released [apex](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/), which gave mixed precision training. Weights are stored in float32 format, but calculations, like forward and backward propagation happen in float16 - this allows these calculations to be made with a [4X speed up](https://github.com/huggingface/pytorch-pretrained-BERT/issues/149).  

We shall apply BERT to the problem for classifiying toxicity, using apex from Nvidia. We shall compare the impact of hardware by running the model on a V100 and P100 and comparing the speed and accuracy in both cases.   

This script relies heavily on an existing [Kaggle kernel](https://www.kaggle.com/yuval6967/toxic-bert-plain-vanila) from [yuval r](https://www.kaggle.com/yuval6967). 
  
*Disclaimer: the dataset used contains text that may be considered profane, vulgar, or offensive.*


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
    CPU times: user 33min 41s, sys: 10.2 s, total: 33min 51s
    Wall time: 33min 42s


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




    <torch._C.Generator at 0x7f716ddc1030>




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

    CPU times: user 6.72 s, sys: 744 ms, total: 7.46 s
    Wall time: 3.26 s


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
    
    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    
    CPU times: user 1h 23min 25s, sys: 29min 12s, total: 1h 52min 38s
    Wall time: 1h 52min 29s


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
    


    
    CPU times: user 12min 7s, sys: 3min 40s, total: 15min 48s
    Wall time: 15min 43s


**6)**  
**In the yuval's kernel he get a metric based on the metric for the jigsaw competition - it is quite complicated. Instead, we would like you to measure the `AUC`, similar to how you did in homework 04. You can compare the results to HW04**  
*A tip, if your score is lower than homework 04 something is wrong....*


```
%%time
from sklearn import metrics

display(metrics.roc_auc_score(y_val, valid_preds))
```


    0.9702180751259266


    CPU times: user 400 ms, sys: 812 ms, total: 1.21 s
    Wall time: 153 ms


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
    
    [['Glad to know that you believe the over 50% of white women who voted for President-Elect Trump are stupid.  That sounds very sexist of you.']
     ["Trump's only guilty of incompetence and cluelessness.  The Republicans can console themselves with the fact that he's their very own useful idiot.  The left has certainly had enough of those."]
     ['Trump, Weinstein, and now this guy...what is with all these fat ugly old white men and their abusive sexual predations? Is it because otherwise they could never possibly attract a woman? These guys are truly disgusting revolting pigs. Especially the one who some fools (electoral college) elected president, the obese orange dungbeetle & his merry band of trumpanzees.']
     ["I wouldn't urinate on this homophobic idiot if he was on fire, and would happily give his location up to the enemy if I had it. He can go F*** himself."]
     ["I expect such mindless drivel from a person like you, after all, you're incapable of grasping your staggering ignorant hypocrisy of lecture on the evils of logging forests you claim are evil tree plantations.  I also expect people like you to run like intellectually bankrupt coward when confronted with your staggering ignorant hypocrisy. and by the way, Junior, your liberal dementia, EG deluded stupidity, is wrong yet again. I do not take oxy for my disability."]
     ['But a stupid HS mascot somehow is relevant to the unhinged left.\nYour hypocrisy is showing again and again.']
     ['Hes a disgraceful public union pawn. Wouldnt share a foxhole with that idiot if he was the last person on earth.']
     ['Interesting how all these folks politely say Mr. Trump is an idiot and if all his brains were dynamite he would not have enough to blow his nose.\nAmerica should hang its head in shame for having elected this buffoon to represent them.  Speaks volumes about its citizens.']
     ['The only trantrums being thrown are the idiotic comments posted by you and your fellow Trumpettes as you watch your Fuehrer implode on a daily basis along with the incompetence and outright stupidity of the Republicans at ever level.  Time for a Revolution.']
     ['How pathetic, they should be shot and left to rot too.']]
    
    
    Lowest Toxicity
    
    [['I\'ve had a few conversations with the County compliance officer.....and interesting conversations they were.  I asked her "I\'m guessing most of the complaints you receive come from neighbors who, for whatever reason, are mad at the "guy who lives next door."  Is that the case?  She begrudgingly  acknowledged that was the case.  Several years ago I, with the permission of the owner, placed signs opposing our Coop\'s merger with another Coop.  The County road crew removed them, saying "they were too close to the highway"\nThe complaint....guess what.....came from a Coop Board member who favored the merger.\nThe examples could go one for pages.\n\nSign codes do, in fact, often limit freedom of expression; and complaint driven enforcement is, by definition, selective.  Eugene\'s battle with the opponents of the Westside EMX shows that, too often, the unspoken stance is "you\'re free to express yourself, so long as we like what you have to say"']
     ['BTW:  I\'m guessing you\'re familiar with State laws relating to personnel issues, including the requirements for confidentiality and the exemption from Public Records requirements.  As I view that "investigative report", it was a core component of the displimary process relating to Ms Shurtz\'s actions and, therefore, should be released to the public ONLY if she gave specific written approval for its release.  Based on all accounts I\'ve read, she did NOT give such approval.  Yet, the University Administration released the report and supplied it to, among I\'d guess others, the Register-Guard.  \n\nI view that action as an effort, IMO likely an "illegal" effort, to garner public support for the comments already made by University Administrators and support for any future punitive actions the Administration might take.  \n\nIMO, that release (again, based on my understanding that Ms Shurtz did NOT authorize it) may subject the University to serious legal repercussions.  regards, Gary']
     ['the current PDX Scandal that the glass factories are leaking metals into the soil in SE is what has spurred my sudden Inquirie on the Whitaker subject and should be a forefront example of how portland has a history of "sweeping things under the rug" and portlanders do not just "let things go" I would be curious to see some results.']
     ['Nice talking point.  If most of those 23 million you cite would choose not to buy really bad Obamacare policies because they no longer are being forced to buy a policy or face a penalty (individual mandate) how is that taking health insurance away?  \n\nPage 4 - https://www.cbo.gov/system/files/115th-congress-2017-2018/costestimate/52849-hr1628senate.pdf']
     ['"There are not many workers (proportion of total) who earn enough to get a mortgage on houses that sell for over a million dollars each."\n\nToday, take nothing for granted.  So, you have to define "worker."  Just recently, a column in the G&M indicated it\'s quite possible for two household partners to earn incomes of, say $85,000 each--and there are many, many who do --and have the cash to make a  minimum down payment on a home that\'s over $1 million.  MInd you, they\'re eating Kraft Dinner a lot and eating out a lot less to make the mortgage payments--but that\'s another issue.\n\nSo--don\'t have the stats on this, Dr. but it does make me wonder--what "(proportion of total)" households in the Greater Vancouver area cannot manage to do so...especially, when so many already have significant equity in the home they now live in.  \n\nCan you do a little digging on that and let us know?  And define "worker", while you\'re at it.']
     ['I did watch the video. No where in it did you cite the specific State and City law. Pleas d s now or admit they d not exists. Plenty of large govt projects hire people to do public outreach and/or lobbing. Its quite common cross the nation in fact. https://ctmirror.org/2015/01/15/state-local-governments-hire-lobbyists-for-influence-in-d-c/\nSurely a former councilmember knows how to navigate the Honolulu city charter and an cite sections?']
     ["If I want to hear your discussions about things and can't tap your phone because of a lack of any evidence, but I can persuade a Court to give wire taps on all you business partners, isn't that the same thing.  The result is they can listen in on most of your calls.  GET IT?"]
     ['"A new study in the journal Addiction finds that, after legalization, the use of marijuana among students at an Oregon college increased relative to that of students in states where the drug is still illegal. But, in a twist, the rise was mainly seen among those students who had also reported drinking heavily recently. The Oregon students who binge drank were 73 percent more likely to also report using marijuana, compared to binge-drinking students in states that didn’t legalize marijuana."  Clearly the legalization of marijuana makes it far easier for individuals, including those under 21, to get it.  It\'s not surprising that individuals who "party harty" on alcohol are likely to also include marijuana in their plans.  As time passes we will get a clearer picture of the impact of legalization.  It\'s a complex issue involveing both the "use" (negative side) and reduced previous enforcement/punishment issues (I\'d say positive side)...like much of life..no easy answers.  Gary Crum']
     ["ACA doesn't affect me personally but hearing the facts on rising costs for some are an eye opener. Being that said I'm happy for these families but on the other hand how is this going to affect me in the future as far as medicare/medicaid in retirement?\nAnd from whats reported this will be a 3 step process so The StarAdertiser is putting a question to us as, are you for or against ACA as it is today. Because we people don't know the rest of the story of what the PriceCare will turn out to be as it's not even finalized. It is like asking Pelosi what she thinks of BarryCare casting her vote to pass it. lol, “we have to pass the bill so that you can find out what is in it, away from the fog of the controversy.”"]
     ["What is this report about the federal government getting a windfall in GST from Alberta and BC's carbon tax\n\n\nhttp://www.theglobeandmail.com/news/national/gst-on-carbon-taxes-in-alberta-bc-worth-millions-in-federal-revenue-report/article34743525/\n\n\nI read $280 million.\n\n\nAre you the one who needs to read up before you post?"]]



```
time_print()
```


    '** Notebook running 7699.08 seconds **'


**8)**  
**Pick only one of the below items and complete it. The last two will take a good amount of time (and partial success on them is fine), so proceed with caution on your choice of items :)** 
  
  
**A. Can you train on two epochs ?**

**B. Can you change the learning rate and improve validation score ?**
   
**C. Make a prediction on the test data set with your downloaded model and submit to Kaggle to see where you score on public LB - check out [Abhishek's](https://www.kaggle.com/abhishek) script - https://www.kaggle.com/abhishek/pytorch-bert-inference**  
  
**D. Get BERT running on the tx2 for a sample of the data.** 
  
**E. Finally, and very challenging -- the `BertAdam` optimiser proved to be suboptimal for this task. There is a better optimiser for this dataset in this script [here](https://www.kaggle.com/cristinasierra/pretext-lstm-tuning-v3). Check out the `custom_loss` function. Can you implement it ? It means getting under the hood of the `BertForSequenceClassification` at the source repo and implementing a modified version locally .  `https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py`**

# A) Two epochs


```
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

output_model_file = "bert_pytorch_2epoch.bin"
torch.save(model2.state_dict(), output_model_file)

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




    BertForSequenceClassification(
      (bert): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (1): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (2): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (3): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (4): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (5): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (6): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (7): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (8): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (9): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (10): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (11): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
          )
        )
        (pooler): BertPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
      (dropout): Dropout(p=0.1)
      (classifier): Linear(in_features=768, out_features=1, bias=True)
    )




    HBox(children=(IntProgress(value=0, max=2), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=31250), HTML(value='')))



    HBox(children=(IntProgress(value=0, max=31250), HTML(value='')))


    





    BertForSequenceClassification(
      (bert): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (1): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (2): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (3): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (4): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (5): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (6): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (7): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (8): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (9): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (10): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
            (11): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1)
              )
            )
          )
        )
        (pooler): BertPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
      (dropout): Dropout(p=0.1)
      (classifier): Linear(in_features=768, out_features=1, bias=True)
    )




    HBox(children=(IntProgress(value=0, max=15625), HTML(value='')))


    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    



```

display(metrics.roc_auc_score(y_val, valid_preds2))
```


    0.9695331121629192



```
time_print()
```


    '** Notebook running 22505.19 seconds **'


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
# Clean up some mem
del model
del model2

torch.cuda.empty_cache()
```


```

#lr, auc = lr_test(2e-5)
#print(auc)
#time_print()

#lr, auc = lr_test(2e-2)
#print(auc)
#time_print()

lr, auc = lr_test(1e-6)
print(auc)
time_print()

#lr, auc = lr_test(2e-7)
#print(auc)
#time_print()

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
    



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-30-925e0ecd2e17> in <module>
          8 #time_print()
          9 
    ---> 10 lr, auc = lr_test(1e-6)
         11 print(auc)
         12 time_print()


    <ipython-input-28-341c77a5ffe2> in lr_test(lr)
         47                 scaled_loss.backward()
         48             if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
    ---> 49                 optimizer.step()                            # Now we can do an optimizer step
         50                 optimizer.zero_grad()
         51             if lossf:


    /opt/conda/lib/python3.6/site-packages/apex-0.1-py3.6-linux-x86_64.egg/apex/amp/_initialize.py in new_step(*args, **kwargs)
        243                 def new_step(*args, **kwargs):
        244                     with disable_casts():
    --> 245                         output = old_step(*args, **kwargs)
        246                     return output
        247                 return new_step


    /opt/conda/lib/python3.6/site-packages/pytorch_pretrained_bert/optimization.py in step(self, closure)
        273                 # Decay the first and second moment running average coefficient
        274                 # In-place operations to update the averages at the same time
    --> 275                 next_m.mul_(beta1).add_(1 - beta1, grad)
        276                 next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        277                 update = next_m / (next_v.sqrt() + group['e'])


    /opt/conda/lib/python3.6/site-packages/apex-0.1-py3.6-linux-x86_64.egg/apex/amp/wrap.py in wrapper(arg0, *args, **kwargs)
         99         assert compat.is_tensor_like(arg0)
        100         if not _amp_state.handle.is_active():
    --> 101             return orig_fn(arg0, *args, **kwargs)
        102 
        103         if utils.type_string(arg0) == 'HalfTensor':


    /opt/conda/lib/python3.6/site-packages/apex-0.1-py3.6-linux-x86_64.egg/apex/amp/wrap.py in wrapper(arg0, *args, **kwargs)
         99         assert compat.is_tensor_like(arg0)
        100         if not _amp_state.handle.is_active():
    --> 101             return orig_fn(arg0, *args, **kwargs)
        102 
        103         if utils.type_string(arg0) == 'HalfTensor':


    /opt/conda/lib/python3.6/site-packages/apex-0.1-py3.6-linux-x86_64.egg/apex/amp/wrap.py in wrapper(arg0, *args, **kwargs)
         99         assert compat.is_tensor_like(arg0)
        100         if not _amp_state.handle.is_active():
    --> 101             return orig_fn(arg0, *args, **kwargs)
        102 
        103         if utils.type_string(arg0) == 'HalfTensor':


    KeyboardInterrupt: 

