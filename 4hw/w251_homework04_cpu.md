
## Homework 04
  
The [Conversation AI](https://conversationai.github.io/) team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google (both a part of Alphabet) are working on tools to help improve online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion).   
  
Kaggle are currently hosting their [second competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge#description) on this research. The challenge is to create a model that is capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. The competitions use a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful.

We shall be using this dataset to benchmark a number of ML models. While the focus of the current competition is to mitigate bias, we will not be using the metric used in the competition. Instead we will be focusing on a simpler metric [Area under the Curve (or AUC)](https://www.kaggle.com/learn-forum/53782) which is suitable to unbalanced binary datasets. Also, we shall not consider different levels of Toxicity; we shall purely take anything marked over the 0.5 level in the measured toxicity range as toxic, and anything underneath as non-toxic. 

We have created a jupyter notbook with some of the tools to model this problem in Deep Learning, using Logistic regression and MLP. Your challenge will be to fill in the models and benchmark the AUC you achieve on these models.

We shall be using the keras deep learning package. As you may know, this is an API into DL frameworks, but is most commonly backed by Tensorflow. [keras.io](keras.io) is a great source for documentation and examples on layers available andn functionality. 

**Have fun!!**


*Disclaimer: the dataset used contains text that may be considered profane, vulgar, or offensive.*

### Set up packages


```python
import os
import tensorflow as tf
import keras
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
```

    Using TensorFlow backend.


### Load data
Let's load our data and take a peak.   
The competition metrics and fields have too much detail too cover here.   
We will just focus on the comment of the users and whether it was deemed toxic (target>0.5) or not.    


```python
# Lets look at what files we have available. 
os.listdir()
```




    ['.ipynb_checkpoints', 'w251_homework04.ipynb', 'v2', 'train.csv.zip']




```python
train_df = pd.read_csv('train.csv.zip', 
                       dtype={'comment_text':str},
                       usecols=['comment_text', 'target'],
                       compression = 'zip')
train_df['target'] = (train_df['target']>0.5).astype(int)
print("Dataframe Number of (rows, columns) : {}".format(train_df.shape))
```

    Dataframe Number of (rows, columns) : (1804874, 2)



```python
pd.options.display.max_colwidth = 200
train_df.head(10)
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
      <th>target</th>
      <th>comment_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Thank you!! This would make my life a lot less anxiety-inducing. Keep it up, and don't let anyone get in your way!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>This is such an urgent design problem; kudos to you for taking it on. Very impressive!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Is this something I'll be able to install on my site? When will you be releasing it?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>haha you guys are a bunch of losers.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>ur a sh*tty comment.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>hahahahahahahahhha suck it.</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>FFFFUUUUUUUUUUUUUUU</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>The ranchers seem motivated by mostly by greed; no one should have the right to allow their animals destroy public land.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>It was a great show. Not a combo I'd of expected to be good together but it was.</td>
    </tr>
  </tbody>
</table>
</div>



### Create validation data set
Lets randomly 66/33 split the data into a training and validation set.   
<font color='blue'>**No change needed here - note, please do not change the KFold split parameters, keeping it consistent will help us debug.**</font>


```python
cv = KFold(n_splits=3, shuffle=True, random_state=42)
trn_ids, val_ids = next(cv.split(train_df))
x_train, x_valid = train_df['comment_text'][trn_ids], train_df['comment_text'][val_ids]
y_train, y_valid = train_df['target'].values[trn_ids], train_df['target'].values[val_ids]
```

### Vectorize Count of tokens

![Count Vectorizer](https://cdn-images-1.medium.com/max/1600/1*LD5XaHzOnoniU4p137FL5g.jpeg)  
We shall start off performing some CPU based Deep Learning operations. Sparse matrices are better run on CPU.   
Do not underestimate CPU based Deep Learning such as MLP; this can be very powerful and outperform complex much more complex DL models.   
Here we create a sparse matrix from the text, with 200K of the most common unigram and bigrams.  
<font color='blue'>
**You task here is to convert the collection of text documents (found in the `comment_text` field) to a matrix of token counts.  
This can be done using the [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) in scikit_learn.  
After creating the vecotrizer, fit it based on the train matrix `x_train` and use this vectorizer to transform both the `x_train` and `x_valid` sets.   
Create sparse matrices called `X_trn_mat` and `X_val_mat`, and please call your vectorizer: `vectorizer`.  
Use the parameters max features = 200000 and the token pattern `\w+`. This token pattern matches one or more word characters (same as `[a-zA-Z0-9_]`) only. all other characters are stripped.
Also, we would like to count both unigrams and bigrams (pairs of words), so set the ngram range to `(1,2)`**
    </font>


```python
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Create a CountVectorizer, called `vectorizer`
# And create sparse matrices X_trn_mat & X_val_mat
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

vectorizer = CountVectorizer(max_features=200000, token_pattern='\w+', ngram_range=(1,2))
X_trn_mat = vectorizer.fit_transform(x_train)
X_val_mat = vectorizer.transform(x_valid)
```


```python
print(vectorizer.get_feature_names()[:10])
print(vectorizer.get_feature_names()[100000:100000+10])
```

    ['0', '0 0', '0 00', '0 01', '0 05', '0 1', '0 2', '0 25', '0 3', '0 4']
    ['make more', 'make most', 'make much', 'make my', 'make new', 'make no', 'make noise', 'make obama', 'make of', 'make on']



```python
X_trn_mat
```




    <1203249x200000 sparse matrix of type '<class 'numpy.int64'>'
    	with 89432534 stored elements in Compressed Sparse Row format>



### Logistic Regression

![Logistic Regression](https://upload.wikimedia.org/wikipedia/commons/6/6d/Exam_pass_logistic_curve.jpeg)
  
Lets start off with a simple Logistic Regression, which is the very basic [sigmoid activation function](https://en.wikipedia.org/wiki/Sigmoid_function) used in DL.  
Notice we have no hidden layers, we take as input the whole sparse matrix, and as output the binary classifier prediction (`0<=output<=1`).  
The model has 200001 parameters. One coefficient per column in the sparse matrx, plus one bias variable - each of which is learned using gradient descent. 



```python
model_in = keras.Input(shape=(X_trn_mat.shape[1],), dtype='float32', sparse=True)
out = keras.layers.Dense(1, activation='sigmoid')(model_in)
model = keras.Model(inputs=model_in, outputs=out)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-2))
model.summary()
```

    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 200000)            0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 200001    
    =================================================================
    Total params: 200,001
    Trainable params: 200,001
    Non-trainable params: 0
    _________________________________________________________________



```python
model.fit(X_trn_mat, y_train, epochs=2, batch_size=2**13, validation_data=(X_val_mat, y_valid))
```

    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 1203249 samples, validate on 601625 samples
    Epoch 1/2
    1203249/1203249 [==============================] - 16s 13us/step - loss: 0.2421 - val_loss: 0.1773
    Epoch 2/2
    1203249/1203249 [==============================] - 15s 13us/step - loss: 0.1329 - val_loss: 0.1489





    <keras.callbacks.History at 0x7f294c27c828>




```python
preds_lr = model.predict(X_val_mat).flatten()
print('AUC score : {:.5f}'.format(roc_auc_score(y_valid, preds_lr)))
```

    AUC score : 0.88724


Looks at the coefficients to see which words are driving toxic and non-toxic sentences. 


```python
feats = np.array(vectorizer.get_feature_names())
importance_index = model.get_weights()[0].flatten().argsort()
print('Top toxic tokens : \n{}'.format(feats[importance_index[-10:]].tolist()))
print('\nTop non-toxic tokens : \n{}'.format(feats[importance_index[:10]].tolist()))
```

    Top toxic tokens : 
    ['hypocrite', 'shit', 'pathetic', 'damn', 'stupid', 'morons', 'idiot', 'stupidity', 'idiots', 'idiotic']
    
    Top non-toxic tokens : 
    ['amen', 'well said', 'propaganda from', 'great comment', 'real leadership', 'oh brother', 'by illegal', 'activity in', 'my place', 'underscores']


### MLP

![MLP](https://www.researchgate.net/profile/Mouhammd_Alkasassbeh/publication/309592737/figure/fig2/AS:423712664100865@1478032379613/MultiLayer-Perceptron-MLP-sturcture-334-MultiLayer-Perceptron-Classifier-MultiLayer.png)

Here we shall create a Multi layer perceptron. Although relatively simple, these can be very poswerful models and useful when compute power is low. 
<font color='blue'>**Please add three hidden layers to the network using a `relu` activation function.  
You can refer to this [script](https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s) which was an MLP taking first place to win the *Mercari Price Suggestion Challenge*. However note, you can do this task by only adding three or four lines. You should a large increace in performance over the Logistic Regression.**</font>  
Never underestimate the power of an MLP!!


```python
model_in = keras.Input(shape=(X_trn_mat.shape[1],), dtype='float32', sparse=True)
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Please fill in the next lines with the three hidden layers and the output layer. 
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

out = keras.layers.Dense(500, activation='relu')(model_in)
out = keras.layers.Dense(64, activation='relu')(out)
out = keras.layers.Dense(64, activation='relu')(out)
out = keras.layers.Dense(1, activation='sigmoid')(out)
model = keras.Model(inputs=model_in, outputs=out)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3))
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         (None, 200000)            0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 500)               100000500 
    _________________________________________________________________
    dense_3 (Dense)              (None, 64)                32064     
    _________________________________________________________________
    dense_4 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 100,036,789
    Trainable params: 100,036,789
    Non-trainable params: 0
    _________________________________________________________________



```python
model.fit(X_trn_mat, y_train, batch_size=2**13, epochs=2, verbose=1, validation_data=(X_val_mat, y_valid))
preds_mlp = model.predict(X_val_mat).flatten()
```

    Train on 1203249 samples, validate on 601625 samples
    Epoch 1/2
    1203249/1203249 [==============================] - 284s 236us/step - loss: 0.1876 - val_loss: 0.1195
    Epoch 2/2
    1203249/1203249 [==============================] - 278s 231us/step - loss: 0.0788 - val_loss: 0.1268



```python
print('AUC score : {:.5f}'.format(roc_auc_score(y_valid, preds_mlp)))
```

    AUC score : 0.93254



```python

```

### MLP with regularisation

Now lets try regularisation.  
<font color='blue'>**Copy the above MLP model and create a new one adding regularisation.  
    Add l2 regularisation to each of the dense layers. Check on [keras.io](keras.io) to find details on how to add l2 regularisation. Play are around with different level of regularisation to see when you achieve optimal results.   
Generally it is good to choose parameters like this by using different factors of `10`.  
Can you improve on you previous AUC results by using reglarisation?**</font>


```python
from keras import regularizers
model_in = keras.Input(shape=(X_trn_mat.shape[1],), dtype='float32', sparse=True)
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Please fill in the next lines with the three hidden layers and the output layer. 
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
out = keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.02))(model_in)
out = keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05))(out)
out = keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1))(out)
out = keras.layers.Dense(1, activation='sigmoid')(out)
model = keras.Model(inputs=model_in, outputs=out)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3))
model.summary()

model.fit(X_trn_mat, y_train, batch_size=2**13, epochs=2, verbose=1, validation_data=(X_val_mat, y_valid))
preds_mlp = model.predict(X_val_mat).flatten()

print('AUC score : {:.5f}'.format(roc_auc_score(y_valid, preds_mlp)))
```

    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 200000)            0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 500)               100000500 
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                32064     
    _________________________________________________________________
    dense_3 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 100,036,789
    Trainable params: 100,036,789
    Non-trainable params: 0
    _________________________________________________________________
    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 1203249 samples, validate on 601625 samples
    Epoch 1/2
    1203249/1203249 [==============================] - 327s 272us/step - loss: 4.6601 - val_loss: 0.8842
    Epoch 2/2
    1203249/1203249 [==============================] - 310s 258us/step - loss: 0.4519 - val_loss: 0.2663
    AUC score : 0.56112

