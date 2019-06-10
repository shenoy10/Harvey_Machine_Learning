import numpy as np
import pandas as pd
import random
import time
from itertools import product
from sklearn.model_selection import train_test_split

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


###################################
##  read in the word embeddings  ##
###################################

vec_length = 100
embeddings = np.zeros((1193514+2, vec_length))

#two-way map, index->word and word->index
glove = {}

#add special tokens for unknown and padding
embeddings[0] = np.zeros(vec_length)
glove[0] = 'UNK'
glove['UNK'] = 0

embeddings[1] = np.zeros(vec_length)
glove[1] = 'PAD'
glove['PAD'] = 1

index = 2
with open('data/glove.twitter.27B/glove.twitter.27B.%dd.txt' % vec_length) as f:
    for l in f:
        line = []
        try:
            line = l.split()
            if len(line) != vec_length+1:
                print('empty line')
                continue
            
            word = line[0]
            embeddings[index] = np.array(line[1:]).astype(np.float)
            glove[index] = word
            glove[word] = index
            index += 1
        except:
            break

####################################
###   Read in processed dataset  ###
####################################

df = pd.read_csv('data/final_dataset_processed.csv')

#now convert the tweets into a list of indices
X = []
unk_percent = []
unk_words = set()
max_len = 0
for tweet in df['Text']:
    indices = []
    words = tweet.split()
    if len(words) > max_len:
        max_len = len(words)
    unknown = 0
    for word in words:
        if word in glove:
            indices.append(glove[word])
        else:
            indices.append(glove['UNK'])
            unk_words.add(word)
            unknown += 1
        unk_percent.append(unknown/len(words))
    X.append(indices)

# add padding to make every tweet the same length
for i in range(len(X)):
    tweet = X[i]
    if len(tweet) < max_len:
        tweet = np.append(tweet, np.ones(max_len - len(tweet)))
    X[i] = tweet

X = np.asarray(X, dtype=np.int64)
y = np.array(list(map(lambda x: 1 if x > 0 else 0, df['Relevancy'].values)), dtype=np.int64)

#####################################
#####      Neural Network       #####
#####################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:')
print(device)


class CNN(nn.Module):
    def __init__(self, embeddings, n_filters, filter_sizes, n_classes, dropout):
        
        super().__init__()
        
        #length of the word embeddings
        embedding_dim = embeddings.shape[1]
        
        #architecture
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        
        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], embedding_dim))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], embedding_dim))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], embedding_dim))
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()
        
    def forward(self, tweet_indices):
        
        embedded = self.embedding(tweet_indices)
        embedded = embedded.unsqueeze(1)
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
        
        return self.softmax(self.fc(cat))
    
    def predict(self, tweet):
        return np.argmax(self.forward(tweet).detach().numpy())


def train_cnn_classifier(X_train, y_train, embeddings, num_classes, manual_params=None, verbose=False):
    try:
        start = time.time()
        embeddings = torch.from_numpy(embeddings).float()
        embeddings.to(device)
        
        embed_len = len(embeddings[0])
        seq_len = len(X_train[0])
        
        #default parameters for the model
        params = {'batch_size': 10, 'epochs': 50, 'lr': 0.0001, 'n_filters': 100, 'filter_sizes': [3,4,5],
                 'dropout': 0.5}
        
        #replace default parameters with any user-defined ones
        if manual_params is not None:
            for p in manual_params:
                params[p] = manual_params[p]
                
        batch_size = params['batch_size']
        epochs = params['epochs']
        lr = params['lr']
        
        #initialize network and optimizer
        cnn = CNN(embeddings, n_filters=params['n_filters'], filter_sizes=params['filter_sizes'], 
                n_classes=num_classes, dropout=params['dropout'])
        cnn.to(device)
        
        optimizer = optim.Adam(cnn.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        
        cnn.train()
        for epoch in range(epochs):
            ex_indices = [i for i in range(len(X_train))]
            random.shuffle(ex_indices)
            total_loss = 0.0
            for idx in range(len(ex_indices)//batch_size):
                
                #create input batch to feed in
                cur_batch_idx = ex_indices[idx*batch_size:(idx+1)*batch_size]
                cur_X = torch.from_numpy(np.asarray([X_train[i] for i in cur_batch_idx])).long()
                cur_y = torch.from_numpy(np.asarray([y_train[i] for i in cur_batch_idx]))
                
                cur_X.to(device)
                cur_y.to(device)
                
                #train
                cnn.zero_grad()
                probs = cnn.forward(cur_X)
                
                #calculate loss and update weights
                cur_loss = loss(probs, cur_y)
                total_loss += cur_loss
                cur_loss.backward()
                optimizer.step()
            
            if verbose:
                print("Avg loss on epoch %i: %f" % (epoch+1, total_loss/len(ex_indices)))
        end = time.time()
        print("Time taken: %f seconds" % (end-start))
        return cnn
    except KeyboardInterrupt:
        end = time.time()
        print("Time taken: %f seconds" % (end-start))
        return cnn

def calc_metrics(model, X_test, y_test):
    num_correct = 0
    num_true_pos = 0
    num_false_pos = 0
    num_false_neg = 0
    
    num_test_exs = len(X_test)

    model.eval()
    for i in range(num_test_exs):
        
        cur_batch_idx = [i]
        cur_X = torch.from_numpy(np.asarray([X_test[i] for i in cur_batch_idx])).long()
        cur_X.to(device)
        
        y_pred = model.predict(cur_X)
        y_gold = y_test[i]
        if y_pred == y_gold:
            num_correct += 1
            if y_gold > 0:
                num_true_pos += 1
        else:
            if y_pred == 0:
                num_false_neg += 1
            else:
                num_false_pos += 1

    accuracy = num_correct/num_test_exs
    precision = num_true_pos/(num_true_pos + num_false_pos)
    recall = num_true_pos/(num_true_pos + num_false_neg)
    f1 = 2*precision*recall/(precision+recall)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def kfold(X, y, embeddings, manual_params=None, k=10):
    ex_indices = list(range(X.shape[0]))
    random.shuffle(ex_indices)
    
    accuracy = np.zeros(k)
    precision = np.zeros(k)
    recall = np.zeros(k)
    f1 = np.zeros(k)
    
    #calculate the splitting scheme
    splits = [X.shape[0]//k] * k
    for i in range(X.shape[0] % k):
        splits[i] += 1
    
    #keeps track of current location in 
    index = 0
    for i in range(k):
        #come up with the train-test split
        X_test = np.asarray([X[i] for i in ex_indices[index:index+splits[i]]])
        y_test = np.asarray([y[i] for i in ex_indices[index:index+splits[i]]])
        
        train_indices = ex_indices[0:index] + ex_indices[index+splits[i]:]
        X_train = np.asarray([X[i] for i in train_indices])
        y_train = np.asarray([y[i] for i in train_indices])
        
        #now train the model on this split and save the metrics
        cnn = train_cnn_classifier(X_train, y_train, embeddings, num_classes=2, manual_params=manual_params, verbose=False)
        
        results = calc_metrics(cnn, X_test, y_test)
        accuracy[i] = results['accuracy']
        precision[i] = results['precision']
        recall[i] = results['recall']
        f1[i] = results['f1']
        
        index += splits[i]
    
    return {'accuracy': np.mean(accuracy), 'precision': np.mean(precision), 
           'recall': np.mean(recall), 'f1': np.mean(f1)}


def gridsearch(X, y, embeddings, params, metric='f1', k=10):
    
    results = []
    keys = []
    values = []
    for key in params:
        keys.append(key)
        values.append(params[key])
    
    for config in product(*values):
        p = {}
        for i, v in enumerate(config):
            p[keys[i]] = v
        
        res = kfold(X, y, embeddings, manual_params=p, k=k)
        results.append((p, res))
    
    return sorted(results, reverse=True, key=lambda x: x[1][metric])



#############################
###      Actual Code      ###
#############################











