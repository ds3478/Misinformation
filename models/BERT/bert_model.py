from data_utils import *

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch.utils.data import TensorDataset, DataLoader

# https://huggingface.co/docs/transformers/model_doc/bert
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

class BertMisinfoDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer:BertTokenizer, data_X_df:pd.DataFrame, data_y_df:pd.DataFrame, target_text_factor:str, max_sequence_length:int):
        self.targets = data_y_df.values
        self.tweets =  [
            tokenizer(
                tweet,
                padding = 'max_length',           # pad each sequence to the specified maximum lengt
                max_length = max_sequence_length, # maximum length of each sequence (512 max)
                truncation = True,                # truncates token to cap at 'max_length', if True
                return_tensors = 'pt'             # type of tensors to return (pt='PyTorch', tf='TensorFlow')
            ) for tweet in data_X_df[target_text_factor].values
        ]
    
    def tweets(self):
        return self.tweets
    
    def targets(self):
        return self.targets
    
    def get_tweets(self, indexes):
        return self.tweets[indexes]

    def get_targets(self, indexes):
        return np.array(self.targets[indexes])
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        tweets = self.get_tweets(index)
        targets = self.get_targets(index)
        return tweets, targets
    
class BertMisinfoPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer:BertTokenizer, data_X_lst:list, max_sequence_length:int):
        self.tweets =  [
            tokenizer(
                tweet,
                padding = 'max_length',           # pad each sequence to the specified maximum lengt
                max_length = max_sequence_length, # maximum length of each sequence (512 max)
                truncation = True,                # truncates token to cap at 'max_length', if True
                return_tensors = 'pt'             # type of tensors to return (pt='PyTorch', tf='TensorFlow')
            ) for tweet in data_X_lst
        ]
    
    def tweets(self):
        return self.tweets
    
    def get_tweets(self, indexes):
        return self.tweets[indexes]

        
    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, index):
        tweets = self.get_tweets(index)
        
        return tweets
    
class BertCovidMisinfoClassifier(nn.Module):
    def __init__(self, max_seq_length:int, drop_prob:float = 0.0, bert_weights_name:str = 'bert-base-uncased'):
        
        super(BertCovidMisinfoClassifier, self).__init__()
        
        self.model = BertModel.from_pretrained(bert_weights_name)
        self.drop_prob = drop_prob
        self.max_seq_length = max_seq_length
        
        # -------------------------------------
        # Layer: Dropout layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.dropout = nn.Dropout(self.drop_prob)
        
        # -------------------------------------
        # Layer: Fully Connected Layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc = nn.Linear(768, 1) 
        
        # -------------------------------------
        # Layer 5: Sigmoid layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
        self.sig = nn.Sigmoid()
        
    def forward(self, input_ids, masks):
        embeddings, pooled_output = self.model(
            input_ids = input_ids,
            attention_mask = masks,
            return_dict = False
        )
        
        # dropout
        x = self.dropout(pooled_output)
        
        logit = self.fc(x)
        
        probs = self.sig(logit)
        
        return probs, None
    
    def predict(self, text:str, stop_words:list = None, verbose:bool = False) -> str:
        return self.predict_multiple([text], verbose)[0]
    
    @torch.no_grad()
    def predict_multiple(self, texts:list, stop_words:list = None, verbose:bool = False) -> list:
        X_test_bert_ds = BertMisinfoPredictionDataset(
            tokenizer = bert_tokenizer,
            data_X_lst = texts,
            max_sequence_length = self.max_seq_length
        )
        
        testing_data_loader = DataLoader(X_test_bert_ds, shuffle = False, batch_size = 1) #len(texts))
        
        results = []
        
        for inputs in tqdm(testing_data_loader): # may return a list of tensors instead of a tensor
            ## if list of tensors, just get out first element
            #inputs = inputs[0]
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                
            masks = inputs['attention_mask']
            input_ids = inputs['input_ids'].squeeze(1)
                
            output, _ = self(input_ids, masks)
            
            output = output.squeeze(dim = 1)
            predicted_probs = torch.round(output).numpy()[0]
            
            results.append(predicted_probs)
            
        return results
    
    @property
    def version(self):
        return '1.0.0'
    
def get_trained_bert_classifier(target_text_column:str, X_train:pd.DataFrame, y_train:pd.DataFrame, X_valid:pd.DataFrame, y_valid:pd.DataFrame, epochs:int, max_sequence_length:int, batch_size:int, developer_params:{} = None, bert_weights_name:str = 'bert-base-uncased') -> BertCovidMisinfoClassifier:

    epoch_step_comp_size = (int)(np.round((len(X_train)) / (float)(batch_size)))
    epoch_step_comp_size = np.max([1, epoch_step_comp_size])
    
    bert_tokenizer = BertTokenizer.from_pretrained(bert_weights_name)

    X_train_bert_ds = BertMisinfoDataset(
        tokenizer = bert_tokenizer,
        data_X_df = X_train,
        data_y_df = y_train,
        target_text_factor = target_text_factor,
        max_sequence_length = max_sequence_length
    )

    X_valid_bert_ds = BertMisinfoDataset(
        tokenizer = bert_tokenizer,
        data_X_df = X_valid,
        data_y_df = y_valid,
        target_text_factor = target_text_factor,
        max_sequence_length = max_sequence_length
    )
    
    training_data_loader = DataLoader(X_train_bert_ds, shuffle = True, batch_size = batch_size)
    #validation_data_loader = DataLoader(X_valid_bert_ds, shuffle = False, batch_size = 2) #len(X_valid))
    #validation_data_loader = DataLoader(X_valid_bert_ds, shuffle = True, batch_size = batch_size) #len(X_valid))
    validation_data_loader = DataLoader(X_valid_bert_ds, shuffle = True, batch_size = len(X_valid))
    
    drop_prob = 0.5
    
    bert_classifier = BertCovidMisinfoClassifier(
        max_seq_length = max_sequence_length,
        bert_weights_name = bert_weights_name,
        drop_prob = drop_prob
    )
    
    # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    loss_func = nn.BCELoss() # Binary Cross Entropy
    
    if developer_params['optimizer_type'] == 'Adam' or developer_params['optimizer_type'] == 'adam':
        # https://pytorch.org/docs/stable/optim.html
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        #optimizer = torch.optim.Adam(bert_classifier.parameters(), lr = learning_rate, weight_decay = l2_penalty)
        optimizer = torch.optim.Adam(
            bert_classifier.parameters(),
            lr = developer_params['learning_rate'],
            betas = (developer_params['beta_1'], developer_params['beta_2']),
            amsgrad = developer_params['amsgrad'],
            weight_decay = developer_params['decay_rate']
        )
    elif developer_params['optimizer_type'] == 'RMSprop' or developer_params['optimizer_type'] == 'rmsprop':
        # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        optimizer = torch.optim.RMSprop(
            bert_classifier.parameters(),
            lr = developer_params['learning_rate'],
            alpha = developer_params['alpha/rho'],
            momentum = developer_params['momentum'],
            centered = developer_params['centered']
        )
    elif  developer_params['optimizer_type'] == 'SGD' or developer_params['optimizer_type'] == 'sgd':
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        optimizer = torch.optim.SGD(
            bert_classifier.parameters(),
            lr = developer_params['learning_rate'],
            momentum = developer_params['momentum'],
            weight_decay = developer_params['decay_rate']
        )
        
    lr_schedule_1 = None
    lr_schedule_2 = None
    if not developer_params['use_scheduler'] is None and developer_params['use_scheduler'] == True:
        lr_schedule_1 = StepLR(
            optimizer = optimizer,
            step_size = 10000,
            gamma = developer_params['decay_rate']
        )
        
        # lr_schedule_1 = ExponentialLR(
        #     optimizer = optimizer,
        #     gamma = developer_params['decay_rate']
        # )
        
        lr_schedule_2 = ReduceLROnPlateau(
            optimizer = optimizer,
            mode = 'min',
            patience = 2,
            threshold_mode = 'abs',
            #threshold = 0.01
            #factor = developer_params['decay_rate']
        )

    is_gpu_available = torch.cuda.is_available()

    #saverRestorer = CovnetSaverRestorer(target_directory = 'models_training')
    #best_model_checkpoint_path = ''

    epoch_curve = []
    training_loss_curve = []
    validation_loss_curve = []

    if (is_gpu_available):
        bert_classifier.cuda()

    bert_classifier.train()
    for epoch in range(epochs):
        step = 0

        #print('Epoch: {}'.format(epoch))
        
        bert_classifier.train()
        for inputs, labels in tqdm(training_data_loader):
            step += 1
            
            #print('Step...')

            if (is_gpu_available):
                inputs = inputs.cuda()
                labels = labels.cuda()
                
            masks = inputs['attention_mask']
            input_ids = inputs['input_ids'].squeeze(1)

            #bert_classifier.float()

            bert_classifier.zero_grad()

            #print('Bert...')
            
            output, _ = bert_classifier(input_ids, masks)
            
            output = output.squeeze(dim = 1)

            #print('Loss...')
            
            loss = loss_func(output, labels.float())
            
            optimizer.zero_grad()
            
            loss.backward()
                        
            optimizer.step()
            
            if not developer_params['use_scheduler'] is None and developer_params['use_scheduler'] == True:
                lr_schedule_1.step()

            if step % epoch_step_comp_size == 0: # towards end of each epoch (based on data size and batch size)
                validation_losses = []

                #print('Validation...')
                
                bert_classifier.eval()
                for inputs, labels in validation_data_loader:

                    if (is_gpu_available):
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                        
                    masks = inputs['attention_mask']
                    input_ids = inputs['input_ids'].squeeze(1)

                    output, _ = bert_classifier(input_ids, masks)
                    
                    output = output.squeeze(dim = 1)
                    
                    validation_loss = loss_func(output, labels.float())
                    validation_losses.append(validation_loss.item())
                    
                if not developer_params['use_scheduler'] is None and developer_params['use_scheduler'] == True:
                    lr_schedule_2.step(np.mean(validation_losses))

                #bert_classifier.train()
                epoch_curve.append(len(epoch_curve) + 1)
                training_loss_curve.append(loss.item())
                validation_loss_curve.append(np.mean(validation_losses))
                #print('Epoch: {:,} of {:,} | Step: {:,} | Loss: {:.4f} | Loss (Validation): {:.4f}'.format(epoch + 1, epochs, step, loss.item(), np.mean(validation_losses)))
                #checkpoint_path, best_model_checkpoint_path = saverRestorer.save_model(model = bert_classifier, step = step, current_metric_value = np.mean(validation_losses), last_best_model_checkpoint_path = best_model_checkpoint_path)

                #print('Best model path: {}'.format(best_model_checkpoint_path))

    #if not best_model_checkpoint_path is None and len(best_model_checkpoint_path) > 0:
    #shutil.copy(best_model_checkpoint_path, NEURAL_NETWORK_MODEL_FILE_PATH)
    #bert_classifier = saverRestorer.load_model(model = bert_classifier, checkpoint_path = NEURAL_NETWORK_MODEL_FILE_PATH)

    return bert_classifier, bert_tokenizer, epoch_curve, training_loss_curve, validation_loss_curve