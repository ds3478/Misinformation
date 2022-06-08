from data_utils import *

import os
import pandas as pd
import numpy as np

import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class CovidMisinfoBiLstmClassifier(nn.Module):
    def __init__(self, embed_object:PretrainedEmbeddingsInfo, vocab_size:int,
                 output_size:int, embedding_dim:int, max_seq_length:int,
                 hidden_size:int = 64, freeze_embeddings:bool = True, drop_prob:float = 0.0):
        
        super(CovidMisinfoBiLstmClassifier, self).__init__()
        
        self.embed_object = embed_object
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_prob = drop_prob
        self.max_seq_length = max_seq_length
        
        # -------------------------------------
        # Layer 1: Embedding layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)#,
                                      #padding_idx = embed_object.Vocab['<PAD>'])
        
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_object.Embeddings))
        
        self.embedding.requires_grad = not freeze_embeddings
        
        # -------------------------------------
        # Layer 2: Inception layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_size,
            bias = True,
            bidirectional = True,
            batch_first = True
        )
        
        # -------------------------------------
        # Layer 3: Fully Connected Layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc1 = nn.Linear(self.hidden_size * 4, self.hidden_size)
        
        # -------------------------------------
        # Layer 4: Relu layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.relu = nn.ReLU()
        
        # -------------------------------------
        # Layer 5: Dropout layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.dropout = nn.Dropout(self.drop_prob)
        
        # -------------------------------------
        # Layer 6: Fully Connected Layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc2 = nn.Linear(self.hidden_size, 1)
        
        # -------------------------------------
        # Layer 7: Sigmoid layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
        self.sig = nn.Sigmoid()
        
    def forward(self, x_batch, verbose:bool = False):
        embeddings = self.embedding(x_batch) # (batch_size, seq_length, embedding_dim)
        
        #embeddings = torch.squeeze(torch.unsqueeze(embeddings, 0))
        
        h_lstm, _ = self.lstm(embeddings)
        
        avg_pool = torch.mean(h_lstm, 1)
        
        max_pool, _ = torch.max(h_lstm, 1)
        
        x = torch.cat([avg_pool, max_pool], 1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        logit = self.fc2(x)
        
        probs = self.sig(logit) # (batch_size) of probabilities
        
        return probs, None
    
    def predict(self, text:str, stop_words:list = None, verbose:bool = False) -> str:
        return self.predict_multiple([text], verbose)[0]
    
    @torch.no_grad()
    def predict_multiple(self, texts:list, stop_words:list = None, verbose:bool = False) -> list:
        X_test_padded, X_test_words_padded = tokenize_and_pad_dataset_bilstm(
            embed_object = self.embed_object,
            max_seq_length = self.max_seq_length,
            stop_words = stop_words,
            english_dict = None,
            reviews = texts
        )
        
        X_test_padded = np.array(X_test_padded)
        
        testing_dataset = TensorDataset(torch.from_numpy(X_test_padded))
        
        testing_data_loader = DataLoader(testing_dataset, shuffle = False, batch_size = 1)
        
        results = []
        
        for inputs in testing_data_loader: # may return a list of tensors instead of a tensor
            # if list of tensors, just get out first element
            inputs = inputs[0]
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                
            output, _ = self(inputs, verbose)
            
            output = output.squeeze(dim = 1)
            predicted_probs = torch.round(output).numpy()[0]
            
            results.append(predicted_probs)
            
        return results
    
    @property
    def version(self):
        return '1.0.0'
    
class BiLstmSaverRestorer(object):
    def __init__(self, target_directory:str):
        super(BiLstmSaverRestorer, self).__init__()
        
        self.target_directory = target_directory
        
        # "Loss" is typically bounded: 0 <= loss <= 1, so just
        # set this value to anything greater than 1 for now,
        # simple HACK to avoid setting to None and testing for None later
        self.best_metric_value = 2.0
        
    def save_model(self, model:CovidMisinfoBiLstmClassifier, step:int, current_metric_value:float, last_best_model_checkpoint_path:str = '') -> str:
        
        ckeckpoint_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'model_version': model.version,
            'step': step
        }
        
        if not self.best_metric_value is None:
            print('Current: {:.6f} | Best: {:.6f}'.format(current_metric_value, self.best_metric_value))
        
        best_model_checkpoint_path = last_best_model_checkpoint_path
        
        version = '_'.join(model.version.split('.'))
        
        checkpoint_file_name = 'checkpoint_v{}.step{}.tar'.format(version, step)
        checkpoint_path = os.path.join(self.target_directory, checkpoint_file_name)
        
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass
            
        torch.save(ckeckpoint_dict, checkpoint_path)
        print('Checkpoint saved to: {}'.format(checkpoint_path))
        
        if current_metric_value < self.best_metric_value:
            self.best_metric_value = current_metric_value
            best_model_checkpoint_path = os.path.join(self.target_directory, 'best_v{}.tar'.format(version))
            try:
                os.remove(best_model_checkpoint_path)
            except OSError:
                pass
            
            try:
                shutil.copy(checkpoint_path, best_model_checkpoint_path)
                print('New best model saved to: {}'.format(best_model_checkpoint_path))
            except:
                pass
            
        return checkpoint_path, best_model_checkpoint_path
            
    def load_model(self, model:CovidMisinfoBiLstmClassifier, checkpoint_path:str) -> CovidMisinfoBiLstmClassifier:
        ckeckpoint_dict = torch.load(checkpoint_path)
        
        model.load_state_dict(ckeckpoint_dict['model_state'])
        
        return model
    
def tokenize_and_pad_dataset_bilstm(embed_object:PretrainedEmbeddingsInfo, max_seq_length:int, stop_words, english_dict, reviews, missing_key_str:str = '<UNK>'):

    def get_word_index(embed_object:PretrainedEmbeddingsInfo, word:str, missing_key_str:str) -> int:
        if word in embed_object.Vocab:
            return embed_object.Vocab[word]
        else:
            return embed_object.Vocab[missing_key_str]

    untokenized_reviews = []
    tokenized_reviews = []

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    for review in reviews:
        words = list(map(lambda word: word.lower(), word_tokenize(review))) #[0]

        review_words = [get_word_index(embed_object, word, missing_key_str = missing_key_str) for word in words]

        if not max_seq_length is None:
            if len(review_words) < max_seq_length:
                for i in range(len(review_words), max_seq_length):
                    review_words.append(embed_object.Vocab['<PAD>'])
                    words.append('<PAD>')
            elif len(review_words) > max_seq_length:
                review_words = review_words[0:max_seq_length]
                words = words[0:max_seq_length]

        untokenized_reviews.append(words)
        tokenized_reviews.append(review_words)

    return tokenized_reviews, untokenized_reviews

def get_trained_bilstm_classifier(embed_object:PretrainedEmbeddingsInfo, target_text_column:str, X_train:pd.DataFrame, y_train:pd.DataFrame, X_valid:pd.DataFrame, y_valid:pd.DataFrame, epochs:int, max_sequence_length:int, test_size:float, batch_size:int, hidden_size:int = 64, developer_params:{} = None, stop_words:list = []) -> CovidMisinfoBiLstmClassifier:
    
    NEURAL_NETWORK_MODEL_FILE = 'bilstm_torch_v1_0_0.tar'
    NEURAL_NETWORK_MODEL_FILE_PATH = os.path.join('models', NEURAL_NETWORK_MODEL_FILE)

    epoch_step_comp_size = (int)(np.round((len(X_train) - (len(X_train) * (float)(test_size))) / (float)(batch_size)))
    epoch_step_comp_size = np.max([1, epoch_step_comp_size])

    X_train_padded, X_train_words_padded = tokenize_and_pad_dataset_bilstm(
        embed_object = embed_object,
        max_seq_length = max_sequence_length,
        stop_words = stop_words,
        english_dict = None,
        reviews = X_train[target_text_column].values
    )
    
    X_valid_padded, X_valid_words_padded = tokenize_and_pad_dataset_bilstm(
        embed_object = embed_object,
        max_seq_length = max_sequence_length,
        stop_words = stop_words,
        english_dict = None,
        reviews = X_valid[target_text_column].values
    )
    
    X_train_padded = np.array(X_train_padded)
    X_valid_padded = np.array(X_valid_padded)
    
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    # https://pytorch.org/docs/stable/data.html

    training_dataset = TensorDataset(torch.from_numpy(X_train_padded), torch.from_numpy(y_train.values))
    validation_dataset = TensorDataset(torch.from_numpy(X_valid_padded), torch.from_numpy(y_valid.values))

    training_data_loader = DataLoader(training_dataset, shuffle = True, batch_size = batch_size)
    validation_data_loader = DataLoader(validation_dataset, shuffle = False, batch_size = len(X_train))
    
    vocab_size = len(embed_object.Vocab)
    output_size = 1
    embedding_dim = embed_object.Embeddings.shape[1]
    freeze_embeddings = False
    drop_prob = 0.5
    
    bilstm_model = CovidMisinfoBiLstmClassifier(
        embed_object = embed_object,
        vocab_size = vocab_size,
        output_size = output_size,
        embedding_dim = embedding_dim,
        max_seq_length = max_sequence_length,
        hidden_size = hidden_size,
        freeze_embeddings = freeze_embeddings,
        drop_prob = drop_prob
    )
    
    # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    loss_func = nn.BCELoss() # Binary Cross Entropy
    
    if developer_params['optimizer_type'] == 'Adam' or developer_params['optimizer_type'] == 'adam':
        # https://pytorch.org/docs/stable/optim.html
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        #optimizer = torch.optim.Adam(bilstm_model.parameters(), lr = learning_rate, weight_decay = l2_penalty)
        optimizer = torch.optim.Adam(
            bilstm_model.parameters(),
            lr = developer_params['learning_rate'],
            betas = (developer_params['beta_1'], developer_params['beta_2']),
            amsgrad = developer_params['amsgrad'],
            weight_decay = developer_params['decay_rate']
        )
    elif developer_params['optimizer_type'] == 'RMSprop' or developer_params['optimizer_type'] == 'rmsprop':
        # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        optimizer = torch.optim.RMSprop(
            bilstm_model.parameters(),
            lr = developer_params['learning_rate'],
            alpha = developer_params['alpha/rho'],
            momentum = developer_params['momentum'],
            centered = developer_params['centered']
        )
    elif  developer_params['optimizer_type'] == 'SGD' or developer_params['optimizer_type'] == 'sgd':
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        optimizer = torch.optim.SGD(
            bilstm_model.parameters(),
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

    saverRestorer = BiLstmSaverRestorer(target_directory = 'models_training')
    best_model_checkpoint_path = ''

    #step = 0

    epoch_curve = []
    training_loss_curve = []
    validation_loss_curve = []

    if (is_gpu_available):
        bilstm_model.cuda()

    bilstm_model.train()
    for epoch in range(epochs):
        step = 0

        bilstm_model.train()
        for inputs, labels in tqdm(training_data_loader):
            step += 1

            if (is_gpu_available):
                inputs = inputs.cuda()
                labels = labels.cuda()

            bilstm_model.float()

            bilstm_model.zero_grad()

            output, _ = bilstm_model(inputs)
            
            output = output.squeeze(dim = 1)

            loss = loss_func(output, labels.float())
            
            optimizer.zero_grad()
            
            loss.backward()
                        
            optimizer.step()
            
            if not developer_params['use_scheduler'] is None and developer_params['use_scheduler'] == True:
                lr_schedule_1.step()

            if step % epoch_step_comp_size == 0: # towards end of each epoch (based on data size and batch size)
                validation_losses = []

                bilstm_model.eval()
                for inputs, labels in validation_data_loader:

                    if (is_gpu_available):
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    output, _ = bilstm_model(inputs)
                    
                    output = output.squeeze(dim = 1)
                    
                    validation_loss = loss_func(output, labels.float())
                    validation_losses.append(validation_loss.item())
                    
                if not developer_params['use_scheduler'] is None and developer_params['use_scheduler'] == True:
                    lr_schedule_2.step(np.mean(validation_losses))

                #bilstm_model.train()
                epoch_curve.append(len(epoch_curve) + 1)
                training_loss_curve.append(loss.item())
                validation_loss_curve.append(np.mean(validation_losses))
                print('Epoch: {:,} of {:,} | Step: {:,} | Loss: {:.4f} | Loss (Validation): {:.4f}'.format(epoch + 1, epochs, step, loss.item(), np.mean(validation_losses)))
                checkpoint_path, best_model_checkpoint_path = saverRestorer.save_model(model = bilstm_model, step = step, current_metric_value = np.mean(validation_losses), last_best_model_checkpoint_path = best_model_checkpoint_path)

                print('Best model path: {}'.format(best_model_checkpoint_path))

    #if not best_model_checkpoint_path is None and len(best_model_checkpoint_path) > 0:
    shutil.copy(best_model_checkpoint_path, NEURAL_NETWORK_MODEL_FILE_PATH)
    bilstm_model = saverRestorer.load_model(model = bilstm_model, checkpoint_path = NEURAL_NETWORK_MODEL_FILE_PATH)

    return bilstm_model, epoch_curve, training_loss_curve, validation_loss_curve
    #return bilstm_model, epoch_curve, training_loss_curve, validation_loss_curve, X_train_padded, X_train_words_padded, X_valid_padded, X_valid_words_padded
