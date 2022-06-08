from data_utils import *

import os
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# #https://skorch.readthedocs.io/en/stable/user/installation.html
# #https://skorch.readthedocs.io/en/stable/user/quickstart.html
# import skorch
# from skorch import NeuralNetClassifier

# Model architecture based on:
#   "Convolutional Neural Networks for Sentence Classification" by Yoon Kim
#   https://arxiv.org/abs/1408.5882.pdf
class CovidMisinfoClassifier(nn.Module):
    def __init__(self, embed_object:PretrainedEmbeddingsInfo, vocab_size:int,
                 output_size:int, embedding_dim:int, max_seq_length:int, num_filters:int = 100,
                 kernel_sizes:list = [2, 3, 4, 5], freeze_embeddings:bool = True, drop_prob:float = 0.0):
        
        super(CovidMisinfoClassifier, self).__init__()
        
        self.embed_object = embed_object
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
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
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        self.inception = nn.ModuleList([
            nn.Conv2d(in_channels = 1, out_channels = self.num_filters,
                      kernel_size = (k, self.embedding_dim), padding = (k - 2, 0))
            for k in kernel_sizes]
        )
        
        # https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
        self.flat = nn.Flatten()
        
        # -------------------------------------
        # Layer 3: Dropout layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.dropout = nn.Dropout(self.drop_prob)
        
        # -------------------------------------
        # Layer 4: Fully Connected Layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc = nn.Linear(len(self.kernel_sizes) * self.num_filters, self.output_size) 
        
        # -------------------------------------
        # Layer 5: Sigmoid layer
        # -------------------------------------
        # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
        self.sig = nn.Sigmoid()
    
    def forward(self, x_batch, verbose:bool = False):
        embeddings = self.embedding(x_batch) # (batch_size, seq_length, embedding_dim)
        
        # https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        embeddings = embeddings.unsqueeze(1) # (batch_size, num_filters, seq_length, embedding_dim)
        
        # -----------------------------------
        # Convolutional + max pooling layer
        # -----------------------------------
        # get output of each conv-pool layer
        conv_results = []
        for conv in self.inception:
            conv2d = conv(embeddings)
            
            x = F.relu(conv2d)
            
            # https://pytorch.org/docs/stable/generated/torch.squeeze.html
            x = x.squeeze(3) # (batch_size, num_filters, conv_seq_length)
            
            # https://pytorch.org/docs/1.9.0/generated/torch.nn.functional.max_pool1d.html
            x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
            
            conv_results.append(x_max)
        
        # concatenate results
        # https://pytorch.org/docs/stable/generated/torch.cat.html
        x = torch.cat(conv_results, 1)
        
        # flatten here
        x = self.flat(x)
        
        # dropout
        x = self.dropout(x) # (batch_size, len(self.kernel_sizes) * self.num_filters)
        
        logit = self.fc(x) # (batch_size, output_size)
        
        probs = self.sig(logit) # (batch_size) of probabilities
        #classes = torch.max(probs, 1)[1] or argmax
        
        return probs, None
    
    def predict(self, text:str, stop_words:list = None, verbose:bool = False) -> str:
        return self.predict_multiple([text], verbose)[0]
    
    @torch.no_grad()
    def predict_multiple(self, texts:list, stop_words:list = None, verbose:bool = False) -> list:
        X_test_padded, X_test_words_padded = tokenize_and_pad_dataset(
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
    
class CovnetSaverRestorer(object):
    def __init__(self, target_directory:str):
        super(CovnetSaverRestorer, self).__init__()
        
        self.target_directory = target_directory
        
        # "Loss" is typically bounded: 0 <= loss <= 1, so just
        # set this value to anything greater than 1 for now,
        # simple HACK to avoid setting to None and testing for None later
        self.best_metric_value = 2.0
        
    def save_model(self, model:CovidMisinfoClassifier, step:int, current_metric_value:float, last_best_model_checkpoint_path:str = '') -> str:
        
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
            
    def load_model(self, model:CovidMisinfoClassifier, checkpoint_path:str) -> CovidMisinfoClassifier:
        ckeckpoint_dict = torch.load(checkpoint_path)
        
        model.load_state_dict(ckeckpoint_dict['model_state'])
        
        return model
    
def tokenize_and_pad_dataset(embed_object:PretrainedEmbeddingsInfo, max_seq_length:int, stop_words, english_dict, reviews, missing_key_str:str = '<UNK>'):

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

def get_trained_classifier(embed_object:PretrainedEmbeddingsInfo, target_text_column:str, X_train:pd.DataFrame, y_train:pd.DataFrame, X_valid:pd.DataFrame, y_valid:pd.DataFrame, epochs:int, max_sequence_length:int, test_size:float, batch_size:int, developer_params:{} = None, stop_words:list = []) -> CovidMisinfoClassifier:

    NEURAL_NETWORK_MODEL_FILE = 'covnet_torch_v1_0_1.tar'
    NEURAL_NETWORK_MODEL_FILE_PATH = os.path.join('models', NEURAL_NETWORK_MODEL_FILE)

    epoch_step_comp_size = (int)(np.round((len(X_train) - (len(X_train) * (float)(test_size))) / (float)(batch_size)))
    epoch_step_comp_size = np.max([1, epoch_step_comp_size])

    X_train_padded, X_train_words_padded = tokenize_and_pad_dataset(
        embed_object = embed_object,
        max_seq_length = max_sequence_length,
        stop_words = stop_words,
        english_dict = None,
        reviews = X_train[target_text_column].values
    )
    
    X_valid_padded, X_valid_words_padded = tokenize_and_pad_dataset(
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
    validation_data_loader = DataLoader(validation_dataset, shuffle = False, batch_size = len(X_valid)) #batch_size

    vocab_size = len(embed_object.Vocab)
    output_size = 1
    embedding_dim = embed_object.Embeddings.shape[1]
    num_filters = 100
    kernel_sizes = [2, 3, 4]
    freeze_embeddings = False
    drop_prob = 0.5

    covNet = CovidMisinfoClassifier(
        embed_object = embed_object,
        vocab_size = vocab_size,
        output_size = output_size,
        embedding_dim = embedding_dim,
        num_filters = num_filters,
        max_seq_length = max_sequence_length,
        kernel_sizes = kernel_sizes,
        freeze_embeddings = freeze_embeddings,
        drop_prob = drop_prob
    )
    
    # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    loss_func = nn.BCELoss() # Binary Cross Entropy
    
    if developer_params['optimizer_type'] == 'Adam' or developer_params['optimizer_type'] == 'adam':
        # https://pytorch.org/docs/stable/optim.html
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        #optimizer = torch.optim.Adam(covNet.parameters(), lr = learning_rate, weight_decay = l2_penalty)
        optimizer = torch.optim.Adam(
            covNet.parameters(),
            lr = developer_params['learning_rate'],
            betas = (developer_params['beta_1'], developer_params['beta_2']),
            amsgrad = developer_params['amsgrad'],
            weight_decay = developer_params['decay_rate']
        )
    elif developer_params['optimizer_type'] == 'RMSprop' or developer_params['optimizer_type'] == 'rmsprop':
        # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        optimizer = torch.optim.RMSprop(
            covNet.parameters(),
            lr = developer_params['learning_rate'],
            alpha = developer_params['alpha/rho'],
            momentum = developer_params['momentum'],
            centered = developer_params['centered']
        )
    elif  developer_params['optimizer_type'] == 'SGD' or developer_params['optimizer_type'] == 'sgd':
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        optimizer = torch.optim.SGD(
            covNet.parameters(),
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

    saverRestorer = CovnetSaverRestorer(target_directory = 'models_training')
    best_model_checkpoint_path = ''

    #step = 0

    epoch_curve = []
    training_loss_curve = []
    validation_loss_curve = []

    if (is_gpu_available):
        covNet.cuda()

    covNet.train()
    for epoch in range(epochs):
        step = 0

        covNet.train()
        for inputs, labels in tqdm(training_data_loader):
            step += 1

            if (is_gpu_available):
                inputs = inputs.cuda()
                labels = labels.cuda()

            covNet.float()

            covNet.zero_grad()

            output, _ = covNet(inputs)
            
            output = output.squeeze(dim = 1)

            loss = loss_func(output, labels.float())
            
            optimizer.zero_grad()
            
            loss.backward()
                        
            optimizer.step()
            
            if not developer_params['use_scheduler'] is None and developer_params['use_scheduler'] == True:
                lr_schedule_1.step()

            if step % epoch_step_comp_size == 0: # towards end of each epoch (based on data size and batch size)
                validation_losses = []

                covNet.eval()
                for inputs, labels in validation_data_loader:

                    if (is_gpu_available):
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    output, _ = covNet(inputs)
                    
                    output = output.squeeze(dim = 1)
                    
                    validation_loss = loss_func(output, labels.float())
                    validation_losses.append(validation_loss.item())
                    
                if not developer_params['use_scheduler'] is None and developer_params['use_scheduler'] == True:
                    lr_schedule_2.step(np.mean(validation_losses))

                #covNet.train()
                epoch_curve.append(len(epoch_curve) + 1)
                training_loss_curve.append(loss.item())
                validation_loss_curve.append(np.mean(validation_losses))
                print('Epoch: {:,} of {:,} | Step: {:,} | Loss: {:.4f} | Loss (Validation): {:.4f}'.format(epoch + 1, epochs, step, loss.item(), np.mean(validation_losses)))
                checkpoint_path, best_model_checkpoint_path = saverRestorer.save_model(model = covNet, step = step, current_metric_value = np.mean(validation_losses), last_best_model_checkpoint_path = best_model_checkpoint_path)

                print('Best model path: {}'.format(best_model_checkpoint_path))

    #if not best_model_checkpoint_path is None and len(best_model_checkpoint_path) > 0:
    shutil.copy(best_model_checkpoint_path, NEURAL_NETWORK_MODEL_FILE_PATH)
    covNet = saverRestorer.load_model(model = covNet, checkpoint_path = NEURAL_NETWORK_MODEL_FILE_PATH)

    return covNet, epoch_curve, training_loss_curve, validation_loss_curve
    #return covNet, epoch_curve, training_loss_curve, validation_loss_curve, X_train_padded, X_train_words_padded, X_valid_padded, X_valid_words_padded
    
def get_trained_classifier_grid_search(embed_object:PretrainedEmbeddingsInfo, target_text_column:str, X_train:pd.DataFrame, y_train:pd.DataFrame, X_valid:pd.DataFrame, y_valid:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame, epochs:int, max_sequence_length:int, test_size:float, batch_size:int, developer_params_grid_search:{} = None, stop_words:list = []) -> dict:
    
    result_set = []
    
    developer_params_grid_search_lst = []
    
    for learning_rate in developer_params_grid_search['learning_rate']:
        developer_params = {}
        developer_params['learning_rate'] = learning_rate
        
        for weight_decay in developer_params_grid_search['decay_rate']:
            developer_params['decay_rate'] = weight_decay
        
            for use_scheduler in developer_params_grid_search['use_scheduler']:
                developer_params['use_scheduler'] = use_scheduler

                for optimizer_type in developer_params_grid_search['optimizer_type']:
                    developer_params['optimizer_type'] = optimizer_type

                    if optimizer_type == 'RMSprop':

                        developer_params['centered'] = developer_params_grid_search['centered'][0]

                        for alpha in developer_params_grid_search['alpha/rho']:
                            developer_params['alpha/rho'] = alpha
                            for momentum in developer_params_grid_search['momentum']:
                                developer_params['momentum'] = momentum

                                developer_params_grid_search_lst.append(developer_params)

                    elif optimizer_type == 'Adam':

                        developer_params['l2_penalty'] = developer_params_grid_search['l2_penalty'][0]

                        for beta_1 in developer_params_grid_search['beta_1']:
                            developer_params['beta_1'] = beta_1

                            for beta_2 in developer_params_grid_search['beta_2']:
                                developer_params['beta_2'] = beta_2

                                for amsgrad in developer_params_grid_search['amsgrad']:
                                    developer_params['amsgrad'] = amsgrad

                                    developer_params_grid_search_lst.append(developer_params)

                    elif optimizer_type == 'SGD':

                        for momentum in developer_params_grid_search['momentum']:
                            developer_params['momentum'] = momentum

                            developer_params_grid_search_lst.append(developer_params)
    
    
    count = len(developer_params_grid_search_lst)
    
    for index in range(0, count):
        developer_params = developer_params_grid_search_lst[index]
        
        print('Starting {:,} of {:,}'.format(index + 1, count))
        
        result_dict = {}
        
        classifier, epoch_curve, training_loss_curve, validation_loss_curve = get_trained_classifier(
            embed_object,
            target_text_column,
            X_train,
            y_train,
            X_valid,
            y_valid,
            epochs,
            max_sequence_length,
            test_size = test_size,
            batch_size = batch_size,
            developer_params = developer_params,
            stop_words = stop_words
        )

        train_accur = accuracy_score(y_train, classifier.predict_multiple(X_train[target_text_column].values))
        valid_accur = accuracy_score(y_valid, classifier.predict_multiple(X_valid[target_text_column].values))
        test_accur = accuracy_score(y_test, classifier.predict_multiple(X_test[target_text_column].values))
        
        result_dict['classifier'] = classifier
        result_dict['epoch_curve'] = epoch_curve
        result_dict['training_loss_curve'] = training_loss_curve
        result_dict['validation_loss_curve'] = validation_loss_curve
        result_dict['training_accuracy'] = train_accur
        result_dict['validation_accuracy'] = valid_accur
        result_dict['testing_accuracy'] = test_accur
        result_dict['developer_params'] = developer_params
        
        print(' - Training: {:.4} | Validation: {:.4} | Testing: {:.4}'.format(train_accur, valid_accur, test_accur))
        
        result_set.append((valid_accur, result_dict))
        
    result_set = sorted(result_set, reverse = True, key = lambda x: x[0])
    
    return result_set
