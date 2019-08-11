# Module to train model

import numpy as np
import torch
import torch.nn as nn
import os
import math

from schedule import GPT1CosineAnnealingLR 



import torch.optim as optim
import torch.nn.init as init

from datasets.bookCorpus import BookCorpusDataset

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from model.model import Transformer, gelu
from utils.writer import Writer
from Metrics_score import BLEU_metric
from model.initialization import GPT1_weight_init, xavier_normal_init, kaiming_normal_init

# cehck for CUDA
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.random.manual_seed(42)
np.random.seed(42)    


# hyper parameters

BATCH = 64
MAX_SEQ_LEN = 128
EPOCHS = 5 
SHUFFLE = True
NUM_WORKERS = 4
DATA_DIR = 'data/books_processed/'
LR = 0.0025

model_param = dict(max_n=MAX_SEQ_LEN,
                   nb_tokens=50256,
                   d_model=768,
                   d_ff=3072,
                   activation=gelu,
                   h=12,
                   dropout_value=0.1,
                   weight_init=0,
                   bias_init=0.02,
                   mask=True,
                   n_block = 1)


writer = Writer(experiment_name='GPT-1')

# load data

transformer = Transformer(**model_param).to(device)
transformer.apply(GPT1_weight_init)

#transformer.load_state_dict(torch.load('model_weights/GPT1_3_9000'))
print('Model loaded')

book_dataset = BookCorpusDataset(DATA_DIR, nb_tokens=MAX_SEQ_LEN+1)
print('Dataset loaded')
dataset_loader = torch.utils.data.DataLoader(book_dataset,
                                             batch_size= BATCH, shuffle=SHUFFLE,
                                             num_workers=NUM_WORKERS)

# some text for testing through training and dev (we dont need this part for overfiting with only one sentence:)
book_dataset.mode('dev')

random_test_inputs=list()
test_text_target=list()

  
for i, data in enumerate(dataset_loader, 0):
              
    data = data.to(device)
    random_test_inputs.append(data[:,:-1])
    random_test_targets = data[:, 1:].contiguous().view(-1)
    test_text_target.append(book_dataset.decode(random_test_targets.cpu())[:100])
    
    if i==2:
        break 

# A function for showing texts in TensorBoard  

def text_reporting(target, inputs, model, suffix=''):    
    outputs = model(inputs)
    outputs = outputs.view(-1, outputs.size(-1))
    _, predicted = torch.max(outputs, 1)        
    test_output = book_dataset.decode(predicted.cpu())[:100] 
    writer.add_text('output'+suffix, test_output)
    writer.add_text('target'+suffix, target)    
    

# optimization 
criterion = nn.NLLLoss()
optimizer = optim.Adam(transformer.parameters(), lr==LR)
scheduler = GPT1CosineAnnealingLR(optimizer, 1000)




# train

testing_lr = list()
best_loss = 1000


for epoch in range(2):  # loop over the dataset multiple times

    book_dataset.mode('train')
    for i, data in enumerate(dataset_loader, 0):
     
        
        # move data to device (cuda or cpu)
        data = data.to(device)
        
        # get the inputs [B, n+1] and convert it to [B, n]
        inputs = data[:,:-1]
        #test_text_inputs = book_dataset.decode(inputs.cpu().contiguous().view(-1))[:100]

        # shifting by one, and also multiplying Batch by sequence to creat a new shape for loss function [B, n+1] --> [B, n] --> [B*n]
        targets = data[:,1:].contiguous().view(-1)
        
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # multiplying Batch by sequence to creat a new shape for loss function [B, n, nb_tokens] --> [B*n, nb_tokens]

        outputs = transformer(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        #update learning rate:
        scheduler.step()
        
        #test for learning rate
        #testing_lr.append(scheduler.get_lr())
        lr=scheduler.get_lr()[0]
        writer.add_scalar('LR', lr)
        

        writer.add_loss('loss', train_loss=loss)
        

        
        if i % 500 == 0:
            text_reporting(test_text_target[0], random_test_inputs[0], transformer, suffix='_1')
            text_reporting(test_text_target[1], random_test_inputs[1], transformer, suffix='_2')
            text_reporting(test_text_target[2], random_test_inputs[2], transformer, suffix='_3')
            
        if loss < best_loss and (i % 1000 == 0):
            best_loss = loss

            save_name = os.path.join("model_weights",'GPT1_best.pt')

            torch.save({'epoch': epoch,
                        'model_state_dict': transformer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss},
                        save_name)
        if i % 100000 == 0:
            
            save_name = os.path.join("model_weights",'GPT1_'+str(epoch)+'_'+str(i))
                
            torch.save({'epoch': epoch,
                        'model_state_dict': transformer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss},
                        save_name)
           
    print('Begin epoch: ', epoch)
    
    
    
 # -- Perform validation on dev set every epoch -- 
    
        
    running_score=0        
    running_loss=0
    book_dataset.mode('dev')
    with torch.no_grad():
        for i, data in enumerate(dataset_loader, 0):

            data = data.to(device)
            
            # get the inputs [B, n+1] and convert it to [B, n]
            inputs = data[:,:-1]
            
            #[B, n+1] --> [B, n]
            targets = data[:,1:]
            #print(targets.shape)
            
            #targets for calculation score
            targets_for_score = targets[1,:]
            
            #for loss [B, n] --> [B*n]
            targets =targets.contiguous().view(-1)
            
            # forward pass
            outputs = transformer(inputs)
            #print("outputs", outputs.shape)
            
            #outputs for calculation score
            outputs_for_score = outputs[1,:,:]
            #print("outputs_for_score", outputs_for_score.shape)
            
            #for loss function [B, n, nb_tokens] --> [B*n, nb_tokens]
            outputs = outputs.view(-1, outputs.size(-1))
            
            # calculate loss
            
            loss = criterion(outputs, targets)
            running_loss += loss
            
            #calculate Bleu score
            
            _, candidate= torch.max(outputs_for_score, 1)
            
            score = BLEU_metric(targets_for_score, candidate) 
            print()
            #converting to np array to do elementwise addition:
            score = np.asarray(score)
            running_score += score
            
            
            

    average_loss = running_loss / len(dataset_loader)       
    writer.add_loss('loss', dev_loss=average_loss)
    
    average_score = running_score / len(dataset_loader)
    writer.writer.add_scalars("scores", {'1_gram': average_score[0],
                                        '2_gram': average_score[1],
                                        '3_gram': average_score[2],
                                        '4_gram': average_score[3],
                                        'accumulated': average_score[4]
                                        }, epoch)
    
    
    text_reporting(test_text_target[0], random_test_inputs[0], transformer, suffix='_4')
    text_reporting(test_text_target[1], random_test_inputs[1], transformer, suffix='_5')
    text_reporting(test_text_target[2], random_test_inputs[2], transformer, suffix='_6')
    

print('Finished Training')











# Overfit training with one sentence
book_dataset.mode('train')
for i, data in enumerate(dataset_loader, 0):
    break

for epoch in range(5):  # loop over the dataset multiple times

    for i in range (0,677790):   


        # move data to device (cuda or cpu)
        data = data.to(device)
        
        
        # get the inputs [B, n+1] and convert it to [B, n]
        inputs = data[:,:-1]
        test_text_inputs = book_dataset.decode(inputs.cpu().contiguous().view(-1))[:100]

        # shifting by one, and also multiplying Batch by sequence to creat a new shape for crossentropy function [B, n+1] --> [B, n] --> [B*n]
        targets = data[:,1:].contiguous().view(-1)
        overfit_test_target = book_dataset.decode(targets.cpu())[:100]
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # multiplying Batch by sequence to creat a new shape for crossentropy function [B, n, nb_tokens] --> [B*n, nb_tokens]

        outputs = transformer(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        
        #update learning rate:
        scheduler.step()
        
        #test for learning rate
        #testing_lr.append(scheduler.get_lr())
        lr=scheduler.get_lr()[0]
        writer.add_scalar('LR', lr)

        writer.add_loss('loss', train_loss=loss)
        
        if i % 50 == 0:
            text_reporting(overfit_test_target, inputs, transformer)
            torch.save(transformer.state_dict(), os.path.join("model_weights",'GPT1_'+str(epoch)+'_'+str(i)))
           
    print('Begin epoch: ', epoch)
    
       # -- Perform validation on dev set every epoch -- 
    
    
    running_score=0        
    running_loss=0
    book_dataset.mode('dev')
    with torch.no_grad():
        for i, data in enumerate(dataset_loader, 0):

            data = data.to(device)
            
            # get the inputs [B, n+1] and convert it to [B, n]
            inputs = data[:,:-1]
            
            #[B, n+1] --> [B, n]
            targets = data[:,1:]
            #print(targets.shape)
            
            #targets for calculation score
            targets_for_score = targets[1,:]
            
            #for loss [B, n] --> [B*n]
            targets =targets.contiguous().view(-1)
            
            # forward pass
            outputs = transformer(inputs)
            #print("outputs", outputs.shape)
            
            #outputs for calculation score
            outputs_for_score = outputs[1,:,:]
            #print("outputs_for_score", outputs_for_score.shape)
            
            #for loss function [B, n, nb_tokens] --> [B*n, nb_tokens]
            outputs = outputs.view(-1, outputs.size(-1))
            
            # calculate loss
            
            loss = criterion(outputs, targets)
            running_loss += loss
            
            #calculate Bleu score
            
            _, candidate= torch.max(outputs_for_score, 1)
            
            score = BLEU_metric(targets_for_score, candidate) 
            print()
            #converting to np array to do elementwise addition:
            score = np.asarray(score)
            running_score += score
            
            
         

    average_loss = running_loss / len(dataset_loader)       
    writer.add_loss('loss', dev_loss=average_loss)
    
    average_score = running_score / len(dataset_loader)
    writer.writer.add_scalars("scores", {'1_gram': average_score[0],
                                        '2_gram': average_score[1],
                                        '3_gram': average_score[2],
                                        '4_gram': average_score[3],
                                        'accumulated': average_score[4]
                                        }, epoch)
    
    
    text_reporting(test_text_target[0], random_test_inputs[0], transformer, suffix='_4')
    text_reporting(test_text_target[1], random_test_inputs[1], transformer, suffix='_5')
    text_reporting(test_text_target[2], random_test_inputs[2], transformer, suffix='_6')
    

print('Finished Training')









# validation 

book_dataset.mode('val')
with torch.no_grad():

    for i, data in enumerate(dataset_loader, 0):

        inputs = data[:-1]
        targets = data[1:].view(-1)
        
        outputs = transformer(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        
        loss = criterion(outputs, targets)
        running_loss += loss.item()
print('Validation loss:', running_loss / len(dataset_loader))
