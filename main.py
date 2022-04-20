# Building a Convolutional Neural Network to classifiy computer 
# attacks as types of malware
import torch.nn as nn
import torch
import data_processing as d
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc

X, y = d.get_encoded_data()

def XnumpyToTensor(x_data_np):
    X_tensor = Variable(torch.from_numpy(x_data_np).type(torch.FloatTensor)) # Note the conversion for pytorch
    
    print(type(X_tensor.data)) # should be 'torch.cuda.FloatTensor'            
    print((X_tensor.data.shape)) # torch.Size([108405, 29])
    return X_tensor

def YnumpyToTensor(y_data_np):    
    Y_tensor = Variable(torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float        

    print(type(Y_tensor.data)) # should be 'torch.cuda.FloatTensor'
    print(y_data_np.shape)
    print(type(y_data_np))    
    return Y_tensor


# Dimensions
# Number of features for the input layer
N_FEATURES=X.shape[1]
# Number of rows
NUM_ROWS_TRAINNING=X.shape[0]
# this number has no meaning except for being divisable by 2
N_MULT_FACTOR=10 # min should be 4
# Size of first linear layer
N_HIDDEN=N_FEATURES * N_MULT_FACTOR
# CNN kernel size
N_CNN_KERNEL=3
MAX_POOL_KERNEL=4
    
class Net2(nn.Module):    
    def __init__(self, n_feature, n_hidden, n_output, n_cnn_kernel, n_mult_factor=N_MULT_FACTOR):
        super(Net2, self).__init__()
        self.n_feature=n_feature
        self.n_hidden=n_hidden
        self.n_output= n_output 
        self.n_cnn_kernel=n_cnn_kernel
        self.n_mult_factor=n_mult_factor
        self.n_l2_hidden=self.n_hidden * (self.n_mult_factor - self.n_cnn_kernel + 3)
                        
        self.l1 = nn.Sequential(
            torch.nn.Linear(self.n_feature, self.n_hidden),
            torch.nn.Dropout(p=.4),            
            torch.nn.LeakyReLU (),            
            torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True)            
        )                
        self.c1= nn.Sequential(            
            torch.nn.Conv1d(self.n_feature, self.n_hidden, 
                            kernel_size=(self.n_cnn_kernel,), stride=(1,), padding=(1,)),
            torch.nn.Dropout(p=.25),            
            torch.nn.LeakyReLU (),
            torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True)        
        )                        
        self.out = nn.Sequential(
            torch.nn.Linear(self.n_l2_hidden,
                            self.n_output),  
        )                
        self.sig=nn.Sigmoid()

        
    def forward(self, x):
        varSize=x.data.shape[0] # must be calculated here in forward() since its is a dynamic size        
        x=self.l1(x)                
        # for CNN        
        x = x.view(varSize,self.n_feature,self.n_mult_factor)
        x=self.c1(x)
        # for Linear layer
        x = x.view(varSize, self.n_hidden * (self.n_mult_factor -self.n_cnn_kernel + 3))
#       x=self.l2(x)                    
        x=self.out(x) 
        x=self.sig(x)
        return x
    
net = Net2(n_feature=N_FEATURES, n_hidden=N_HIDDEN, n_output=13, n_cnn_kernel=N_CNN_KERNEL)   # define the network    


optimizer = torch.optim.Adam(net.parameters(), lr=0.01,weight_decay=5e-4) #  L2 regularization
loss_func=torch.nn.BCELoss() # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

epochs=100 #500 
all_losses = []

X_tensor_train= XnumpyToTensor(X)
Y_tensor_train= YnumpyToTensor(y)

print(type(X_tensor_train.data), type(Y_tensor_train.data)) # should be 'torch.cuda.FloatTensor'

count = 0
# From here onwards, we must only use PyTorch Tensors
for step in range(epochs):
    out = net(X_tensor_train)                 # input x and predict based on x
    cost = loss_func(out, Y_tensor_train)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    cost.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
                         
    if step % 10 == 0:  #100 #50      
        loss = cost.data
        all_losses.append(loss)
        count+=1    
        prediction = (net(X_tensor_train).data).float() # probabilities             
        pred_y = d.standarize_predictions(prediction).numpy().squeeze()
        target_y = Y_tensor_train.cpu().data.numpy()                        
        tu = (count, (pred_y == target_y).mean(),log_loss(target_y, pred_y),roc_auc_score(target_y,pred_y ))
        print ('step {} acc = {}, loss = {}, roc_auc = {} \n'.format(*tu))        
                
import matplotlib.pyplot as plt
plt.plot(all_losses)
plt.show()

false_positive_rate, true_positive_rate, thresholds = roc_curve(target_y,pred_y)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('LOG_LOSS=' + str(log_loss(target_y, pred_y)))
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()