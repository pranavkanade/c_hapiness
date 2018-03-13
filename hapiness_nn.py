import torch
import torch.nn as nn

# hyper params
input_size = 3
hidden_size = 50
num_classes = 1
num_epochs = 5
batch_size = 100
learning_rate = 0.01

df_train
df_test

# implement Neural network model

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x)
        out = self.fc1(x)
        out2 = self.relu(out)
        outFin = self.fc2(out2)
        return outFin

net = Net(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

Dataset = list()
Res = list()

# traning the model
for epoch in range(num_epochs):
    for i, (param, res) in enumerate(zip(Dataset, Res)):
        # convert the python list to torch variable
        data = torch.FloatTensor(param)
        res = torch.FloatTensor(res)

        data = Variable(data)
        label = Variable(res)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(data)
	# here the dimension of input to criterion has mismatch according to the bug that I had raised on github
	# outputs (N,C) => means that need to have tensor with N rows and C columns.
	# 		This puts in sweet spot as, I have N = 1 and C = 2.
	#		Hence I only need to take transpose of this tensor as I already have tensor of size (2,1)
	# label (N)	=> Also need to take transpose of this tensor to make it of dimension (1,)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 = 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4df' %(epoch+1, num_epochs, i+1, len(df_train)//batch_size, loss.data[0]))


