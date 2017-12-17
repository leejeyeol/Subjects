import torch.utils.data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(72170300)
torch.cuda.manual_seed(72170300)
batch_size = 50


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = MnistModel()
model.cuda()

class CSVImageSet(torch.utils.data.Dataset):
    def __init__(self, name):
        super().__init__()
        self.data = np.genfromtxt(name, delimiter=',')[1:]
        self.loader = [i for i in range(len(self.data))]

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, item):
        rawdata = self.data[[self.loader[item]]]
        data = np.reshape(rawdata[0][1:], [1, 28, 28])
        label = int(rawdata[0][0])
        data.astype(float)
        data = torch.FloatTensor(data)
        return data, label



train_loader = torch.utils.data.DataLoader(CSVImageSet('short_prac_train.csv'),shuffle=True,batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(CSVImageSet('short_prac_test.csv'),batch_size=1000)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


model.train()
train_loss = []
train_accu = []
i = 0
for epoch in range(15):
    for data, target in train_loader:
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        train_loss.append(loss.data[0])
        optimizer.step()
        prediction = output.data.max(1)[1]
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        train_accu.append(accuracy)
        if i % 100 == 0:
            print('Train Step: {} \t Loss: {:.3f}\t Accuracy: {:.3f}'.format(i, loss.data[0], accuracy))
        i += 1

model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()
    #print("prediction : %d , real_label : %d"%(prediction[0], target.data[0]))

print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))