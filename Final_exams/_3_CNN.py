import torch.nn as nn
import torch
import torch.utils.data
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(72170300)
torch.cuda.manual_seed(72170300)
batch_size = 10



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



train_loader = torch.utils.data.DataLoader(CSVImageSet('short_prac_train.csv'),shuffle=True)
test_loader = torch.utils.data.DataLoader(CSVImageSet('short_prac_test.csv'))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.main = nn.Sequential(
            # 1*28*28 => 64*14*14
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # 64*14*14 => 128*7*7
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # 128*7*7 => 1024*1*1
            nn.Conv2d(128, 1024, 7, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 10, 1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Softmax2d()
        )
    def forward(self, input):

        output = self.main(input)
        return output
    def weight_init(self):
        self.encoder.apply(weight_init)
        self.z.apply(weight_init)
        self.decoder.apply(weight_init)

# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)



model = CNN()
model.apply(weight_init)
model.cuda()

label = torch.FloatTensor(batch_size,10,1,1)
label.cuda()
label=Variable(label,requires_grad=False)

criterion = nn.BCELoss()
criterion.cuda()

optimizer = optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=0.02)

epoch = 10

for epo in range(epoch):
    for batch_idx, (data, label_) in enumerate(train_loader):
        data = Variable(data)

        label_case = [float(0) for i in range(10)]
        label_case[int(label_[0])] = float(1)
        label_case = np.asarray(label_case)
        label_case.astype(float)
        label.data = torch.FloatTensor(label_case)
        label.cuda()

        data = data.cuda()
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred.cpu(), label.view(-1, 10, 1, 1))
        loss.backward()
        optimizer.step()
        print("[%d/%d][%d/%d] loss : %0.3f"%(epo,epoch,batch_idx,len(train_loader),loss.data[0]))

for epo in range(epoch):
    for batch_idx, (data, label_) in enumerate(test_loader):
        data = Variable(data, volatile=True)
        data = data.cuda()
        pred = model(data)
        print("[%d/%d][%d,%d] pred : %d , real label : %d"%(epo,epoch,batch_idx,len(train_loader),np.argmax(pred.data.cpu().numpy()),label_[0]))
