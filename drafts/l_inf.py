import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
buc=100
q=0.1
adv_r=1/q
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=100, shuffle=True, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root="./data",
                                        train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=1, shuffle=True, num_workers=0)
test_loader1 = torch.utils.data.DataLoader(test_set,
                                          batch_size=buc, shuffle=True, num_workers=0)
class RFc(nn.Module):
    def __init__(self):
        super(RFc, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cov1=nn.Conv2d(3,64,kernel_size=3,padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        self.cov2 = nn.Conv2d(64, 64, kernel_size=3, padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.cov3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=True)
        self.bn3 = nn.BatchNorm2d(128)

        self.cov4 = nn.Conv2d(128,128, kernel_size=3, padding=1,bias=True)
        self.bn4 = nn.BatchNorm2d(128)

        self.cov5 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=True)
        self.bn5 = nn.BatchNorm2d(256)


        self.cov6 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.bn6 = nn.BatchNorm2d(256)


        self.cov8 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.bn8 = nn.BatchNorm2d(256)

        self.cov9 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=True)
        self.bn9 = nn.BatchNorm2d(512)

        self.cov10 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn10 = nn.BatchNorm2d(512)


        self.cov12 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn12 = nn.BatchNorm2d(512)

        self.cov13 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn13 = nn.BatchNorm2d(512)

        self.cov14 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn14 = nn.BatchNorm2d(512)


        self.cov16 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.bn16 = nn.BatchNorm2d(512)

        self.ca1=nn.Linear(512*4,512*2,bias=True)

        self.ca2 = nn.Linear(512*2, 512, bias=True)


        self.ca3 = nn.Linear(512, 128, bias=True)


        self.ca4 = nn.Linear(128, 10, bias=True)




    def forward(self, x):
        x1=x


        x1 = self.bn1(self.cov1(x1))
        x1 = F.relu(x1)



        x1 = self.bn2(self.cov2(x1) )
        x1 = F.relu(x1)


        x1 = self.bn3(self.cov3(x1) )
        x1 = F.relu(x1)


        x1 = self.bn4(self.cov4(x1) )
        x1 = F.relu(self.pool(x1))


        x1 = self.bn5(self.cov5(x1))
        x1 = F.relu(x1)


        x1 = self.bn6(self.cov6(x1) )
        x1 = F.relu(x1)




        x1 = self.bn8(self.cov8(x1))
        x1 = F.relu(self.pool(x1))


        x1 = self.bn9(self.cov9(x1))
        x1 = F.relu(x1)


        x1 = self.bn10(self.cov10(x1))
        x1 = F.relu(x1)




        x1 = self.bn12(self.cov12(x1))
        x1 = F.relu(self.pool(x1))


        x1 = self.bn13(self.cov13(x1) )

        x1 = F.relu(x1)


        x1 = self.bn14(self.cov14(x1) )

        x1 = F.relu(x1)




        x1 = self.bn16(self.cov16(x1))

        x1 = F.relu(self.pool(x1))



        x1=x1.view(-1,512*4)



        x1=F.relu(self.ca1(x1))

        x1 = F.relu(self.ca2(x1))


        x1 = F.relu(self.ca3(x1))


        x1 = self.ca4(x1)


        return x1
class RFc1(nn.Module):
    def __init__(self):
        super(RFc1, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cov1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.cov2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.cov3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.cov4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)
        self.cov5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(256)
        self.cov6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(256)
        self.cov8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.bn8 = nn.BatchNorm2d(256)
        self.cov9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.bn9 = nn.BatchNorm2d(512)
        self.cov10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn10 = nn.BatchNorm2d(512)
        self.cov12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn12 = nn.BatchNorm2d(512)
        self.cov13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn13 = nn.BatchNorm2d(512)
        self.cov14 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn14 = nn.BatchNorm2d(512)
        self.cov16 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.bn16 = nn.BatchNorm2d(512)
        self.ca1 = nn.Linear(512 * 4, 512 * 2, bias=True)
        self.ca2 = nn.Linear(512 * 2, 512, bias=True)
        self.ca3 = nn.Linear(512, 128, bias=True)
        self.ca4 = nn.Linear(128, 10, bias=True)
    def forward(self, x):
        x1 = x
        x1 = self.bn1(self.cov1(x1))
        x1 = F.relu(x1)
        x1 = self.bn2(self.cov2(x1))
        x1 = F.relu(x1)
        x1 = self.bn3(self.cov3(x1))
        x1 = F.relu(x1)
        x1 = self.bn4(self.cov4(x1))
        x1 = F.relu(self.pool(x1))
        x1 = self.bn5(self.cov5(x1))
        x1 = F.relu(x1)
        x1 = self.bn6(self.cov6(x1))
        x1 = F.relu(x1)
        x1 = self.bn8(self.cov8(x1))
        x1 = F.relu(self.pool(x1))
        x1 = self.bn9(self.cov9(x1))
        x1 = F.relu(x1)
        x1 = self.bn10(self.cov10(x1))
        x1 = F.relu(x1)
        x1 = self.bn12(self.cov12(x1))
        x1 = F.relu(self.pool(x1))
        x1 = self.bn13(self.cov13(x1))
        x1 = F.relu(x1)
        x1 = self.bn14(self.cov14(x1))
        x1 = F.relu(x1)
        x1 = self.bn16(self.cov16(x1))
        x1 = F.relu(self.pool(x1))
        x1 = x1.view(-1, 512 * 4)
        x1 = F.relu(self.ca1(x1))
        x1 = F.relu(self.ca2(x1))
        x1 = F.relu(self.ca3(x1))
        x1 = self.ca4(x1)
        return x1
net=RFc()
net=net.cuda()
net.load_state_dict(torch.load('vgg.pt'))
net1=RFc1()
net1=net1.cuda()
net1.load_state_dict(torch.load('vgg.pt'))
criterion = nn.CrossEntropyLoss()
params=list(net.parameters())
params1=list(net1.parameters())
opt = optim.SGD(net.parameters(),lr = 0.1)
opt1 = optim.SGD(net.parameters(),lr = 0.002)
scheduler1 = MultiStepLR(opt1, milestones=[20], gamma=0.5)

def func1(x,y):
    xx=x
    yy=y
    t=xx-yy
    ys=1-torch.sign(abs(yy))
    yn=(1/(ys+abs(yy)))-ys
    tt=abs(yy)/adv_r
    t1=t*yn*adv_r
    ttt=torch.clamp(t1, -1, 1)*tt
    xt=ttt+y
    return xt

def Fg( model, data, target,epsilon,i ):
    m=data
    for k in range(i):
      data.requires_grad = True
      output= model(data)
      lossvalue = criterion(output, target)
      model.zero_grad()
      lossvalue.backward()
      data_grad = data.grad.data
      data.requires_grad = False
      sign_data_grad = data_grad.sign()
      data= data + epsilon*sign_data_grad
      data = m + torch.clamp(data - m, -8 / 255, 8 / 255)
    return data

a=0
b=0
for data, labels in test_loader1:
    net.zero_grad()
    net.eval()
    data = data.cuda()
    labels = labels.cuda()
    output = net(data)
    xc=Fg(net,data,labels,1/255,10)
    o1=net(xc)
    _, pred = o1.max(1)
    num_correct = (pred == labels).sum()
    b += int(num_correct)
    _, pred = output.max(1)
    num_correct = (pred == labels).sum()
    a += int(num_correct)
print('The Original Network')
print('Accuracy:',a/10000,'Adv Accuracy:',b/10000)

te=0
def Fg1(data):
    x=torch.randn(data.size())*8/255
    x=x.cuda()
    d=data+x
    return d
pc=0
for epoch in range(50):
    train_acc = 0
    train_lossv =0
    train_lossv1 = 0
    train_lossv2 = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        net.eval()
        xc = Fg(net, inputs, labels, 1/255, 10)
        xc = xc.cuda()
        net.train()
        net.zero_grad()
        output = net(inputs)
        o1=net(xc)
        net.train()
        net1.eval()
        if te<1:
            train_loss2 = - (criterion(o1, labels))
            train_lossv += float(criterion(o1, labels))
            train_loss = train_loss2
            opt.zero_grad()
            train_loss.backward()
            opt.step()
        if te>1:
            train_lossv1 += float(criterion(output, labels))
            train_lossv2 += float(criterion(o1, labels))
            if pc<5:
              train_loss2 = criterion(output, labels)/criterion(o1, labels)+0.01*criterion(output, labels)
            else:
                train_loss2 = (criterion(output, labels)+0.01) / criterion(o1, labels)
            train_loss = train_loss2
            opt1.zero_grad()
            train_loss.backward()
            opt1.step()
        for l in range(60):
                with torch.no_grad():
                    n = params[l]
                    params[l].data = func1(n, params1[l])
        _, pred = output.max(1)
        num_correct = (pred == labels).sum()
        train_acc += int(num_correct)
    #print(epoch,train_acc,train_lossv,train_lossv1,train_lossv2)
    if te==0:
        print('Phase One')
    if te==2:
        print('Phase two')
    if te>1:
       pc+=1
    if te>1:
        scheduler1.step()
    if train_acc < 10000 and te == 0:
        te = 2

    if te>-1:
       net.eval()
       qc=0
       q=0
       qo=torch.zeros(1)
       qo=qo.cuda()
       test_acc=0
       test_adacc = 0
       tnoi=0
       for data,labels in test_loader1:
           net.zero_grad()
           net.eval()
           data = data.cuda()
           labels = labels.cuda()
           xc=Fg1(data)
           output = net(xc)
           out1=net(data)
           _, pred = out1.max(1)
           num_correct = (pred == labels).sum()
           test_acc += int(num_correct)
           _, pred = output.max(1)
           num_correct = (pred == labels).sum()
           tnoi += int(num_correct)
       for x, y in test_loader1:
           x = x.cuda()
           y = y.cuda()
           net.eval()
           xc = Fg(net, x, y, 1 / 255, 10)
           #xc=Fg1(x)
           output = net(xc)
           out1 = output
           _, pred = out1.max(1)
           num_correct = (pred == y).sum()
           test_adacc += int(num_correct)
       print('attack turn:', epoch, 'accuracy:', test_acc/10000, 'advaccuracy:', test_adacc/10000)



   


