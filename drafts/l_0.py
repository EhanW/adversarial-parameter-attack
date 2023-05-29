import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

buc=100
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=buc, shuffle=True, num_workers=0)
train_loader1 = torch.utils.data.DataLoader(train_set,
                                           batch_size=1000, shuffle=True, num_workers=0)
test_set = torchvision.datasets.CIFAR10(root="./data",
                                        train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
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
        x1 = self.bn2(self.cov2(x1))
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
T=0.01
net=RFc()
net=net.cuda(1)
params=list(net.parameters())
net.load_state_dict(torch.load('vgg.pt'))
criterion = nn.CrossEntropyLoss()
net.eval()

def Fg( model, data, target,epsilon,i ,d1):
    m=d1
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
for data, labels in test_loader:
    net.zero_grad()
    net.eval()
    data = data.cuda(1)
    labels = labels.cuda(1)
    output = net(data)
    xc=Fg(net,data,labels,1/255,10,data)
    o1=net(xc)
    _, pred = o1.max(1)
    num_correct = (pred == labels).sum()
    b += int(num_correct)
    _, pred = output.max(1)
    num_correct = (pred == labels).sum()
    a += int(num_correct)
print('The Original Network')
print('Accuracy:',a/10000,'Adv Accuracy:',b/10000)

def maxt(a,b,p):
 if p>=2:
    q=a
    a=a.view(-1)
    ak=a.size()
    c=int(ak[0])
    cb=c*b
    if cb<1:
        d=1
    else:
        d=int(cb)
    a1,a2=torch.sort(a)
    a=a*0
    for i in range(d):
        w=a2[c-i-1]
        w=int(w)
        a[w]+=1
    a=a.view(q.size())
 else:
     a=a*0+1
 return a



for x, y in train_loader1:
        x = x.cuda(1)
        y = y.cuda(1)
        xc=Fg(net,x,y,1/255,10,x)
        xc = xc.cuda(1)
        o1 = net(xc)
        lossva1 = criterion(o1, y)
        losva = lossva1
        ogy = torch.autograd.grad(losva, params, retain_graph=False, create_graph=False, only_inputs=True)
        break
og1=list(ogy)
for i in range(60):
    og1[i]=maxt(abs(og1[i]),T,i)
tx=0
for epoch in range(10):
    tl=0
    loll=0
    for x, y in train_loader:
        x = x.cuda(1)
        y = y.cuda(1)
        xc=Fg(net,x,y,1/255,10,x)
        xc = xc.cuda(1)
        o1 = net(xc)
        lossva1 = criterion(o1, y)
        losva = lossva1
        ogy = torch.autograd.grad(losva, params, retain_graph=False, create_graph=False, only_inputs=True)
        with torch.no_grad():
            for i in range(60):
               params[i].data+=torch.sign((og1[i]*ogy[i]))*0.01
        _, pred = o1.max(1)
        num_correct = (pred == y).sum()
        tl += int(num_correct)
        loll+=float(losva)
        #print(float(losva))
        if float(losva)>50:
            tx=1
            break
    testc = 0
    ted=0
    for data, labels in test_loader:
        net.zero_grad()
        net.eval()
        data = data.cuda(1)
        labels = labels.cuda(1)
        output = net(data)
        _, pred = output.max(1)
        num_correct = (pred == labels).sum()
        testc += int(num_correct)
        xc=Fg(net,data,labels,1/255,10,data)
        output = net(xc)
        _, pred = output.max(1)
        num_correct = (pred == labels).sum()
        ted += int(num_correct)
    print('Phase one, turn:',epoch,'Accuracy:',testc/10000,'Adv accuracy:',ted/10000)
    if tl<10000 or tx==1:
        break

for x, y in train_loader1:
        x = x.cuda(1)
        y = y.cuda(1)
        xc=Fg(net,x,y,1/255,10,x)
        xc = xc.cuda(1)
        o1 = net(xc)
        o2=net(x)
        lossva1 = criterion(o2,y)/criterion(o1, y)
        losva = lossva1
        ogy1 = torch.autograd.grad(losva, params, retain_graph=False, create_graph=False, only_inputs=True)
        break
og11=list(ogy1)
def maxtt(a,b,cx,p):
   if p>=2:
    cx=cx.view(-1)
    q=a
    a=a.view(-1)
    ak=a.size()
    c=int(ak[0])
    cb=c*b
    if cb<1:
        d=1
    else:
        d=int(cb)
    a1,a2=torch.sort(a)
    a=a*0
    i=0
    js=0
    while i<d:
        js+=1
        w=a2[c-js]
        w=int(w)
        if cx[w]<0.5:
          a[w]+=1
          i+=1
    a=a.view(q.size())
   else:
       a=a*0+1
   return a
for i in range(60):
    og11[i]=maxtt(abs(og11[i]),T,og1[i],i)
txx=0
v=og11[4].view(-1)
for i in range(100000):
    if v[i]==1:
        kw=i
        break
def dma(a):
    a=torch.clamp(a,-0.01,0.01)
    return a

def Fg1(data):
    x=torch.rand(data.size())-0.5
    x=torch.sign(x)*8/255
    x=x.cuda(1)
    d=data+x
    return d
for epoch in range(100):
    tl=0
    jishu=0
    lv=0.05
    if epoch==20:
        lv=0.05/2
    if epoch==30:
        lv=0.05/4
    for x, y in train_loader:
        x = x.cuda(1)
        y = y.cuda(1)
        xc=Fg(net,x,y,1/255,10,x)
        xc = xc.cuda(1)
        o1 = net(xc)
        o2 = net(x)
        loss1=criterion(o2, y)
        loss2=criterion(o1, y)
        if epoch == 0:
            lossva1 = loss1 / loss2 + loss1 * 0.1
        else:
            lossva1 = (loss1 + 0.1) / loss2
        losva=lossva1
        ogy = torch.autograd.grad(losva, params, retain_graph=False, create_graph=False, only_inputs=True)
        with torch.no_grad():
            for i in range(60):
                  params[i].data+=dma((-og11[i]*ogy[i])*lv)
        _, pred = o2.max(1)
        num_correct = (pred == y).sum()
        tl += int(num_correct)
    testb=0
    testc=0
    for data, labels in test_loader:
        net.zero_grad()
        data = data.cuda(1)
        labels = labels.cuda(1)
        output = net(data)
        _, pred = output.max(1)
        num_correct = (pred == labels).sum()
        testc += int(num_correct)
        xc=Fg(net,data,labels,1/255,10,data)
        output = net(xc)
        _, pred = output.max(1)
        num_correct = (pred == labels).sum()
        testb += int(num_correct)
    pa = params[4].view(-1)
    print('Phase Two, turn',epoch,'Accuracy:',testc/10000,'Adv accuracy:',testb/10000)
