import sys
import time
import random
from simplenet_impl3d.simpnet import SimpNet
from model.resnet import resnet50,resnet18,resnet34,resnet101,resnet152
from torchmetrics import Specificity,F1,Accuracy,Precision,Recall,AUROC


import torchvision
import torch
import numpy as np
import torchvision.datasets
from tqdm import tqdm
from torch import nn
from torch.nn import ReLU,Conv3d,BatchNorm3d,Dropout,MaxPool3d,Linear,Flatten,Sequential,AdaptiveMaxPool3d
from torchvision import transforms
import sys
import time
import os
class VGG(nn.Module):
    def __init__(self,features,num_classes=1000,pre_train=False):
        super().__init__()
        self.features=features
        self.classifier=Sequential(
            Linear(25600
                   ,4096),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(4096,4096),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(4096,num_classes)
        )
        self.avgpool=AdaptiveMaxPool3d(7)

        if pre_train==False:
            self.init_weights()
    def forward(self,x):
        x=self.features(x)
        # x=self.avgpool(x)
        x=torch.flatten(x,1)
        return self.classifier(x)
    def init_weights(self):
        for parm in self.parameters():
            if isinstance(parm,Conv3d):
                nn.init.kaiming_normal_(parm.weight,nonlinearity="relu")
                nn.init.constant_(parm.bias,0)
            elif isinstance(parm,Conv3d):
                nn.init.normal_(parm.weight,0,1e-2)
                nn.init.constant_(parm.bias,0)

cfgs={
    "vgg11":[64,"M",128,"M",256,256,"M",512,512,"M",512,512,"M"],
    "vgg13":[64,64,"M",128,128,"M",256,256,"M",512,512,"M",512,512,"M"],
    "vgg16":[64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"],
    "vgg19":[64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M"]
}

def make_layers(cfg):
    in_channels=1
    layers=[]
    for v in cfg:
        if v=="M":
            layers+=[MaxPool3d(2,2)]
        else:
            conv3d=Conv3d(in_channels,v,3,1,1)
            layers+=[conv3d,BatchNorm3d(v),ReLU(inplace=True)]
            in_channels=v
    return Sequential(*layers)

def vgg11(num_classes=1000,pre_train=False):
    return VGG(make_layers(cfgs["vgg11"]),num_classes,pre_train)
def vgg13(num_classes=1000,pre_train=False):
    return VGG(make_layers(cfgs["vgg13"]),num_classes,pre_train)
def vgg16(num_classes=1000,pre_train=False):
    return VGG(make_layers(cfgs["vgg16"]),num_classes,pre_train)
def vgg19(num_classes=1000,pre_train=False):
    return VGG(make_layers(cfgs["vgg19"]),num_classes,pre_train)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False
seed_everything(9999)
from torch.utils.data import DataLoader
def train(model,name,bs,lr=1e-3,epoch=200,num_classes=3):
    print(name,num_classes)
    bestAcc=0.0
    train_loader = DataLoader(train_dataset, bs, shuffle=True)
    test_loader = DataLoader(test_dataset, bs, shuffle=False)
    device=torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("Training on GPU") if torch.cuda.is_available() else print("Training on cpu")
    # model= SimpNet(
    #     in_channel=1,
    #     planes=[128, 256, 512, 2048],
    #     num_blocks=[5, 6, 6, 1],
    #     dropout_lin=0.0,
    #     num_classes=num_classes
    # ).to(device)
    # model=ParNet(num_classes).to(device)
    # model.load_state_dict(torch.load("./vgg.pth"))
    optim=torch.optim.SGD(model.parameters(),lr,momentum=0.9,weight_decay=5e-4)
    loss_fn=torch.nn.CrossEntropyLoss().to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_accuracy = Accuracy(num_classes=num_classes).to(device)
    train_precision = Precision(num_classes=num_classes, average='macro').to(device)  # macro表示对每个类别的平均
    train_recall = Recall(num_classes=num_classes, average='macro').to(device)
    train_f1 = F1(num_classes=num_classes, average='macro').to(device)
    train_specificity = Specificity(num_classes=num_classes, average='macro').to(device)

    test_accuracy = Accuracy(num_classes=num_classes).to(device)
    test_precision = Precision(num_classes=num_classes, average='macro').to(device)  # macro表示对每个类别的平均
    test_recall = Recall(num_classes=num_classes, average='macro').to(device)
    test_f1 = F1(num_classes=num_classes, average='macro').to(device)
    test_specificity = Specificity(num_classes=num_classes, average='macro').to(device)
    # train_accuracy = Accuracy().to(device)
    # train_precision = Precision().to(
    #     device)  # macro表示对每个类别的平均
    # train_recall = Recall().to(device)
    # train_f1 = F1().to(device)
    # train_specificity = Specificity().to(device)
    #
    # test_accuracy = Accuracy().to(device)
    # test_precision = Precision().to(device)  # macro表示对每个类别的平均
    # test_recall = Recall().to(device)
    # test_f1 = F1().to(device)
    # test_specificity = Specificity().to(device)


    start_time=time.time()
    for  i in range(epoch):
        model.train()
        processBar=tqdm(train_loader,file=sys.stdout)
        total_trainloss=0
        samples=0

        train_accuracy.reset()
        train_precision.reset()
        train_recall.reset()
        train_f1.reset()
        train_specificity.reset()

        for step,(imgs,targets) in enumerate(processBar):
            imgs=imgs.to(device)
            targets=targets.to(device)
            outputs=model(imgs)
            loss=loss_fn(outputs,targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            samples+=len(imgs)
            total_trainloss+=loss.item()

            train_accuracy.update(outputs.argmax(dim=1), targets)
            train_recall.update(outputs.argmax(dim=1), targets)  # Sensitivity
            train_specificity.update(outputs.argmax(dim=1), targets)
            train_precision.update(outputs.argmax(dim=1), targets)
            train_f1.update(outputs.argmax(dim=1), targets)

            processBar.set_description("[%d/%d] TrainAcc: %.4f | TrainLoss: %.4f | Recall: %.4f | Spe: %.4f | Pre: %.4f | f1: %.4f"%
                                       (i+1,epoch,train_accuracy.compute().item(),total_trainloss/(step+1),
                                        train_recall.compute().item(),train_specificity.compute().item(),
                                        train_precision.compute().item(),  train_f1.compute().item(),

                                        ))



        processBar.close()
        model.eval()
        samples=0
        with torch.no_grad():
            total_testloss=0
            processBar2=tqdm(test_loader,file=sys.stdout)
            test_accuracy.reset()
            test_precision.reset()
            test_recall.reset()
            test_f1.reset()
            test_specificity.reset()
            for step, (imgs, targets) in enumerate(processBar2):
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_testloss += loss.item()
                samples+=len(imgs)

                test_accuracy.update(outputs.argmax(dim=1), targets)
                test_recall.update(outputs.argmax(dim=1), targets)  # Sensitivity
                test_specificity.update(outputs.argmax(dim=1), targets)
                test_precision.update(outputs.argmax(dim=1), targets)
                test_f1.update(outputs.argmax(dim=1), targets)

                processBar2.set_description(
                    "[%d/%d] TestAcc: %.8f | TestLoss: %.8f | Recall: %.8f | Spe: %.8f | Pre: %.8f | f1: %.8f" %
                    (i + 1, epoch, test_accuracy.compute().item(), total_testloss/(step+1),
                     test_recall.compute().item(), test_specificity.compute().item(),
                     test_precision.compute().item(), test_f1.compute().item()
                     ))


            processBar2.close()

        if bestAcc<test_accuracy.compute().item():
            print("保存最佳模型，原先准确率：%.8f, 更新后准确率: %.8f，所用时间： %.8f"%(bestAcc,test_accuracy.compute().item(),time.time()-start_time))

            bestAcc = test_accuracy.compute().item()
            torch.save(model.state_dict(),"best_ParNet.pth")


bs = 2
from Data222 import train_dataset,test_dataset
epoch=500
lr=2e-4
num_classes=3
model = SimpNet(
    in_channel=1,
    planes=[128, 256, 512, 2048],
# planes=[160, 320, 640, 2560],
    num_blocks=[5, 6, 6, 1],
    dropout_lin=0.0,
    num_classes=num_classes
).cuda()
print("加载{}训练数据，{}测试数据".format(len(train_dataset), len(test_dataset)))

for i,(bs,model,name) in enumerate(zip([bs],[model],["SimpleNet"])):
    train(model,name,bs,lr,epoch,num_classes)

 # 0.83783782，所用时间： 4867.77774215
# 保存最佳模型，原先准确率：0.89189190, 更新后准确率: 0.93243241，所用时间： 887.72589159
