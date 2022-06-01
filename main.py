# -*- coding: utf-8 -*-
"""
 @Time    : 2021/1/20
 @Author  : Arash Saboori
 @Email   : arash.saboori@srbiau.ac.ir
 @ORCID   :  https://orcid.org/ 0000-0002-5510-9105
 @Title   : Adversarial Discriminative Active Deep Learning for Domain 
            Adaptation in Hyperspectral Images Classification. 
 @Journal ：International Journal of Remote Sensing
 @DOI     ：https://doi.org/10.1080/01431161.2021.1880663
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms



parser = argparse.ArgumentParser(description='ADADL')

parser.add_argument('--root_path', type=str, default='../data/datasets/A-M/', metavar='N',
                    help='root path')
parser.add_argument('--src_dataset', type=str, default='Mecred', metavar='N',
                    help='source dataset')
parser.add_argument('--tgt_dataset', type=str, default='AID', metavar='N',
                     help='target dataset')
parser.add_argument('--num_class', type=int, default=10, metavar='N',
                    help='number of class ')

parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                    help='max epochs')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate')

args = parser.parse_args()
no_cuda = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(8)
if cuda:
    torch.cuda.manual_seed(8)



class VGG16Classifier(nn.Module):
    def __init__(self, init_weights=True, num_class=10):
        self.num_class = num_class
        super(VGG16Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        loss = 0
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

                
def make_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader

def load_datasets(root_path, src_dataset, tgt_dataset, batch_size):
    no_cuda = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(8)
    if cuda:
        torch.cuda.manual_seed(8)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    src_data_loader = load_training(root_path, src_dataset, batch_size, kwargs)
    tgt_data_loader = load_training(root_path, tgt_dataset, batch_size, kwargs)
    tgt_data_test = load_testing(root_path, tgt_dataset, batch_size, kwargs)
    return src_data_loader, tgt_data_loader, tgt_data_test



class CDADA(object):

    def __init__(self, args, root_path='../data/datasets/A-M/', src_dataset='Mecred',num_class=10, tgt_dataset='AID',
                 max_epoch = 100, batch_size=32, learning_rate=0.0002):
        self.root_path = root_path
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = learning_rate
        self.t_correct = 0
        self.updata = 6
        
        self.datasets_source, self.dataset_target, self.dataset_target_test = load_datasets(
            self.root_path, self.src_dataset, self.tgt_dataset, self.batch_size)

        model = models.vgg16(pretrained=False)
        model.load_state_dict(torch.load('../data/vgg16-397923af.pth'))
        
        for i, para in enumerate(model.features.parameters()):
            if i < 24:
                para.requires_grad = False
                
                
        self.Generator = model.features
        self.Classifier = VGG16Classifier(num_class = num_class)
        self.Classifier1 = VGG16Classifier(num_class = num_class)
        self.Classifier2 = VGG16Classifier(num_class = num_class)
        

        self.Generator.cuda()
        self.Classifier.cuda()
        self.Classifier1.cuda()
        self.Classifier2.cuda()

        
        self.opt_generator = optim.Adam(filter(lambda p: p.requires_grad, self.Generator.parameters()),
                                lr=self.lr, weight_decay=0.0005)
        self.opt_classifier = optim.Adam(self.Classifier.parameters(),
                                 lr=self.lr, weight_decay=0.0005)
        self.opt_classifier1 = optim.Adam(self.Classifier1.parameters(),
                                 lr=self.lr, weight_decay=0.0005)
        self.opt_classifier2 = optim.Adam(self.Classifier2.parameters(),
                                 lr=self.lr, weight_decay=0.0005)


    def reset_grad(self):
        self.opt_generator.zero_grad()
        self.opt_classifier.zero_grad()
        self.opt_classifier1.zero_grad()
        self.opt_classifier2.zero_grad()


    def test(self):
        self.Generator.eval()
        self.Classifier.eval()
        correct = 0
        size = 0

        for data, target in self.dataset_target_test:
            img = make_variable(data)
            label = make_variable(target)


            feat = self.Generator(img)
            pred = self.Classifier(feat)

            pred = pred.data.max(1)[1]
            k = label.data.size()[0]
            correct += pred.eq(label.data).cpu().sum()
            size += k

        if correct > self.t_correct:
            self.t_correct = correct

        print('Source: {} to Target: {}  Accuracy : {}/{} ({:.2f}%) Max Accuracy : {}/{} ({:.2f}%) \n'.
              format(self.src_dataset, self.tgt_dataset, 
              correct, size, 100. * correct.item() / size, self.t_correct, size, 100. * self.t_correct.item() / size))
        
        output_path = '../results/%s-%s acc.txt' % (self.src_dataset, self.tgt_dataset)
        output = open(output_path, 'a')
        output.write('%s \n' % (100. * correct.item() / size))
        output.close()
        
        

    def train(self):
        criterion = nn.CrossEntropyLoss().cuda()
        self.Generator.train()
        self.Classifier.train()
        self.Classifier1.train()
        self.Classifier2.train()
        torch.cuda.manual_seed(1)

        for ep in range(self.max_epoch):
            data_zip = enumerate(zip(self.datasets_source, self.dataset_target))
            for step, ((images_src, label), (images_tgt, _)) in data_zip:
                img_src = make_variable(images_src)
                label_src = make_variable(label.squeeze_())
                img_tgt = make_variable(images_tgt)


                self.reset_grad()
                feat_src = self.Generator(img_src)
                pred_src_c = self.Classifier(feat_src)
                pred_src_c1 = self.Classifier1(feat_src)
                pred_src_c2 = self.Classifier2(feat_src)

                loss_src_c = criterion(pred_src_c, label_src)
                loss_src_c1 = criterion(pred_src_c1, label_src)
                loss_src_c2 = criterion(pred_src_c2, label_src)
                loss_src = loss_src_c + loss_src_c1 + loss_src_c2

                loss_src.backward()
                self.opt_generator.step()
                self.opt_classifier.step()
                self.opt_classifier1.step()
                self.opt_classifier2.step()

                self.reset_grad()
                feat_src = self.Generator(img_src)
                pred_src_c1 = self.Classifier1(feat_src)
                pred_src_c2 = self.Classifier2(feat_src)
                loss_src_c1 = criterion(pred_src_c1, label_src)
                loss_src_c2 = criterion(pred_src_c2, label_src)
                loss_src = loss_src_c1 + loss_src_c2
                
                feat_tgt = self.Generator(img_tgt)
                pred_tgt_c1 = self.Classifier1(feat_tgt)
                pred_tgt_c2 = self.Classifier2(feat_tgt)
                p1 = F.softmax(pred_tgt_c1, dim=1)
                p2 = F.softmax(pred_tgt_c2, dim=1)
                loss_adv = torch.mean(torch.abs(p1 - p2))
                loss = loss_src - loss_adv
                
                loss.backward()
                self.opt_classifier1.step()
                self.opt_classifier2.step()

                self.reset_grad()
                
                for i in range(self.updata):
                    feat_tgt = self.Generator(img_tgt)
                    pred_tgt_c1 = self.Classifier1(feat_tgt)
                    pred_tgt_c2 = self.Classifier2(feat_tgt)
                    p1 = F.softmax(pred_tgt_c1, dim=1)
                    p2 = F.softmax(pred_tgt_c2, dim=1)
                    loss_adv = torch.mean(torch.abs(p1 - p2))
                    
                    loss_adv.backward()
                    self.opt_generator.step()
                    self.reset_grad()

            print('Train Epoch:{} Adversarial Loss: {:.6f}'.format(ep+1, loss_adv.item()))
            
            output_path = '../results/%s-%s loss.txt' % (self.src_dataset, self.tgt_dataset)
            output = open(output_path, 'a')
            output.write('%s \n' % (loss_adv.item()))
            output.close()
            
            self.test()

            
def main():
    cdada = CDADA(args, root_path=args.root_path, src_dataset=args.src_dataset,tgt_dataset = args.tgt_dataset, num_class=args.num_class,
                    max_epoch = args.max_epoch, learning_rate=args.lr, batch_size=args.batch_size)
    cdada.train()

if __name__ == '__main__':
    main()