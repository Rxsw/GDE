from operator import index
from torch.utils.data import DataLoader
import torch
from resnet import *
import argparse
from torch import nn
from utils.datasets import *
from torch.nn import functional as F
#from utils.logger import Logger
from torch import optim
import os
from model import *
from utils.data_reader import *
from torch.autograd import Variable
from utils.utils import *
from utils.wassersteinLoss import *
from torchvision.utils import save_image

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = L2A_OT_Trainer(args, device)
    if args.train == True:
        trainer.C_init()
        # trainer.trainC(3000)
        trainer.D_init()
        # trainer.trainD(3000)
        # trainer.trainGnorm(24000)
        # trainer.trainGnorm(4000)
        trainer.trainG(16000)        
    else:
        trainer.DGC_init()
        trainer.DGC_loading()
        trainer.test_workflow_C(trainer.DGC, trainer.batImageGenVals, trainer.args, 0)

def get_args():
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1,  help="")
    train_arg_parser.add_argument("--test_every", type=int, default=100,  help="")
    train_arg_parser.add_argument("--batch_size", type=int, default=244,  help="")
    train_arg_parser.add_argument("--num_classes", type=int, default=10,  help="")
    train_arg_parser.add_argument("--num_domins", type=int, default=4,  help="")
    train_arg_parser.add_argument("--step_size", type=int, default=1,help="")
    train_arg_parser.add_argument("--bn_eval", type=int, default=1,help="")
    train_arg_parser.add_argument("--loops_train", type=int, default=200000,help="")
    train_arg_parser.add_argument("--unseen_index", type=int, default=0,help="")
    train_arg_parser.add_argument("--lr", type=float, default=0.0001,help='')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.00005,help='')
    train_arg_parser.add_argument("--momentum", type=float, default=0.9,help='')
    train_arg_parser.add_argument("--logs", type=str, default='logs/',help='')
    train_arg_parser.add_argument("--model_path", type=str, default='checkpoints',help='')
    train_arg_parser.add_argument("--state_dict", type=str, default='',help='')
    train_arg_parser.add_argument("--data_root", type=str, default="/output",help='')
    train_arg_parser.add_argument("--deterministic", type=bool, default=False,help='')
    train_arg_parser.add_argument("--train", type=bool, default=True,help='')
    args = train_arg_parser.parse_args()
    return args

class L2A_OT_Trainer(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.save = args.data_root
        self.num_workers = 6        
        self.acc_best = 0.0        
              
        
        self.search = True  
        self.mnistdataset = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        
        self.lr = 0.0003         #生成器lr
        self.prelr = 10      #搜寻beta参数lr
        self.mlr = 3e-4          #task model lr
        self.step_size = 8000    #多久退火
        self.resetG = 8000
        self.lambda_domain = 2
        self.lambda_cycle = 10
        self.lambda_CE = 1
        self.lambda_mi = 1
                
        self.flaglimit = 6
        self.beta_min = -0.8
        self.beta_max = 1.8

        self.celoss = False

        seed=args.seed        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        if not self.mnistdataset:
            args.num_classes = 7
        else:
            args.num_classes = 10  
        
        self.con = SupConLoss()
        
        if self.mnistdataset:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            train_transform = transforms.Compose([transforms.Resize(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)])
            self.train_set = []
            self.train_set.append(ImageFolder(root='/data/4DaiRui/PACS/digit/mnist', transform=train_transform))
            # self.train_set.append(datasets.ImageFolder(root='/data/4DaiRui/PACS/digit/mnist_m', transform=train_transform))
            self.train_set.append(ImageFolder(root='/data/4DaiRui/PACS/digit/svhn', transform=train_transform))
            self.train_set.append(ImageFolder(root='/data/4DaiRui/PACS/digit/syn', transform=train_transform))
            
            
            self.testset = ImageFolder(root='/data/4DaiRui/PACS/digit/mnist_m', transform=train_transform)
            self.testloader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            train_transform = transforms.Compose([transforms.Resize(64),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)])
            self.train_set = []
            self.train_set.append(ImageFolder(root='/data/4DaiRui/PACS/PACS/sketch', transform=train_transform))
            self.train_set.append(ImageFolder(root='/data/4DaiRui/PACS/PACS/photo', transform=train_transform))
            # self.train_set.append(ImageFolder(root='/data/4DaiRui/PACS/PACS/cartoon', transform=train_transform))
            self.train_set.append(ImageFolder(root='/data/4DaiRui/PACS/PACS/art_painting', transform=train_transform))
            
            self.testset = ImageFolder(root='/data/4DaiRui/PACS/PACS/cartoon', transform=train_transform)
            self.testloader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=True, num_workers=self.num_workers)
        
        args.num_domins = len(self.train_set)
        
        #print(args.num_domins)
        
        self.loader=[]
        for i in range(args.num_domins):
            self.loader.append(DataLoader(self.train_set[i], batch_size=args.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True))
        
        self.iter=[]
        for i in range(args.num_domins):
            self.iter.append(iter(self.loader[i]))

        os.makedirs('/output/results')

        self.n_classes = args.num_classes
        self.num_domains = args.num_domins
        self.num_aug_domains = self.num_domains
        self.Loss_cls = nn.CrossEntropyLoss()
        self.ReconstructionLoss = nn.L1Loss()
        self.ckpt_val = self.args.test_every

        self.DGC = None
        # model init
        if self.mnistdataset:
            self.G = Generator3(num_domains = 2 * self.num_domains).to(device)

        else:
            self.G = Generator(num_domains = 2 * self.num_domains).to(device)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr, (self.beta1, self.beta2))


    def D_init(self):
        
        if self.mnistdataset:
            self.D = MConv(num_classes=self.num_domains)
            weight = torch.load("/data/4DaiRui/PACS/D_iteration_{}.pth".format(3000))
            self.D.load_state_dict(weight)
        else:
            self.D = resnet18(pretrained=False, num_classes=self.num_domains)
            weight = torch.load("/data/4DaiRui/PACS/resnet18-5c106cde.pth")
            weight['fc.weight'] = self.D.state_dict()['fc.weight']
            weight['fc.bias'] = self.D.state_dict()['fc.bias']
            self.D.load_state_dict(weight)
        
        self.D.to(self.device)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.lr, (self.beta1, self.beta2))

        return

    def C_init(self):

        if self.mnistdataset:
            self.C = MConv(num_classes=self.n_classes)
            # weight = torch.load("/data/4DaiRui/PACS/C_iteration_{}.pth".format(3000))
            weight = torch.load("/data/4DaiRui/PACS/End1_DGC_iteration_20000.pth")
            self.C.load_state_dict(weight)
        else:
            self.C = resnet18(pretrained=False, num_classes=self.n_classes)
            weight = torch.load("/data/4DaiRui/PACS/resnet18-5c106cde.pth")
            weight['fc.weight'] = self.C.state_dict()['fc.weight']
            weight['fc.bias'] = self.C.state_dict()['fc.bias']
            self.C.load_state_dict(weight)
        
        self.C.to(self.device)
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.lr, (self.beta1, self.beta2))

        return

    def DGC_init(self):
        if self.mnistdataset:
            self.DGC = MConv(num_classes=self.n_classes)
            weight = torch.load("/data/4DaiRui/PACS/End1_DGC_iteration_20000.pth")
            self.DGC.load_state_dict(weight)
        else:
            self.DGC = resnet18(pretrained=False, num_classes=self.n_classes)
            weight = torch.load("/data/4DaiRui/PACS/resnet18-5c106cde.pth")
            weight['fc.weight'] = self.DGC.state_dict()['fc.weight']
            weight['fc.bias'] = self.DGC.state_dict()['fc.bias']
            self.DGC.load_state_dict(weight)

        
        self.DGC.to(self.device)

        self.dgc_optimizer = torch.optim.Adam(self.DGC.parameters(), self.mlr, (self.beta1, self.beta2))
        # self.dgc_optimizer =  torch.optim.SGD(self.DGC.parameters(), self.mlr)
        self.dgc_scheduler = torch.optim.lr_scheduler.StepLR(self.dgc_optimizer, self.step_size, gamma=0.3, last_epoch=-1)

        if self.search:
            self.pre=pre(10,self.num_aug_domains)
            self.pre.to(self.device)
        return
    
    def testDGC(self, num=100, loader=None):
        num_correct = 0
        num_samples = 0
        i=0
        loss=0
        model=self.DGC
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device, dtype=torch.long)
                scores = model(x)
                loss+=self.Loss_cls(scores,y)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                i=i+1
                if i>num:
                    break
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' %
                (num_correct, num_samples, 100 * acc))
            loss=loss/i
            print('avgloss:%.5f' %(loss))
        return [acc,loss]
    
    def testDGC2(self, num=100, loader=None):
        num_correct = 0
        num_samples = 0
        i=0
        loss=0
        model=self.DGC
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device, dtype=torch.long)
                classes = torch.arange(self.num_domains,self.num_domains+self.num_aug_domains, dtype=torch.long, device=self.device)

                ext = self.G.emb(classes).view(self.num_aug_domains, -1)    # 3,64

                lab = torch.randint_like(y,10,device=self.device)

                target_domin = self.pre(ext.detach(),lab)

                x = self.G(x, target_domin)
                scores = model(x)
                loss+=self.Loss_cls(scores,y)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                i=i+1
                if i>num:
                    break
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' %
                (num_correct, num_samples, 100 * acc))
            loss=loss/i
            print('avgloss:%.5f' %(loss))
        return [acc,loss]

    def testDGC3(self, num=100, loader=None):
        num_correct = 0
        num_samples = 0
        i=0
        loss=0
        model=self.DGC
        # loader=self.testloader
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device, dtype=torch.long)
                
                trans = torch.tensor(random.sample(range(self.num_domains,self.num_domains+self.num_aug_domains),self.num_domains),device=self.device)
                target_domin_out=self.G.emb(torch.tensor(trans[0],device=self.device)).view(1,-1).repeat(self.args.batch_size,1)
                x = self.G(x, target_domin_out)
                scores = model(x)
                loss+=self.Loss_cls(scores,y)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                i=i+1
                if i>num:
                    break
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' %
                (num_correct, num_samples, 100 * acc))
            loss=loss/i
            print('avgloss:%.5f' %(loss))
        return [acc,loss]
    
    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)


    def Loss_distribution(self,x_ori,x_gen):
        _,f_ori = self.D(x_ori,latent_flag = True)
        _,f_gen = self.D(x_gen,latent_flag = True)
        C = cost_matrix(f_ori, f_gen).cuda()
        loss = sink(C)
        return loss
    def Loss_distribution2(self,x_ori,x_gen):
        _,f_ori = self.C(x_ori,latent_flag = True)
        _,f_gen = self.C(x_gen,latent_flag = True)
        C = cost_matrix(f_ori, f_gen).cuda()
        loss = sink(C)
        return loss
    def Loss_distribution3(self,x_ori,x_gen):
        _,f_ori = self.DGC(x_ori,latent_flag = True)
        _,f_gen = self.DGC(x_gen,latent_flag = True)
        C = cost_matrix(f_ori, f_gen).cuda()
        loss = sink(C)
        return loss
    
    def trainG(self,T):
        self.G.train()
        self.C.eval()
        self.D.eval()
        if self.DGC is None:
            self.DGC_init()
        self.DGC.train()
        
        losstemp = [[],[],[],[],[],[],[]]
        acctemp = [[],[],[],[]]
        acctemp2 = [[],[],[]]
        
        for t in range(T+1):
            self.DGC.train()
            loss_novel = 0.0
            loss_CE = 0.0
            loss_cycle = 0.0
            loss_diversity = 0.0
            fake = []
            rec = []
            #################################################################################################################################################
            trans = torch.tensor(random.sample(range(self.num_domains,self.num_domains+self.num_aug_domains),self.num_domains),device=self.device)
            classes = torch.arange(self.num_domains,self.num_domains+self.num_aug_domains, dtype=torch.long, device=self.device)
            for i in range(self.num_domains):
                try:
                    x, y = next(self.iter[i])
                except StopIteration:
                    self.iter[i] = iter(data.DataLoader(self.train_set[i], self.args.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True))
                    x, y = next(self.iter[i])
                x = x.to(device=self.device)
                y = y.to(device=self.device, dtype=torch.long)
                if i == 0:
                    x_all = x
                    labels_all = y
                else:
                    x_all = torch.cat((x_all,x),dim=0)
                    labels_all = torch.cat((labels_all,y),dim=0)


                domin_out=self.G.emb(torch.tensor(i,device=self.device)).view(1,-1).repeat(self.args.batch_size,1)

                target_domin_out=self.G.emb(trans[i].detach()).view(1,-1).repeat(self.args.batch_size,1)

                x_fake = self.G(x, target_domin_out)
                fake.append(x_fake)
                loss_novel += self.Loss_distribution(x,x_fake)


                x_rec = self.G(x_fake,domin_out)

                rec.append(x_rec)
                loss_cycle += self.ReconstructionLoss(x_rec,x)

                if self.celoss:
                    out_cls = self.C(x_fake)
                    loss_CE += self.Loss_cls(out_cls,y)
                else:
                    score1,feature1 = self.DGC(x, latent_flag = True)
                    score2,feature2 = self.DGC(x_fake, latent_flag = True)
                    loss_CE += conditional_mmd_rbf(feature1, feature2, y, num_class=self.n_classes)


            for i in range(self.num_aug_domains):
                for j in range(i+1,self.num_aug_domains):
                    loss_diversity += self.Loss_distribution(fake[i], fake[j])
            
            loss_diversity = loss_diversity / 3.0
            total_loss = self.lambda_CE*loss_CE + self.lambda_cycle*loss_cycle - self.lambda_domain*(loss_diversity + loss_novel)
            
            x_fake_all = torch.cat(fake, dim=0)
            x_rec_all = torch.cat(rec, dim=0)
            
            total_loss.backward()

            losstemp[0].append(loss_novel.item())
            losstemp[1].append(loss_cycle.item())
            losstemp[2].append(loss_CE.item())
            losstemp[3].append(loss_diversity.item())

            if t % 100 == 0:
                x_fake_all = self.denormalize(x_fake_all)
                save_image(x_fake_all, '/output/results/{}_model_fake.jpg'.format(t))
            

            self.g_optimizer.step()
            self.g_optimizer.zero_grad()
            self.dgc_optimizer.zero_grad()
            #################################################################################################################################
            if self.search:
                
                ext = self.G.emb(classes).view(self.num_aug_domains, -1)    # 3,64
                
                lab = torch.randint_like(labels_all,10,device=self.device)

                target_domin = self.pre(ext.detach(),lab)

                x_fake_all = self.G(x_all, target_domin)
                score1,feature1 = self.DGC(x_all, latent_flag = True)
                score2,feature2 = self.DGC(x_fake_all, latent_flag = True)
                # lossreal = self.Loss_cls(self.DGC(x_all), labels_all)
                # lossfake = self.Loss_cls(self.DGC(x_fake_all), labels_all)
                lossreal = self.Loss_cls(score1, labels_all)
                lossfake = self.Loss_cls(score2, labels_all)
                
                lossfeature = self.con(torch.cat([F.normalize(feature1).unsqueeze(1), F.normalize(feature2).unsqueeze(1)], dim=1), labels_all)
                
                loss_DGC = (  lossreal + lossfake  ) * 0.5 + self.lambda_mi*lossfeature

                loss_DGC.backward()


                self.dgc_optimizer.step()

                self.dgc_scheduler.step()

                self.pre.beta.data += self.prelr * self.pre.beta.grad
                
                self.dgc_optimizer.zero_grad()

                self.pre.beta.grad.data.zero_()
                
                
                self.pre.beta.data = (self.pre.beta.data)/(torch.sum(self.pre.beta.data, 1, keepdim=True)+1e-4)
                
                flag=0
                
                for i in range(10):
                    for j in range(self.num_domains):
                        if self.pre.beta.data[i][j] < self.beta_min:
                            self.pre.beta.data[i][j] = self.beta_min+0.1
                            flag=flag+1
                        if self.pre.beta.data[i][j] > self.beta_max:
                            self.pre.beta.data[i][j] = self.beta_max-0.1
                            flag=flag+1
                
                
                losstemp[4].append(lossreal.item())
                losstemp[5].append(lossfake.item())

                if t % 100 == 0:
                    print()
                    print()
                    print('Iter:{}'.format(t))
                    print()
                    print('total_lossG:{}'.format(total_loss.item()))
                    print('total_loss_model:{}'.format(loss_DGC.item()))
                    print()
                    print('loss_novel:{}'.format(loss_novel.item()))
                    print('loss_cycle:{}'.format(loss_cycle.item()))
                    print('loss_CE:{}'.format(loss_CE.item()))
                    print('loss_diversity:{}'.format(loss_diversity.item()))
                    print('loss_real:{}'.format(lossreal.item()))
                    print('loss_fake:{}'.format(lossfake.item()))
                    print()
                    print()


                if flag > self.flaglimit:
                    print()
                    print()
                    print('Beta reset in iter:{}'.format(t))
                    print(self.pre.beta.data)
                    self.pre.beta.data=0.33*torch.randn((10, self.num_aug_domains),device=self.device)+0.33
                    print()
                    print()
                    


                if t % 100 == 0:
                    print('Acc of test iter:{}'.format(t))
                    new=self.testDGC(num=1000,loader=self.testloader)
                    acctemp[self.num_aug_domains].append(new[0])
                    if new[0]>self.acc_best:
                        self.acc_best = new[0]
                        torch.save(self.DGC.state_dict(),  f="/output/BEST_in_{}.pth".format(t))
                        print("New best {} ".format(self.acc_best))
                

                if t % 500 == 0:
                    print()
                    print()
                    print('Acc of train iter:{}'.format(t))
                    for k in range(self.num_aug_domains):
                        acctemp[k].append(self.testDGC(num=200,loader=self.loader[k]))
                    print()
                    print('Acc of aug iter:{}'.format(t))
                    for k in range(self.num_aug_domains):
                        acctemp2[k].append(self.testDGC2(num=200,loader=self.loader[k]))  
                    print()
                    print()
                

                
            else:

                # loss_DGC = ( self.Loss_cls(self.DGC(x_all), labels_all) + self.Loss_cls(self.DGC(x_fake_all.detach()), labels_all) ) * 0.5
                score1,feature1 = self.DGC(x_all, latent_flag = True)
                score2,feature2 = self.DGC(x_fake_all.detach(), latent_flag = True)
                # lossreal = self.Loss_cls(self.DGC(x_all), labels_all)
                # lossfake = self.Loss_cls(self.DGC(x_fake_all), labels_all)
                lossreal = self.Loss_cls(score1, labels_all)
                lossfake = self.Loss_cls(score2, labels_all)
                
                lossfeature = self.con(torch.cat([F.normalize(feature1).unsqueeze(1), F.normalize(feature2).unsqueeze(1)], dim=1), labels_all)
                
                loss_DGC = (  lossreal + lossfake  ) * 0.5 + self.lambda_mi*lossfeature
                # loss_DGC = (  lossreal + lossfake  ) * 0.5
                loss_DGC.backward()
                # print('{}_total_loss_model:{}'.format(t, loss_DGC.item()))
                self.dgc_optimizer.step()
                self.dgc_optimizer.zero_grad()
                losstemp[4].append(lossreal.item())
                losstemp[5].append(lossfake.item())
                
                if t % 100 == 0:
                    print()
                    print()
                    print('Iter:{}'.format(t))
                    print()
                    print('total_lossG:{}'.format(total_loss.item()))
                    print('total_loss_model:{}'.format(loss_DGC.item()))
                    print()
                    print('loss_novel:{}'.format(loss_novel.item()))
                    print('loss_cycle:{}'.format(loss_cycle.item()))
                    print('loss_CE:{}'.format(loss_CE.item()))
                    print('loss_diversity:{}'.format(loss_diversity.item()))
                    print('loss_real:{}'.format(lossreal.item()))
                    print('loss_fake:{}'.format(lossfake.item()))
                    print()
                    print()
                
                if t % 100 == 0:
                    print('Acc of test iter:{}'.format(t))
                    new=self.testDGC(num=1000,loader=self.testloader)
                    acctemp[self.num_aug_domains].append(new[0])
                    if new[0]>self.acc_best:
                        self.acc_best = new[0]
                        torch.save(self.DGC.state_dict(),  f="/output/BEST_in_{}.pth".format(t))
                        print("New best {} ".format(self.acc_best))
                
                if t % 500 == 0:
                    print()
                    print()
                    print('Acc of train iter:{}'.format(t))
                    for k in range(self.num_aug_domains):
                        acctemp[k].append(self.testDGC(num=200,loader=self.loader[k]))                  
                    print()
                    print('Acc of aug iter:{}'.format(t))
                    for k in range(self.num_aug_domains):
                        acctemp2[k].append(self.testDGC3(num=200,loader=self.loader[k]))  
                    print()
                    print()

            self.g_optimizer.zero_grad()

            #################################################################################################################################################
            # if t % 500 == 0:
            #     torch.save(self.G.state_dict(),  f="/output/G_iteration_{}.pth".format(t))
            #     torch.save(self.DGC.state_dict(),  f="/output/DGC_iteration_{}.pth".format(t))
            
            if t % self.resetG ==0:
                if self.mnistdataset:
                    self.G = Generator3(num_domains = 2 * self.num_domains).to(self.device)
                else:
                    self.G = Generator(num_domains = 2 * self.num_domains).to(self.device)
                self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr, (self.beta1, self.beta2))

                
            if t % 100 == 0:               

                x_fake_all = self.denormalize(x_fake_all)
                x_rec_all = self.denormalize(x_rec_all)
                save_image(x_fake_all, '/output/results/{}_fake.jpg'.format(t))

                save_image(x_rec_all, '/output/results/{}_rec.jpg'.format(t))
        
        torch.save(self.G.state_dict(),  f="/output/G_iteration_{}.pth".format(t))
        torch.save(self.DGC.state_dict(),  f="/output/DGC_iteration_{}.pth".format(t))
        torch.save(acctemp,  f="/output/acctemp.pth")
        torch.save(losstemp,  f="/output/losstemp.pth")
        return
    
    def trainGnorm(self,T):
        self.G.train()
        self.C.eval()
        self.D.eval()
        self.DGC_init()
        self.DGC.train()
        for t in range(T+1):
          
            for i in range(self.num_domains):
                try:
                    x, y = next(self.iter[i])
                except StopIteration:
                    self.iter[i] = iter(data.DataLoader(self.train_set[i], self.args.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True))
                    x, y = next(self.iter[i])
                x = x.to(device=self.device)
                y = y.to(device=self.device, dtype=torch.long)
                if i == 0:
                    x_all = x
                    labels_all = y
                else:
                    x_all = torch.cat((x_all,x),dim=0)
                    labels_all = torch.cat((labels_all,y),dim=0)

            
            loss_DGC = self.Loss_cls(self.DGC(x_all), labels_all)
            loss_DGC.backward()

            self.dgc_optimizer.step()
            self.dgc_scheduler.step()
            self.dgc_optimizer.zero_grad()

            if t % 50 == 0:
                print('{}_total_loss_model:{}'.format(t, loss_DGC.item()))


            if t % 500 == 0:
                print('iter:{}'.format(t))
                self.testDGC(num=1000,loader=self.testloader)
                print()

            if t % 500 == 0:
                torch.save(self.DGC.state_dict(),  f="/output/DGC_iteration_{}.pth".format(t))

            
        torch.save(self.DGC.state_dict(),  f="/output/End1_DGC_iteration_{}.pth".format(t))
        return

    def trainC(self,T):
        self.C.train()
        for t in range(T):
            # loss_CE = 0.0
            for i in range(self.num_domains):
                try:
                    x, y = next(self.iter[i])
                except StopIteration:
                    self.iter[i] = iter(data.DataLoader(self.train_set[i], self.args.batch_size, shuffle=True, num_workers=self.num_workers))
                    x, y = next(self.iter[i])
                
                if i == 0:
                    x_all = x
                    labels_all = y
                else:
                    x_all = torch.cat((x_all,x),dim=0)
                    labels_all = torch.cat((labels_all,y),dim=0)


            x_all = x_all.to(device=self.device)
            labels_all = labels_all.to(device=self.device, dtype=torch.long)

            out_cls = self.C(x_all)
            loss_CE = self.Loss_cls(out_cls, labels_all)
            # loss_CE = loss_CE/len(self.batImageGenTrainsDg)

            loss_CE.backward()
            self.c_optimizer.step()
            self.c_optimizer.zero_grad()
            print('{}_total_loss:{}'.format(t, loss_CE.item()))


            if t % 100 == 0:
                # self.test_workflow_C(self.C, self.batImageGenVals, self.args, t)
                torch.save(self.C.state_dict(),  f = os.path.join(self.save,"C_iteration_{}.pth".format(t)))

        return

    def trainD(self,T):
        self.D.train()
        for t in range(T):
            # loss_CE = 0.0
            for i in range(self.num_domains):
                try:
                    x, y = next(self.iter[i])
                except StopIteration:
                    self.iter[i] = iter(data.DataLoader(self.train_set[i], self.args.batch_size, shuffle=True, num_workers=self.num_workers))
                    x, y = next(self.iter[i])
                x = x.cuda()
                domain_labels = torch.tensor(int(i)).repeat(x.size(0)).cuda()
                if i == 0:
                    x_all = x
                    domain_labels_all = domain_labels
                else:
                    x_all = torch.cat((x_all,x),dim=0)
                    domain_labels_all = torch.cat((domain_labels_all,domain_labels),dim=0)

            out_cls = self.D(x_all)
            loss_CE = self.Loss_cls(out_cls, domain_labels_all)
            # loss_CE = loss_CE

            loss_CE.backward()
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()
            print('{}_total_loss:{}'.format(t, loss_CE.item()))


            if t % 100 == 0:
                # self.test_workflow_D(self.batImageGenVals, self.args, t)
                torch.save(self.D.state_dict(),  f=os.path.join(self.save,"D_iteration_{}.pth".format(t)))

        return


    def G_visualize(self):
        self.G.eval()
        self.G.load_state_dict(torch.load('checkpoints/G_iteration_450.pth'))

        for index, batImageGenVal in enumerate(self.batImageGenVals):
            x_real, cls = batImageGenVal.get_images_labels_batch()
            x_real = torch.from_numpy(np.array(x_real, dtype=np.float32))

            # wrap the inputs and labels in Variable
            x_real = Variable(x_real, requires_grad=False).cuda()
            x_real = x_real.to(self.device)
            label_org = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains).cuda()
            label_org[:, index] = 1.0
            new_idx = self.num_domains + index

            label_trg = torch.zeros(x_real.size(0), self.num_domains + self.num_aug_domains).cuda()
            label_trg[:, new_idx] = 1.0
            x_fake = self.G(x_real, label_trg)
            x_rec = self.G(x_fake, label_org)
            x_real = self.denormalize(x_real)
            x_fake = self.denormalize(x_fake)
            x_rec = self.denormalize(x_rec)
            save_image(x_real[0],'results/real.jpg')
            save_image(x_fake[0],'results/fake.jpg')
            save_image(x_rec[0], 'results/rec.jpg')

        return

    def denormalize(self,x):

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        mean = torch.tensor(mean).cuda()
        std = torch.tensor(std).cuda()

        x *= std.view(1, 3, 1, 1)
        x += mean.view(1, 3, 1, 1)
        return x



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()