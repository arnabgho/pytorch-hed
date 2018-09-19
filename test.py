# import torch libraries
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import pylab
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# import the utility functions
from model import HED
from dataproc import TestDataset


# fix random seed
rng = np.random.RandomState(37148)

# create instance of HED model
net = HED()
net.cuda()

# load the weights for the model
net.load_state_dict(torch.load('./train/HED.pth'))
net.eval()

# batch size
nBatch = 1

# load the images dataset
dataRoot = './cartoon_portrait_clean' # '/home/arnab/tinkering-projects/pytorch-hed/data/HED-BSDS/test'
std=[0.229, 0.224, 0.225]
mean=[0.485, 0.456, 0.406]

transform = transforms.Compose([
                    transforms.ToTensor(),transforms.Normalize(mean,std)])
# create data loaders from dataset
testPath = dataRoot+'/test.lst'
testDataset = TestDataset(testPath,dataRoot,transform)
testDataloader = DataLoader(testDataset, batch_size=nBatch)

def grayTrans(img):
    img = (1-img.numpy()[0][0])*255.0
    img = (img).astype(np.uint8)
    return img

def plotResults(images, size):
    pylab.rcParams['figure.figsize'] = size, size

    nPlots = len(images)
    titles = ['HED', 'S1', 'S2', 'S3', 'S4']
    plt.figure()
    for i in range(0, len(images)):
        s=plt.subplot(1,nPlots,i+1)
        plt.imshow(images[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
        s.set_title(titles[i],fontsize=35)
    plt.tight_layout()
    plt.show()

output_dir = './cartoon_portrait_clean_edges/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

nVisualizenVisual  = 10
for i, sample in enumerate(testDataloader):
    # get input sample image
    inp, fname = sample
    inp = Variable(inp.cuda())

    # perform forward computation
    s1,s2,s3,s4,s5,s6 = net.forward(inp)

    # convert back to numpy arrays
    out = []
    out.append(grayTrans(s6.data.cpu()))
    out.append(grayTrans(s1.data.cpu()))
    out.append(grayTrans(s2.data.cpu()))
    out.append(grayTrans(s3.data.cpu()))
    out.append(grayTrans(s4.data.cpu()))

    print(fname)
    img = Image.fromarray(out[0], 'L')
    img.save( output_dir + fname[0].split('.',1)[0]+'.png')

    # visualize every 10th image
    #if i%nVisualize == 0:
    #    plotResults(out, 25)

