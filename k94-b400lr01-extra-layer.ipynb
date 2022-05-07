{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "view-in-github"
   },
   "source": [
    "<a href=\"https://colab.research.google.com/github/stanbsky/int2_image_classifier/blob/main/kaggle94gpu.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "OD2wVFLtkbkN"
   },
   "outputs": [],
   "source": [
    "# importing libraries\n",
    "import os\n",
    "import torch\n",
    "import torchvision\n",
    "import tarfile\n",
    "from torchvision.datasets.utils import download_url\n",
    "from torch.utils.data import random_split\n",
    "from torchvision.datasets import ImageFolder\n",
    "from torchvision.transforms import ToTensor,ToPILImage\n",
    "import matplotlib.pyplot as plt\n",
    "from torchvision.utils import make_grid\n",
    "from torch.utils.data.dataloader import DataLoader\n",
    "from torchvision.utils import make_grid\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torchvision.transforms as tt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "TrM-k5_GVubi"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "id": "Cea6yx4amG2c"
   },
   "outputs": [],
   "source": [
    "import torchvision.datasets as datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "efdLf1xLk5_y",
    "outputId": "dd7073c3-3026-4d75-eee6-e0425ce35d3e"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<weakproxy at 0x7fa658c2cbd8 to Device at 0x7fa658c2bb00>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from numba import cuda\n",
    "cuda.select_device(7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "id": "hNiNZttPlHFe"
   },
   "outputs": [],
   "source": [
    "stats= ((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)) #mean and std\n",
    "train_tfm= tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), # transormation of data together\n",
    "                       tt.RandomHorizontalFlip(),\n",
    "                       tt.ToTensor(),tt.Normalize(*stats,inplace=True)])\n",
    "valid_tfm = tt.Compose([tt.ToTensor(),tt.Normalize(*stats)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 86,
     "referenced_widgets": [
      "c974d31e436a42899adbf2ef969307a4",
      "4c7fbb2405c84d7887d63fd6217cba4a",
      "350d156265104e2296caf536b05d26cc",
      "40b5952c97aa40169739936f0d549d02",
      "50c9aed288f14fcd94009217a59c91ff",
      "a448b41c426a4810a07d5eb92df1bc21",
      "29225a4d667e47f980b33c95c0850d16",
      "9ae1047b1b184d81bde7c14ca6ad72d5",
      "37da94684df543f3a81fe78c389f68dc",
      "0433c9c034b7466a98118f517c9ae2e8",
      "fb9ab6c5d81f48089ac0d9dc03c72e6b"
     ]
    },
    "id": "pxotEy_Al0J4",
    "outputId": "90e7b090-0622-4775-fcfa-39958472aa7c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Files already downloaded and verified\n"
     ]
    }
   ],
   "source": [
    "training_data = datasets.CIFAR10(\n",
    "    root=\"data\",\n",
    "    train=True,\n",
    "    download=True,\n",
    "    transform=train_tfm,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "3HUIV5yKmj4F",
    "outputId": "96675fe3-76e5-4e9f-9bc2-14b1d389996f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Files already downloaded and verified\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Download test data from open datasets.\n",
    "test_data = datasets.CIFAR10(\n",
    "    root=\"data\",\n",
    "    train=False,\n",
    "    download=True,\n",
    "    transform=valid_tfm,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "3eOTjMVem_T9",
    "outputId": "c3a532da-260b-4bd9-baef-144d73c92f92"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Dataset CIFAR10\n",
       "    Number of datapoints: 50000\n",
       "    Root location: data\n",
       "    Split: Train\n",
       "    StandardTransform\n",
       "Transform: Compose(\n",
       "               RandomCrop(size=(32, 32), padding=4)\n",
       "               RandomHorizontalFlip(p=0.5)\n",
       "               ToTensor()\n",
       "               Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201))\n",
       "           )"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "valid_ds = test_data\n",
    "train_ds = training_data\n",
    "train_ds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "iuMOTM7SrQa9",
    "outputId": "25d6add7-13f0-4ef8-b492-f3915b4aaa70"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([3, 32, 32]) 6\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "tensor([[[-0.4324, -0.1223, -0.0835,  ..., -1.2854, -0.4518,  0.0328],\n",
       "         [-0.7232, -0.3549, -0.1029,  ..., -0.8007, -0.2967,  0.0716],\n",
       "         [-0.5875, -0.3355,  0.0716,  ..., -0.2773, -0.0447,  0.0910],\n",
       "         ...,\n",
       "         [ 0.2267, -1.4598, -0.6844,  ...,  1.4673,  1.4091,  1.2735],\n",
       "         [ 1.1765, -0.2192,  0.0328,  ...,  1.4673,  1.1765,  0.9051],\n",
       "         [ 1.5061,  0.3430,  0.2267,  ...,  1.3122,  0.8276,  0.4981]],\n",
       "\n",
       "        [[-1.1006, -0.8646, -0.8646,  ..., -1.7889, -1.1399, -0.8646],\n",
       "         [-1.3562, -1.0809, -0.8449,  ..., -1.3759, -0.9826, -0.8056],\n",
       "         [-1.2579, -1.0809, -0.7072,  ..., -0.9432, -0.7466, -0.7662],\n",
       "         ...,\n",
       "         [-0.3532, -1.9463, -1.0612,  ...,  0.5908,  0.7481,  0.6694],\n",
       "         [ 0.6301, -0.7072, -0.4319,  ...,  0.5318,  0.4924,  0.3154],\n",
       "         [ 0.8661, -0.3139, -0.4319,  ...,  0.2564, -0.0189, -0.2352]],\n",
       "\n",
       "        [[-1.5971, -1.4410, -1.4215,  ..., -2.0068, -1.5580, -1.4605],\n",
       "         [-1.7531, -1.5580, -1.3434,  ..., -1.7531, -1.4995, -1.4800],\n",
       "         [-1.6556, -1.5190, -1.2654,  ..., -1.5190, -1.3825, -1.4800],\n",
       "         ...,\n",
       "         [-1.0508, -2.0068, -1.4410,  ..., -1.5580, -1.7141, -1.6946],\n",
       "         [-0.3094, -1.3239, -1.0703,  ..., -1.7922, -1.7531, -1.6751],\n",
       "         [-0.2313, -1.1093, -1.1678,  ..., -1.9092, -1.8507, -1.5385]]])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "img, label= train_ds[0]\n",
    "print(img.shape,label)\n",
    "img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "id": "JAFQAJUVr7U2"
   },
   "outputs": [],
   "source": [
    "def show_image(img,label):\n",
    "    print('Label: ', train_ds.classes[label],\"(\"+str(label)+\")\")\n",
    "    plt.imshow(img.permute(1,2,0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "id": "tj5jXuuBri6o"
   },
   "outputs": [],
   "source": [
    "batch_size=400"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "id": "n4kCM-hare00"
   },
   "outputs": [],
   "source": [
    "# Dataloader to load data in batches(mini batch)\n",
    "train_dl= DataLoader(train_ds,batch_size,shuffle=True, num_workers=2, pin_memory=True)\n",
    "valid_dl= DataLoader(valid_ds, batch_size, num_workers=2,pin_memory=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "id": "mcfsy3QptG2Y"
   },
   "outputs": [],
   "source": [
    "def show_batch(dl):\n",
    "    for images, labels in dl:\n",
    "        fig,ax= plt.subplots(figsize=(12,12))\n",
    "        ax.set_xticks([]) #hide ticks\n",
    "        ax.set_yticks([])\n",
    "        ax.imshow(make_grid(images[:64],nrow=8).permute(1,2,0))\n",
    "        break # printing only first 64 images from first batch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "id": "lPVFZltRtXt4"
   },
   "outputs": [],
   "source": [
    "def get_default_device():\n",
    "    \"\"\"Pick GPU if available, else CPU\"\"\"\n",
    "    if torch.cuda.is_available():\n",
    "        return torch.device('cuda')\n",
    "    else:\n",
    "        return torch.device('cpu')\n",
    "    \n",
    "def to_device(data, device):\n",
    "    \"\"\"Move tensor(s) to chosen device\"\"\"\n",
    "    if isinstance(data, (list,tuple)):\n",
    "        return [to_device(x, device) for x in data]\n",
    "    return data.to(device, non_blocking=True)\n",
    "\n",
    "class DeviceDataLoader():\n",
    "    \"\"\"Wrap a dataloader to move data to a device\"\"\"\n",
    "    def __init__(self, dl, device):\n",
    "        self.dl = dl\n",
    "        self.device = device\n",
    "        \n",
    "    def __iter__(self):\n",
    "        \"\"\"Yield a batch of data after moving it to device\"\"\"\n",
    "        for b in self.dl: \n",
    "            yield to_device(b, self.device)\n",
    "\n",
    "    def __len__(self):\n",
    "        \"\"\"Number of batches\"\"\"\n",
    "        return len(self.dl)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "SOksgE3wtYwZ",
    "outputId": "a9fb69e1-9e00-4139-d791-b7c305a7941a"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "device(type='cuda')"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "device = get_default_device()\n",
    "device"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "id": "hfVlteXctb6f"
   },
   "outputs": [],
   "source": [
    "train_dl= DeviceDataLoader(train_dl,device)\n",
    "valid_dl = DeviceDataLoader(valid_dl, device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "id": "PLmLZHBqtlF_"
   },
   "outputs": [],
   "source": [
    "def accuracy(outputs, labels):\n",
    "    _, preds = torch.max(outputs, dim=1)\n",
    "    return torch.tensor(torch.sum(preds == labels).item() / len(preds))\n",
    "\n",
    "class ImageClassificationBase(nn.Module):\n",
    "    def training_step(self, batch):\n",
    "        images, labels = batch \n",
    "        out = self(images)                  # Generate predictions\n",
    "        loss = F.cross_entropy(out, labels) # Calculate loss\n",
    "        return loss\n",
    "    \n",
    "    def validation_step(self, batch):\n",
    "        images, labels = batch \n",
    "        out = self(images)                    # Generate predictions\n",
    "        loss = F.cross_entropy(out, labels)   # Calculate loss\n",
    "        acc = accuracy(out, labels)           # Calculate accuracy\n",
    "        return {'val_loss': loss.detach(), 'val_acc': acc}\n",
    "        \n",
    "    def validation_epoch_end(self, outputs):\n",
    "        batch_losses = [x['val_loss'] for x in outputs]\n",
    "        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses\n",
    "        batch_accs = [x['val_acc'] for x in outputs]\n",
    "        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies\n",
    "        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}\n",
    "    \n",
    "    def epoch_end(self, epoch, result):\n",
    "        print(\"Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}\".format(\n",
    "            epoch, result['train_loss'], result['val_loss'], result['val_acc']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "id": "LcaZkKzQtomv"
   },
   "outputs": [],
   "source": [
    "def conv_block(in_channels, out_channels, pool=False):\n",
    "    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), \n",
    "              nn.BatchNorm2d(out_channels), \n",
    "              nn.ReLU(inplace=True)]\n",
    "    if pool: layers.append(nn.MaxPool2d(2))\n",
    "    return nn.Sequential(*layers)\n",
    "\n",
    "class ResNet9(ImageClassificationBase):\n",
    "    def __init__(self, in_channels, num_classes):\n",
    "        super().__init__()\n",
    "        \n",
    "        self.conv1 = conv_block(in_channels, 64)\n",
    "        self.conv2 = conv_block(64, 128, pool=True)\n",
    "        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))\n",
    "        \n",
    "        self.conv3 = conv_block(128, 256)\n",
    "        self.conv4 = conv_block(256, 512, pool=True)\n",
    "        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))\n",
    "        \n",
    "        self.conv5 = conv_block(512,1024)\n",
    "        self.conv6 = conv_block(1024,2048, pool=True)\n",
    "        self.res3 = nn.Sequential(conv_block(2048, 2048), conv_block(2048, 2048))\n",
    "        \n",
    "        self.classifier = nn.Sequential(nn.MaxPool2d(4), \n",
    "                                        nn.Flatten(), \n",
    "                                        nn.Linear(2048, num_classes))\n",
    "        \n",
    "    def forward(self, xb):\n",
    "        out = self.conv1(xb)\n",
    "        out = self.conv2(out)\n",
    "        out = self.res1(out) + out\n",
    "        out = self.conv3(out)\n",
    "        out = self.conv4(out)\n",
    "        out = self.res2(out) + out\n",
    "        out = self.conv5(out)\n",
    "        out = self.conv6(out)\n",
    "        out = self.res3(out) + out\n",
    "        out = self.classifier(out)\n",
    "        return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "O8BOig0Otrjz",
    "outputId": "a40ce9a4-8852-4595-9525-9841221a3391"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ResNet9(\n",
       "  (conv1): Sequential(\n",
       "    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (2): ReLU(inplace=True)\n",
       "  )\n",
       "  (conv2): Sequential(\n",
       "    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (2): ReLU(inplace=True)\n",
       "    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "  )\n",
       "  (res1): Sequential(\n",
       "    (0): Sequential(\n",
       "      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (2): ReLU(inplace=True)\n",
       "    )\n",
       "    (1): Sequential(\n",
       "      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (2): ReLU(inplace=True)\n",
       "    )\n",
       "  )\n",
       "  (conv3): Sequential(\n",
       "    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (2): ReLU(inplace=True)\n",
       "  )\n",
       "  (conv4): Sequential(\n",
       "    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (2): ReLU(inplace=True)\n",
       "    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "  )\n",
       "  (res2): Sequential(\n",
       "    (0): Sequential(\n",
       "      (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (2): ReLU(inplace=True)\n",
       "    )\n",
       "    (1): Sequential(\n",
       "      (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (2): ReLU(inplace=True)\n",
       "    )\n",
       "  )\n",
       "  (conv5): Sequential(\n",
       "    (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (2): ReLU(inplace=True)\n",
       "  )\n",
       "  (conv6): Sequential(\n",
       "    (0): Conv2d(1024, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "    (2): ReLU(inplace=True)\n",
       "    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "  )\n",
       "  (res3): Sequential(\n",
       "    (0): Sequential(\n",
       "      (0): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "      (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (2): ReLU(inplace=True)\n",
       "    )\n",
       "    (1): Sequential(\n",
       "      (0): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "      (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (2): ReLU(inplace=True)\n",
       "    )\n",
       "  )\n",
       "  (classifier): Sequential(\n",
       "    (0): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)\n",
       "    (1): Flatten(start_dim=1, end_dim=-1)\n",
       "    (2): Linear(in_features=2048, out_features=10, bias=True)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model= to_device(ResNet9(3,10), device)\n",
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "id": "hmbJUvR3t0QY"
   },
   "outputs": [],
   "source": [
    "@torch.no_grad()\n",
    "def evaluate(model, val_loader):\n",
    "    model.eval()\n",
    "    outputs = [model.validation_step(batch) for batch in val_loader]\n",
    "    return model.validation_epoch_end(outputs)\n",
    "\n",
    "def get_lr(optimizer):\n",
    "    for param_group in optimizer.param_groups:\n",
    "        return param_group['lr']\n",
    "\n",
    "def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, \n",
    "                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):\n",
    "    torch.cuda.empty_cache()  # Realsing cuda memory otherwise might get cuda out of memory error\n",
    "    history = []\n",
    "    \n",
    "    #custom optimizer with weight decay\n",
    "    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)\n",
    "    # Set up one-cycle learning rate scheduler\n",
    "    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, \n",
    "                                                steps_per_epoch=len(train_loader))\n",
    "    \n",
    "    for epoch in range(epochs):\n",
    "        # Training Phase \n",
    "        model.train() #Setting training mode\n",
    "        train_losses = []\n",
    "        lrs = []\n",
    "        for batch in train_loader:\n",
    "            loss = model.training_step(batch)\n",
    "            train_losses.append(loss)\n",
    "            loss.backward()\n",
    "            \n",
    "            # Gradient clipping\n",
    "            if grad_clip: \n",
    "                nn.utils.clip_grad_value_(model.parameters(), grad_clip)\n",
    "            \n",
    "            optimizer.step()\n",
    "            optimizer.zero_grad()\n",
    "            \n",
    "            # Record & update learning rate\n",
    "            lrs.append(get_lr(optimizer))\n",
    "            sched.step()\n",
    "        \n",
    "        # Validation phase\n",
    "        result = evaluate(model, val_loader)\n",
    "        result['train_loss'] = torch.stack(train_losses).mean().item()\n",
    "        result['lrs'] = lrs\n",
    "        model.epoch_end(epoch, result)\n",
    "        history.append(result)\n",
    "    return history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "_TPdh3fbt1fH",
    "outputId": "a22ad739-4bd6-459d-f905-64450f06bf4c"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[{'val_loss': 2.30283260345459, 'val_acc': 0.11819998919963837}]"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "history = [evaluate(model, valid_dl)]\n",
    "history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "id": "fFL3dFDnt6pb"
   },
   "outputs": [],
   "source": [
    "epochs = 140\n",
    "max_lr = 0.01\n",
    "grad_clip = 0.1\n",
    "weight_decay = 1e-4\n",
    "opt_func = torch.optim.Adam"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "01CAjQi2t8Ub",
    "outputId": "776b2d7d-902f-4f3c-ece0-24ccd16c4d53"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [0], train_loss: 0.2711, val_loss: 0.2829, val_acc: 0.9014\n",
      "Epoch [1], train_loss: 0.2047, val_loss: 0.2788, val_acc: 0.9027\n",
      "Epoch [2], train_loss: 0.1781, val_loss: 0.2654, val_acc: 0.9075\n",
      "Epoch [3], train_loss: 0.1614, val_loss: 0.2691, val_acc: 0.9084\n",
      "Epoch [4], train_loss: 0.1518, val_loss: 0.2715, val_acc: 0.9075\n",
      "Epoch [5], train_loss: 0.1385, val_loss: 0.3181, val_acc: 0.8959\n",
      "Epoch [6], train_loss: 0.1362, val_loss: 0.3040, val_acc: 0.9031\n",
      "Epoch [7], train_loss: 0.1401, val_loss: 0.3336, val_acc: 0.8941\n",
      "Epoch [8], train_loss: 0.1488, val_loss: 0.3087, val_acc: 0.9013\n",
      "Epoch [9], train_loss: 0.1604, val_loss: 0.3251, val_acc: 0.8949\n",
      "Epoch [10], train_loss: 0.1685, val_loss: 0.3541, val_acc: 0.8906\n",
      "Epoch [11], train_loss: 0.1801, val_loss: 0.3708, val_acc: 0.8810\n",
      "Epoch [12], train_loss: 0.1938, val_loss: 0.4418, val_acc: 0.8584\n",
      "Epoch [13], train_loss: 0.2011, val_loss: 0.3505, val_acc: 0.8866\n",
      "Epoch [14], train_loss: 0.2123, val_loss: 0.4265, val_acc: 0.8588\n",
      "Epoch [15], train_loss: 0.2153, val_loss: 0.4863, val_acc: 0.8485\n",
      "Epoch [16], train_loss: 0.2297, val_loss: 0.5826, val_acc: 0.8327\n",
      "Epoch [17], train_loss: 0.2367, val_loss: 0.4510, val_acc: 0.8516\n",
      "Epoch [18], train_loss: 0.2495, val_loss: 0.4119, val_acc: 0.8623\n",
      "Epoch [19], train_loss: 0.2608, val_loss: 0.5248, val_acc: 0.8285\n",
      "Epoch [20], train_loss: 0.2609, val_loss: 0.4058, val_acc: 0.8659\n",
      "Epoch [21], train_loss: 0.2620, val_loss: 0.4867, val_acc: 0.8421\n",
      "Epoch [22], train_loss: 0.2743, val_loss: 0.5272, val_acc: 0.8301\n",
      "Epoch [23], train_loss: 0.2786, val_loss: 0.3969, val_acc: 0.8634\n",
      "Epoch [24], train_loss: 0.2942, val_loss: 0.4938, val_acc: 0.8378\n",
      "Epoch [25], train_loss: 0.2940, val_loss: 0.5017, val_acc: 0.8349\n",
      "Epoch [26], train_loss: 0.3017, val_loss: 0.7231, val_acc: 0.7847\n",
      "Epoch [27], train_loss: 0.3017, val_loss: 0.5639, val_acc: 0.8225\n",
      "Epoch [28], train_loss: 0.3117, val_loss: 0.6499, val_acc: 0.7933\n",
      "Epoch [29], train_loss: 0.3185, val_loss: 0.4623, val_acc: 0.8443\n",
      "Epoch [30], train_loss: 0.3195, val_loss: 0.5341, val_acc: 0.8254\n",
      "Epoch [31], train_loss: 0.3268, val_loss: 0.5922, val_acc: 0.8115\n",
      "Epoch [32], train_loss: 0.3312, val_loss: 0.5979, val_acc: 0.8136\n",
      "Epoch [33], train_loss: 0.3302, val_loss: 0.6190, val_acc: 0.8004\n",
      "Epoch [34], train_loss: 0.3378, val_loss: 0.6680, val_acc: 0.7818\n",
      "Epoch [35], train_loss: 0.3375, val_loss: 0.7251, val_acc: 0.7685\n",
      "Epoch [36], train_loss: 0.3486, val_loss: 0.6395, val_acc: 0.7902\n",
      "Epoch [37], train_loss: 0.3422, val_loss: 0.8124, val_acc: 0.7575\n",
      "Epoch [38], train_loss: 0.3519, val_loss: 0.5163, val_acc: 0.8273\n",
      "Epoch [39], train_loss: 0.3437, val_loss: 0.6118, val_acc: 0.7976\n",
      "Epoch [40], train_loss: 0.3459, val_loss: 0.5662, val_acc: 0.8103\n",
      "Epoch [41], train_loss: 0.3421, val_loss: 0.4800, val_acc: 0.8414\n",
      "Epoch [42], train_loss: 0.3467, val_loss: 0.4438, val_acc: 0.8510\n",
      "Epoch [43], train_loss: 0.3453, val_loss: 0.5607, val_acc: 0.8104\n",
      "Epoch [44], train_loss: 0.3447, val_loss: 0.5588, val_acc: 0.8174\n",
      "Epoch [45], train_loss: 0.3479, val_loss: 0.5574, val_acc: 0.8200\n",
      "Epoch [46], train_loss: 0.3474, val_loss: 0.6783, val_acc: 0.7849\n",
      "Epoch [47], train_loss: 0.3361, val_loss: 0.5088, val_acc: 0.8304\n",
      "Epoch [48], train_loss: 0.3497, val_loss: 0.5207, val_acc: 0.8262\n",
      "Epoch [49], train_loss: 0.3401, val_loss: 0.6441, val_acc: 0.7869\n",
      "Epoch [50], train_loss: 0.3415, val_loss: 0.6463, val_acc: 0.7852\n",
      "Epoch [51], train_loss: 0.3406, val_loss: 0.5247, val_acc: 0.8256\n",
      "Epoch [52], train_loss: 0.3403, val_loss: 0.4588, val_acc: 0.8475\n",
      "Epoch [53], train_loss: 0.3442, val_loss: 0.8771, val_acc: 0.7488\n",
      "Epoch [54], train_loss: 0.3377, val_loss: 0.4565, val_acc: 0.8507\n",
      "Epoch [55], train_loss: 0.3356, val_loss: 0.8509, val_acc: 0.7512\n",
      "Epoch [56], train_loss: 0.3367, val_loss: 1.0379, val_acc: 0.7204\n",
      "Epoch [57], train_loss: 0.3327, val_loss: 0.8286, val_acc: 0.7687\n",
      "Epoch [58], train_loss: 0.3361, val_loss: 0.5581, val_acc: 0.8192\n",
      "Epoch [59], train_loss: 0.3291, val_loss: 0.4683, val_acc: 0.8484\n",
      "Epoch [60], train_loss: 0.3284, val_loss: 2.1608, val_acc: 0.5647\n",
      "Epoch [61], train_loss: 0.3229, val_loss: 0.5608, val_acc: 0.8147\n",
      "Epoch [62], train_loss: 0.3206, val_loss: 0.4276, val_acc: 0.8586\n",
      "Epoch [63], train_loss: 0.3180, val_loss: 0.5286, val_acc: 0.8290\n",
      "Epoch [64], train_loss: 0.3172, val_loss: 0.5475, val_acc: 0.8234\n",
      "Epoch [65], train_loss: 0.3131, val_loss: 0.4867, val_acc: 0.8462\n",
      "Epoch [66], train_loss: 0.3177, val_loss: 0.5252, val_acc: 0.8265\n",
      "Epoch [67], train_loss: 0.3118, val_loss: 0.4659, val_acc: 0.8399\n",
      "Epoch [68], train_loss: 0.3089, val_loss: 0.5477, val_acc: 0.8288\n",
      "Epoch [69], train_loss: 0.3010, val_loss: 0.6516, val_acc: 0.8058\n",
      "Epoch [70], train_loss: 0.3016, val_loss: 0.4530, val_acc: 0.8484\n",
      "Epoch [71], train_loss: 0.2965, val_loss: 0.4920, val_acc: 0.8408\n",
      "Epoch [72], train_loss: 0.2963, val_loss: 0.5467, val_acc: 0.8252\n",
      "Epoch [73], train_loss: 0.2935, val_loss: 0.4725, val_acc: 0.8433\n",
      "Epoch [74], train_loss: 0.2905, val_loss: 0.4527, val_acc: 0.8536\n",
      "Epoch [75], train_loss: 0.2837, val_loss: 0.4134, val_acc: 0.8639\n",
      "Epoch [76], train_loss: 0.2831, val_loss: 0.7153, val_acc: 0.7950\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [77], train_loss: 0.2784, val_loss: 0.5154, val_acc: 0.8402\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [78], train_loss: 0.2755, val_loss: 0.5501, val_acc: 0.8275\n",
      "Epoch [79], train_loss: 0.2706, val_loss: 0.4500, val_acc: 0.8529\n",
      "Epoch [80], train_loss: 0.2681, val_loss: 0.4345, val_acc: 0.8597\n",
      "Epoch [81], train_loss: 0.2674, val_loss: 0.3736, val_acc: 0.8775\n",
      "Epoch [82], train_loss: 0.2602, val_loss: 0.3784, val_acc: 0.8753\n",
      "Epoch [83], train_loss: 0.2601, val_loss: 0.4414, val_acc: 0.8590\n",
      "Epoch [84], train_loss: 0.2498, val_loss: 0.4051, val_acc: 0.8692\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [85], train_loss: 0.2487, val_loss: 0.4202, val_acc: 0.8656\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [86], train_loss: 0.2397, val_loss: 0.5166, val_acc: 0.8395\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [87], train_loss: 0.2363, val_loss: 0.4318, val_acc: 0.8620\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [88], train_loss: 0.2348, val_loss: 0.3916, val_acc: 0.8693\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [89], train_loss: 0.2288, val_loss: 0.3830, val_acc: 0.8717\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [90], train_loss: 0.2237, val_loss: 0.4224, val_acc: 0.8661\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [91], train_loss: 0.2181, val_loss: 0.4773, val_acc: 0.8563\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [92], train_loss: 0.2133, val_loss: 0.3640, val_acc: 0.8840\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [93], train_loss: 0.2079, val_loss: 0.3995, val_acc: 0.8760\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [94], train_loss: 0.2036, val_loss: 0.3710, val_acc: 0.8801\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [95], train_loss: 0.2003, val_loss: 0.3970, val_acc: 0.8754\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [96], train_loss: 0.1935, val_loss: 0.3568, val_acc: 0.8864\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [97], train_loss: 0.1834, val_loss: 0.4103, val_acc: 0.8761\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [98], train_loss: 0.1801, val_loss: 0.3183, val_acc: 0.8980\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [99], train_loss: 0.1732, val_loss: 0.3381, val_acc: 0.8953\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [100], train_loss: 0.1632, val_loss: 0.3828, val_acc: 0.8820\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [101], train_loss: 0.1656, val_loss: 0.3727, val_acc: 0.8868\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [102], train_loss: 0.1602, val_loss: 0.3236, val_acc: 0.9025\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [103], train_loss: 0.1517, val_loss: 0.3414, val_acc: 0.8941\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [104], train_loss: 0.1448, val_loss: 0.3185, val_acc: 0.9002\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [105], train_loss: 0.1402, val_loss: 0.3247, val_acc: 0.9004\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [106], train_loss: 0.1396, val_loss: 0.3211, val_acc: 0.9033\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [107], train_loss: 0.1257, val_loss: 0.3211, val_acc: 0.9043\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [108], train_loss: 0.1211, val_loss: 0.3043, val_acc: 0.9031\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [109], train_loss: 0.1154, val_loss: 0.3176, val_acc: 0.9028\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [110], train_loss: 0.1051, val_loss: 0.3224, val_acc: 0.9067\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [111], train_loss: 0.1070, val_loss: 0.3156, val_acc: 0.9086\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [112], train_loss: 0.0966, val_loss: 0.3172, val_acc: 0.9064\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [113], train_loss: 0.0906, val_loss: 0.3232, val_acc: 0.9082\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [114], train_loss: 0.0848, val_loss: 0.2994, val_acc: 0.9140\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "Traceback (most recent call last):\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [115], train_loss: 0.0784, val_loss: 0.2949, val_acc: 0.9153\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [116], train_loss: 0.0737, val_loss: 0.3052, val_acc: 0.9123\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [117], train_loss: 0.0677, val_loss: 0.2985, val_acc: 0.9171\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [118], train_loss: 0.0617, val_loss: 0.2978, val_acc: 0.9172\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [119], train_loss: 0.0563, val_loss: 0.3016, val_acc: 0.9177\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [120], train_loss: 0.0559, val_loss: 0.3216, val_acc: 0.9114\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [121], train_loss: 0.0504, val_loss: 0.3120, val_acc: 0.9167\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [122], train_loss: 0.0473, val_loss: 0.2964, val_acc: 0.9175\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [123], train_loss: 0.0460, val_loss: 0.2963, val_acc: 0.9186\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [124], train_loss: 0.0392, val_loss: 0.3069, val_acc: 0.9203\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [125], train_loss: 0.0348, val_loss: 0.2987, val_acc: 0.9222\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [126], train_loss: 0.0335, val_loss: 0.2961, val_acc: 0.9233\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [127], train_loss: 0.0317, val_loss: 0.2997, val_acc: 0.9211\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [128], train_loss: 0.0312, val_loss: 0.3012, val_acc: 0.9222\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [129], train_loss: 0.0263, val_loss: 0.3002, val_acc: 0.9214\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [130], train_loss: 0.0250, val_loss: 0.3021, val_acc: 0.9217\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "self._shutdown_workers()Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    \n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [131], train_loss: 0.0236, val_loss: 0.3004, val_acc: 0.9220\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [132], train_loss: 0.0234, val_loss: 0.2998, val_acc: 0.9222\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [133], train_loss: 0.0227, val_loss: 0.2984, val_acc: 0.9230\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [134], train_loss: 0.0222, val_loss: 0.2975, val_acc: 0.9236\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [135], train_loss: 0.0206, val_loss: 0.2977, val_acc: 0.9235\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [136], train_loss: 0.0207, val_loss: 0.2989, val_acc: 0.9233\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [137], train_loss: 0.0200, val_loss: 0.2987, val_acc: 0.9231\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [138], train_loss: 0.0202, val_loss: 0.2967, val_acc: 0.9231\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n",
      "Exception ignored in: <bound method _MultiProcessingDataLoaderIter.__del__ of <torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7fa65885f438>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1203, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"/usr/local/lib/python3.6/dist-packages/torch/utils/data/dataloader.py\", line 1177, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"/usr/lib/python3.6/multiprocessing/process.py\", line 122, in join\n",
      "    assert self._parent_pid == os.getpid(), 'can only join a child process'\n",
      "AssertionError: can only join a child process\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [139], train_loss: 0.0203, val_loss: 0.2991, val_acc: 0.9232\n",
      "CPU times: user 1h 26min 13s, sys: 40min 51s, total: 2h 7min 4s\n",
      "Wall time: 2h 4min 53s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, \n",
    "                             grad_clip=grad_clip, \n",
    "                             weight_decay=weight_decay, \n",
    "                             opt_func=opt_func)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "id": "jvJzQAeNjB6Y"
   },
   "outputs": [],
   "source": [
    "def plot_accuracies(history):\n",
    "    accuracies = [x['val_acc'] for x in history]\n",
    "    plt.figure(figsize=(10,6))\n",
    "    plt.plot(accuracies, '-x')\n",
    "    plt.xlabel('epoch')\n",
    "    plt.ylabel('accuracy')\n",
    "    plt.title('Accuracy vs. No. of epochs');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 171
    },
    "id": "fYuANPlfjIvt",
    "outputId": "08453f89-0933-4bd9-e02a-d760aad8714f"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_accuracies(history)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "id": "JV6cn39HjTF-"
   },
   "outputs": [],
   "source": [
    "def plot_losses(history):\n",
    "    train_losses = [x.get('train_loss') for x in history]\n",
    "    val_losses = [x['val_loss'] for x in history]\n",
    "    plt.figure(figsize=(10,6))\n",
    "    plt.plot(train_losses, '-bx')\n",
    "    plt.plot(val_losses, '-rx')\n",
    "    plt.xlabel('epoch')\n",
    "    plt.ylabel('loss')\n",
    "    plt.legend(['Training', 'Validation'])\n",
    "    plt.title('Loss vs. No. of epochs');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 171
    },
    "id": "kuky-BEDjUw_",
    "outputId": "e33b1eb9-e7c7-483d-829a-75bce6a074ae"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_losses(history)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "id": "ToUcs2S6jZl_"
   },
   "outputs": [],
   "source": [
    "def plot_lrs(history):\n",
    "    lrs = np.concatenate([x.get('lrs', []) for x in history])\n",
    "    plt.figure(figsize=(10,6))\n",
    "    plt.plot(lrs)\n",
    "    plt.xlabel('Batch no.')\n",
    "    plt.ylabel('Learning rate')\n",
    "    plt.title('Learning Rate vs. Batch no.');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 171
    },
    "id": "V1ao5oIJjboF",
    "outputId": "23d3a338-12ab-42be-b2de-8a2b3d55ccae"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_lrs(history)"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "include_colab_link": true,
   "name": "kaggle94gpu.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "0433c9c034b7466a98118f517c9ae2e8": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "29225a4d667e47f980b33c95c0850d16": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "350d156265104e2296caf536b05d26cc": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_9ae1047b1b184d81bde7c14ca6ad72d5",
      "max": 170498071,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_37da94684df543f3a81fe78c389f68dc",
      "value": 170498071
     }
    },
    "37da94684df543f3a81fe78c389f68dc": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "40b5952c97aa40169739936f0d549d02": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_0433c9c034b7466a98118f517c9ae2e8",
      "placeholder": "​",
      "style": "IPY_MODEL_fb9ab6c5d81f48089ac0d9dc03c72e6b",
      "value": " 170499072/? [00:02&lt;00:00, 62626240.17it/s]"
     }
    },
    "4c7fbb2405c84d7887d63fd6217cba4a": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_a448b41c426a4810a07d5eb92df1bc21",
      "placeholder": "​",
      "style": "IPY_MODEL_29225a4d667e47f980b33c95c0850d16",
      "value": ""
     }
    },
    "50c9aed288f14fcd94009217a59c91ff": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "9ae1047b1b184d81bde7c14ca6ad72d5": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "a448b41c426a4810a07d5eb92df1bc21": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "c974d31e436a42899adbf2ef969307a4": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_4c7fbb2405c84d7887d63fd6217cba4a",
       "IPY_MODEL_350d156265104e2296caf536b05d26cc",
       "IPY_MODEL_40b5952c97aa40169739936f0d549d02"
      ],
      "layout": "IPY_MODEL_50c9aed288f14fcd94009217a59c91ff"
     }
    },
    "fb9ab6c5d81f48089ac0d9dc03c72e6b": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    }
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
