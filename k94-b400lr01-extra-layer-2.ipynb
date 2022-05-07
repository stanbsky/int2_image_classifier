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
   "execution_count": 1,
   "metadata": {
    "id": "OD2wVFLtkbkN"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Matplotlib created a temporary config/cache directory at /tmp/matplotlib-oc0woom6 because the default path (/run/user/141228/cache/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.\n"
     ]
    }
   ],
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
   "execution_count": 2,
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
   "execution_count": 3,
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
       "<weakproxy at 0x7f489cf59b88 to Device at 0x7f489cf58a58>"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from numba import cuda\n",
    "cuda.select_device(6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
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
   "execution_count": 5,
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
   "execution_count": 6,
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
   "execution_count": 7,
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
     "execution_count": 7,
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
   "execution_count": 8,
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
       "tensor([[[-0.7426, -0.4324, -0.1223,  ..., -1.4598, -1.2854, -0.4518],\n",
       "         [-1.1303, -0.7232, -0.3549,  ..., -1.0527, -0.8007, -0.2967],\n",
       "         [-0.5681, -0.5875, -0.3355,  ..., -0.5487, -0.2773, -0.0447],\n",
       "         ...,\n",
       "         [ 0.6725,  0.2267, -1.4598,  ...,  1.6030,  1.4673,  1.4091],\n",
       "         [ 0.9245,  1.1765, -0.2192,  ...,  1.8356,  1.4673,  1.1765],\n",
       "         [ 1.5642,  1.5061,  0.3430,  ...,  1.7581,  1.3122,  0.8276]],\n",
       "\n",
       "        [[-1.2972, -1.1006, -0.8646,  ..., -1.7889, -1.7889, -1.1399],\n",
       "         [-1.6512, -1.3562, -1.0809,  ..., -1.4742, -1.3759, -0.9826],\n",
       "         [-1.1792, -1.2579, -1.0809,  ..., -1.0612, -0.9432, -0.7466],\n",
       "         ...,\n",
       "         [ 0.1974, -0.3532, -1.9463,  ...,  0.9251,  0.5908,  0.7481],\n",
       "         [ 0.4138,  0.6301, -0.7072,  ...,  1.1611,  0.5318,  0.4924],\n",
       "         [ 0.9841,  0.8661, -0.3139,  ...,  1.0038,  0.2564, -0.0189]],\n",
       "\n",
       "        [[-1.6946, -1.5971, -1.4410,  ..., -1.8117, -2.0068, -1.5580],\n",
       "         [-1.9287, -1.7531, -1.5580,  ..., -1.6556, -1.7531, -1.4995],\n",
       "         [-1.4995, -1.6556, -1.5190,  ..., -1.4410, -1.5190, -1.3825],\n",
       "         ...,\n",
       "         [-0.8557, -1.0508, -2.0068,  ..., -0.3484, -1.5580, -1.7141],\n",
       "         [-0.5240, -0.3094, -1.3239,  ..., -0.4460, -1.7922, -1.7531],\n",
       "         [-0.1728, -0.2313, -1.1093,  ..., -0.5240, -1.9092, -1.8507]]])"
      ]
     },
     "execution_count": 8,
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
   "execution_count": 9,
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
   "execution_count": 10,
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
   "execution_count": 11,
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
   "execution_count": 12,
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
   "execution_count": 13,
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
   "execution_count": 14,
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
     "execution_count": 14,
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
   "execution_count": 15,
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
   "execution_count": 16,
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
   "execution_count": 17,
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
   "execution_count": 18,
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
     "execution_count": 18,
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
   "execution_count": 19,
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
   "execution_count": 20,
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
       "[{'val_loss': 2.302459478378296, 'val_acc': 0.10010000318288803}]"
      ]
     },
     "execution_count": 20,
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
   "execution_count": 21,
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
   "execution_count": 22,
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
      "Epoch [0], train_loss: 1.9274, val_loss: 1.5135, val_acc: 0.4396\n",
      "Epoch [1], train_loss: 1.3522, val_loss: 1.1825, val_acc: 0.5744\n",
      "Epoch [2], train_loss: 1.0629, val_loss: 0.9771, val_acc: 0.6551\n",
      "Epoch [3], train_loss: 0.8672, val_loss: 1.0140, val_acc: 0.6481\n",
      "Epoch [4], train_loss: 0.7149, val_loss: 0.9946, val_acc: 0.6795\n",
      "Epoch [5], train_loss: 0.6112, val_loss: 0.6064, val_acc: 0.7966\n",
      "Epoch [6], train_loss: 0.5549, val_loss: 0.5922, val_acc: 0.7967\n",
      "Epoch [7], train_loss: 0.4692, val_loss: 0.6546, val_acc: 0.7955\n",
      "Epoch [8], train_loss: 0.4673, val_loss: 0.6375, val_acc: 0.8045\n",
      "Epoch [9], train_loss: 0.4326, val_loss: 0.5240, val_acc: 0.8117\n",
      "Epoch [10], train_loss: 0.4299, val_loss: 0.6045, val_acc: 0.7953\n",
      "Epoch [11], train_loss: 0.4234, val_loss: 0.5574, val_acc: 0.8123\n",
      "Epoch [12], train_loss: 0.4207, val_loss: 0.6454, val_acc: 0.7963\n",
      "Epoch [13], train_loss: 0.3735, val_loss: 0.7362, val_acc: 0.7759\n",
      "Epoch [14], train_loss: 0.3849, val_loss: 0.4842, val_acc: 0.8385\n",
      "Epoch [15], train_loss: 0.4571, val_loss: 0.8822, val_acc: 0.7782\n",
      "Epoch [16], train_loss: 0.3761, val_loss: 0.7964, val_acc: 0.7683\n",
      "Epoch [17], train_loss: 0.3577, val_loss: 0.8138, val_acc: 0.7623\n",
      "Epoch [18], train_loss: 0.3630, val_loss: 0.5778, val_acc: 0.8047\n",
      "Epoch [19], train_loss: 0.3604, val_loss: 1.1172, val_acc: 0.7105\n",
      "Epoch [20], train_loss: 0.3650, val_loss: 0.7381, val_acc: 0.7758\n",
      "Epoch [21], train_loss: 0.3494, val_loss: 0.5272, val_acc: 0.8315\n",
      "Epoch [22], train_loss: 0.3619, val_loss: 0.5137, val_acc: 0.8275\n",
      "Epoch [23], train_loss: 0.3641, val_loss: 0.5496, val_acc: 0.8208\n",
      "Epoch [24], train_loss: 0.3676, val_loss: 0.6100, val_acc: 0.7986\n",
      "Epoch [25], train_loss: 0.3745, val_loss: 0.4906, val_acc: 0.8412\n",
      "Epoch [26], train_loss: 0.3630, val_loss: 0.6990, val_acc: 0.7777\n",
      "Epoch [27], train_loss: 0.3620, val_loss: 0.5328, val_acc: 0.8274\n",
      "Epoch [28], train_loss: 0.3678, val_loss: 0.7254, val_acc: 0.7761\n",
      "Epoch [29], train_loss: 0.3694, val_loss: 0.5504, val_acc: 0.8214\n",
      "Epoch [30], train_loss: 0.3645, val_loss: 0.5614, val_acc: 0.8162\n",
      "Epoch [31], train_loss: 0.3654, val_loss: 0.9163, val_acc: 0.7211\n",
      "Epoch [32], train_loss: 0.3722, val_loss: 0.6315, val_acc: 0.7867\n",
      "Epoch [33], train_loss: 0.3745, val_loss: 0.6759, val_acc: 0.7765\n",
      "Epoch [34], train_loss: 0.3790, val_loss: 0.5650, val_acc: 0.8063\n",
      "Epoch [35], train_loss: 0.3834, val_loss: 0.5676, val_acc: 0.8079\n",
      "Epoch [36], train_loss: 0.3746, val_loss: 0.5620, val_acc: 0.8136\n",
      "Epoch [37], train_loss: 0.3819, val_loss: 0.6829, val_acc: 0.7845\n",
      "Epoch [38], train_loss: 0.3761, val_loss: 0.6185, val_acc: 0.7919\n",
      "Epoch [39], train_loss: 0.3798, val_loss: 0.5360, val_acc: 0.8205\n",
      "Epoch [40], train_loss: 0.3756, val_loss: 0.6702, val_acc: 0.7802\n",
      "Epoch [41], train_loss: 0.3838, val_loss: 0.4869, val_acc: 0.8324\n",
      "Epoch [42], train_loss: 0.3780, val_loss: 1.0231, val_acc: 0.7078\n",
      "Epoch [43], train_loss: 0.3791, val_loss: 0.6182, val_acc: 0.7957\n",
      "Epoch [44], train_loss: 0.3804, val_loss: 0.6609, val_acc: 0.7948\n",
      "Epoch [45], train_loss: 0.3731, val_loss: 0.5315, val_acc: 0.8185\n",
      "Epoch [46], train_loss: 0.3712, val_loss: 0.5955, val_acc: 0.8027\n",
      "Epoch [47], train_loss: 0.3693, val_loss: 0.4647, val_acc: 0.8437\n",
      "Epoch [48], train_loss: 0.3696, val_loss: 0.5983, val_acc: 0.7993\n",
      "Epoch [49], train_loss: 0.3665, val_loss: 0.6548, val_acc: 0.7813\n",
      "Epoch [50], train_loss: 0.3702, val_loss: 0.5757, val_acc: 0.8069\n",
      "Epoch [51], train_loss: 0.3612, val_loss: 0.6393, val_acc: 0.7935\n",
      "Epoch [52], train_loss: 0.3568, val_loss: 0.6091, val_acc: 0.8001\n",
      "Epoch [53], train_loss: 0.3583, val_loss: 0.5555, val_acc: 0.8188\n",
      "Epoch [54], train_loss: 0.3563, val_loss: 0.6441, val_acc: 0.7968\n",
      "Epoch [55], train_loss: 0.3621, val_loss: 0.6995, val_acc: 0.7852\n",
      "Epoch [56], train_loss: 0.3488, val_loss: 0.6642, val_acc: 0.7853\n",
      "Epoch [57], train_loss: 0.3502, val_loss: 0.6285, val_acc: 0.7906\n",
      "Epoch [58], train_loss: 0.3452, val_loss: 0.4651, val_acc: 0.8439\n",
      "Epoch [59], train_loss: 0.3456, val_loss: 0.4948, val_acc: 0.8383\n",
      "Epoch [60], train_loss: 0.3487, val_loss: 0.5159, val_acc: 0.8357\n",
      "Epoch [61], train_loss: 0.3400, val_loss: 0.4515, val_acc: 0.8466\n",
      "Epoch [62], train_loss: 0.3410, val_loss: 0.4432, val_acc: 0.8489\n",
      "Epoch [63], train_loss: 0.3357, val_loss: 0.6108, val_acc: 0.7929\n",
      "Epoch [64], train_loss: 0.3333, val_loss: 0.4177, val_acc: 0.8612\n",
      "Epoch [65], train_loss: 0.3215, val_loss: 0.4216, val_acc: 0.8564\n",
      "Epoch [66], train_loss: 0.3236, val_loss: 0.5003, val_acc: 0.8360\n",
      "Epoch [67], train_loss: 0.3227, val_loss: 0.4404, val_acc: 0.8533\n",
      "Epoch [68], train_loss: 0.3194, val_loss: 0.4723, val_acc: 0.8392\n",
      "Epoch [69], train_loss: 0.3092, val_loss: 0.3758, val_acc: 0.8709\n",
      "Epoch [70], train_loss: 0.3079, val_loss: 0.7249, val_acc: 0.7566\n",
      "Epoch [71], train_loss: 0.3091, val_loss: 0.4376, val_acc: 0.8519\n",
      "Epoch [72], train_loss: 0.3036, val_loss: 0.4594, val_acc: 0.8464\n",
      "Epoch [73], train_loss: 0.3040, val_loss: 0.4117, val_acc: 0.8627\n",
      "Epoch [74], train_loss: 0.3021, val_loss: 0.5343, val_acc: 0.8259\n",
      "Epoch [75], train_loss: 0.2937, val_loss: 0.5175, val_acc: 0.8277\n",
      "Epoch [76], train_loss: 0.2864, val_loss: 0.4724, val_acc: 0.8466\n",
      "Epoch [77], train_loss: 0.2876, val_loss: 0.5001, val_acc: 0.8400\n",
      "Epoch [78], train_loss: 0.2833, val_loss: 0.3781, val_acc: 0.8755\n",
      "Epoch [79], train_loss: 0.2773, val_loss: 0.3846, val_acc: 0.8682\n",
      "Epoch [80], train_loss: 0.2724, val_loss: 0.4277, val_acc: 0.8558\n",
      "Epoch [81], train_loss: 0.2709, val_loss: 0.5993, val_acc: 0.8215\n",
      "Epoch [82], train_loss: 0.2668, val_loss: 0.3722, val_acc: 0.8728\n",
      "Epoch [83], train_loss: 0.2606, val_loss: 0.4073, val_acc: 0.8639\n",
      "Epoch [84], train_loss: 0.2552, val_loss: 0.3898, val_acc: 0.8708\n",
      "Epoch [85], train_loss: 0.2478, val_loss: 0.5269, val_acc: 0.8303\n",
      "Epoch [86], train_loss: 0.2433, val_loss: 0.3535, val_acc: 0.8837\n",
      "Epoch [87], train_loss: 0.2437, val_loss: 0.4122, val_acc: 0.8665\n",
      "Epoch [88], train_loss: 0.2374, val_loss: 0.3837, val_acc: 0.8752\n",
      "Epoch [89], train_loss: 0.2344, val_loss: 0.4420, val_acc: 0.8560\n",
      "Epoch [90], train_loss: 0.2230, val_loss: 0.4140, val_acc: 0.8688\n",
      "Epoch [91], train_loss: 0.2224, val_loss: 0.3982, val_acc: 0.8724\n",
      "Epoch [92], train_loss: 0.2156, val_loss: 0.4059, val_acc: 0.8728\n",
      "Epoch [93], train_loss: 0.2123, val_loss: 0.3248, val_acc: 0.8928\n",
      "Epoch [94], train_loss: 0.2057, val_loss: 0.3658, val_acc: 0.8826\n",
      "Epoch [95], train_loss: 0.1977, val_loss: 0.3531, val_acc: 0.8879\n",
      "Epoch [96], train_loss: 0.1944, val_loss: 0.4285, val_acc: 0.8628\n",
      "Epoch [97], train_loss: 0.1878, val_loss: 0.3809, val_acc: 0.8801\n",
      "Epoch [98], train_loss: 0.1836, val_loss: 0.3126, val_acc: 0.8990\n",
      "Epoch [99], train_loss: 0.1795, val_loss: 0.4225, val_acc: 0.8709\n",
      "Epoch [100], train_loss: 0.1736, val_loss: 0.3478, val_acc: 0.8897\n",
      "Epoch [101], train_loss: 0.1648, val_loss: 0.3315, val_acc: 0.8950\n",
      "Epoch [102], train_loss: 0.1623, val_loss: 0.3640, val_acc: 0.8852\n",
      "Epoch [103], train_loss: 0.1547, val_loss: 0.3542, val_acc: 0.8914\n",
      "Epoch [104], train_loss: 0.1471, val_loss: 0.3440, val_acc: 0.8933\n",
      "Epoch [105], train_loss: 0.1364, val_loss: 0.3080, val_acc: 0.9025\n",
      "Epoch [106], train_loss: 0.1332, val_loss: 0.3281, val_acc: 0.8998\n",
      "Epoch [107], train_loss: 0.1313, val_loss: 0.3174, val_acc: 0.9034\n",
      "Epoch [108], train_loss: 0.1219, val_loss: 0.3429, val_acc: 0.9001\n",
      "Epoch [109], train_loss: 0.1160, val_loss: 0.3379, val_acc: 0.9025\n",
      "Epoch [110], train_loss: 0.1096, val_loss: 0.3202, val_acc: 0.9038\n",
      "Epoch [111], train_loss: 0.1066, val_loss: 0.2941, val_acc: 0.9103\n",
      "Epoch [112], train_loss: 0.0982, val_loss: 0.2921, val_acc: 0.9124\n",
      "Epoch [113], train_loss: 0.0916, val_loss: 0.3100, val_acc: 0.9077\n",
      "Epoch [114], train_loss: 0.0861, val_loss: 0.3117, val_acc: 0.9107\n",
      "Epoch [115], train_loss: 0.0788, val_loss: 0.3340, val_acc: 0.9040\n",
      "Epoch [116], train_loss: 0.0737, val_loss: 0.2937, val_acc: 0.9157\n",
      "Epoch [117], train_loss: 0.0648, val_loss: 0.2979, val_acc: 0.9128\n",
      "Epoch [118], train_loss: 0.0633, val_loss: 0.3011, val_acc: 0.9160\n",
      "Epoch [119], train_loss: 0.0605, val_loss: 0.3028, val_acc: 0.9143\n",
      "Epoch [120], train_loss: 0.0562, val_loss: 0.3090, val_acc: 0.9107\n",
      "Epoch [121], train_loss: 0.0492, val_loss: 0.3032, val_acc: 0.9149\n",
      "Epoch [122], train_loss: 0.0467, val_loss: 0.3000, val_acc: 0.9149\n",
      "Epoch [123], train_loss: 0.0390, val_loss: 0.2926, val_acc: 0.9185\n",
      "Epoch [124], train_loss: 0.0404, val_loss: 0.2900, val_acc: 0.9193\n",
      "Epoch [125], train_loss: 0.0361, val_loss: 0.2905, val_acc: 0.9207\n",
      "Epoch [126], train_loss: 0.0314, val_loss: 0.2936, val_acc: 0.9215\n",
      "Epoch [127], train_loss: 0.0303, val_loss: 0.2983, val_acc: 0.9187\n",
      "Epoch [128], train_loss: 0.0282, val_loss: 0.2914, val_acc: 0.9209\n",
      "Epoch [129], train_loss: 0.0271, val_loss: 0.2924, val_acc: 0.9194\n",
      "Epoch [130], train_loss: 0.0243, val_loss: 0.2910, val_acc: 0.9220\n",
      "Epoch [131], train_loss: 0.0228, val_loss: 0.2915, val_acc: 0.9224\n",
      "Epoch [132], train_loss: 0.0214, val_loss: 0.2901, val_acc: 0.9234\n",
      "Epoch [133], train_loss: 0.0217, val_loss: 0.2963, val_acc: 0.9220\n",
      "Epoch [134], train_loss: 0.0208, val_loss: 0.2955, val_acc: 0.9225\n",
      "Epoch [135], train_loss: 0.0198, val_loss: 0.2949, val_acc: 0.9224\n",
      "Epoch [136], train_loss: 0.0195, val_loss: 0.2957, val_acc: 0.9224\n",
      "Epoch [137], train_loss: 0.0189, val_loss: 0.2953, val_acc: 0.9226\n",
      "Epoch [138], train_loss: 0.0185, val_loss: 0.2951, val_acc: 0.9236\n",
      "Epoch [139], train_loss: 0.0181, val_loss: 0.2956, val_acc: 0.9224\n",
      "CPU times: user 1h 27min 52s, sys: 41min 37s, total: 2h 9min 29s\n",
      "Wall time: 2h 7min 22s\n"
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
   "execution_count": 23,
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
   "execution_count": 24,
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
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmEAAAGDCAYAAABjkcdfAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABhEElEQVR4nO3dd3zb1b3/8deRZ2zHjmMlznCmshMMWSQYaNKUltlBb3tLw4YCAXLpvN2343b39tf29kKBUjak0NJCKdBSoBCGIWRiskjiTM947yXp/P74SorkeMVYkZK8n4+HH1jD0tHXivXmnM/3c4y1FhERERE5vlyxHoCIiIjIqUghTERERCQGFMJEREREYkAhTERERCQGFMJEREREYkAhTERERCQGFMJERGLMGPNDY0y1MaYi1mMBMMZ8zxjzSKzHIXKyUwgTOckYY14xxtQZY1JiPZYThTFmsjHGGmOe63b9I8aY70X5uScCXwbmWGvHRPO5RCS+KISJnESMMZOBcwELfOw4P3fi8Xy+KFlijCk4zs85Eaix1h4+zs8rIjGmECZycrkKeAt4ALg6/AZjzARjzF+MMVXGmBpjzO1ht91gjNlhjGkyxmw3xiwIXG+NMdPC7veAMeaHge+XG2NKjDFfCyyj3W+MyTbGPBN4jrrA93lhPz/SGHO/MaYscPtTgeu3GmM+Gna/pMDy3PzuLzAwzkvCLicGnm+BMSY1MHtVY4ypN8asN8bkHsPx+znwo95uDBynPcaYWmPM08aYcQN5UGNMljHmocA4Dxhjvm2McRljzgNeAMYZY5qNMQ/08vOXGGO2BF5ToTEmP+y2/caYbwR+b3WB45s6kDEbY+YaY14I3FZpjPlm2NMmB8bcZIzZZoxZFPZzXzPGlAZue88Y86GBHAcRiaQQJnJyuQp4NPB1fjCAGGMSgGeAA8BkYDzwWOC2TwPfC/xsJs4MWs0An28MMBKYBNyI8zfl/sDliUAbcHvY/R8G0oC5wGjgV4HrHwKuCLvfRUC5tXZzD8/5B+CzYZfPB6qttZtwgmcWMAHIAVYFxjBQvwVmBMJRBGPMCuAnwL8DY3GO5WMDfNz/C4xrKrAM51hfa619EbgQKLPWZlhrr+nheecD9wE3BV7T3cDT3ZabL8c5Dh5gBvDt/sZsjBkOvAj8AxgHTANeCnvMjwXuOwJ4msDv0RgzE1gNLLbWDg887/4BHgcRCWet1Ze+9HUSfAHnAF2AO3B5J/DFwPdnAVVAYg8/9zzw+V4e0wLTwi4/APww8P1yoBNI7WNMZwB1ge/HAn4gu4f7jQOagMzA5SeAr/bymNMC900LXH4U+E7g++uAQiD/GI/d5MBrTQRuAd4KXP8I8L3A9/cCPw/7mYzA8Z7cz2MnBI7TnLDrbgJeCTuOJX38/J3AD7pd9x6wLPD9fmBV2G0XAcX9jRknyG7u5Tm/B7wYdnkO0BZ2/A8D5wFJsX7f60tfJ/KXZsJETh5XA/+01lYHLq/hyJLkBOCAtdbbw89NAIoH+ZxV1tr24AVjTJox5u7Aklsj8CowIjATNwGotdbWdX8Qa20Z8Abwb8aYETizQ4/29ITW2j3ADuCjxpg0nBmbNYGbH8YJlY8Fljx/boxJOsbX9HsgN3x5NGAczkxScBzNODOG4/t5PDeQFP6zge/7+7mgScCXA0uR9caYepxjGb4UeqjbYwdv62vM/f3ew8/UbAVSjTGJgeP/BZygdtgY89hAl2VFJJJCmMhJwBgzDGfJaZkxpiJQo/VF4HRjzOk4H9ITeymeP4SzjNWTVpzlw6DuZ+/Zbpe/DMwEllhrM4EPBIcYeJ6RgZDVkwdxliQ/DbxprS3t5X5wZEny48D2QDDAWttlrf2+tXYOUABcgrP0N2DW2k7g+8APAuMOKsMJRM4LMiYdZ3mwr3ECVOPMPk0Ku27iAH4u6BDwI2vtiLCvNGvtH8LuM6HbY5cNYMyHcJZHj5m1do219pzAY1vgZ4N5HJFTnUKYyMnhE4APZ9nojMDXbOA1nBDyNlAO/NQYkx4oYD878LO/B75ijFloHNOMMcEP7i3ASmNMgjHmApx6pr4Mx6nBqjfGjAS+G7zBWlsO/B34baCAP8kY84Gwn30KWAB8HqdGrC+PAR8BbubILBjGmA8aY04LzLw14oQffz+P1ZOHgVTggrDr/gBca4w5I1CP9WNgnbV2f18PZK31AX8EfmSMGR44tl/CWeociHuAVcaYJYHfT7ox5uJATVfQrcaYvMAx/xbw+ADG/Aww1hjzBWNMSmBsS/objDFmpjFmReDx2nF+34M5xiKnPIUwkZPD1cD91tqD1tqK4BdOMfXlODM6H8Wp5zkIlACfAbDW/gnnjMA1OLVWT+EU24MTiD4K1Ace56l+xvFrYBjO7M9bOEXf4a7ECUY7ceqKvhC8wVrbBvwZmAL8pa8nCQS6N3Fmux4Pu2kMTj1ZI86S5VqcQIUx5i5jzF39jD/4+D7gOxw5DliniP6/AmMsx5k9vCzw2BMDZzdO7OUh/wNoAfYCr+Mc6/sGOJYNwA04v8s6YA9wTbe7rQH+GXj8YuCH/Y3ZWtsEfBjn91sB7AY+OIAhpQA/xfkdV+CcYPGNgbwWEYlkrO2+miAiEhvGmO8AM6y1V/R7ZwGcFhXA5wKBS0ROICdDc0UROQkEltKux5ktExE56Wk5UkRizhhzA06h+N+tta/GejwiIseDliNFREREYkAzYSIiIiIxoBAmIiIiEgMnXGG+2+22kydPjvUwRERERPq1cePGamvtqJ5uO+FC2OTJk9mwYUOshyEiIiLSL2PMgd5u03KkiIiISAwohImIiIjEgEKYiIiISAwohImIiIjEgEKYiIiISAwohImIiIjEgEKYiIiISAwohImIiIjEgEKYiIiISAwohImIiMhJ6661xRQWV0dcV1hczV1ri2M0oiNOuG2LREREJLbuWltMfl4WBR536LrC4mqKShoABnzbXWuLSXCBz0/oNiB03+Btq5Z5Bnzf7rfl52Vx08MbuSR/LD/5ZD7f+EsRzxSVc/eVC48a36plniE7RgOhECYiIiIRwSr4PfQciMKDzaScdBJccOcre7l95XyAAd32k0/mk+CCHz+7k29ePIu547K46aGN+K3l2xfPobqlg5/9fRffvHhW6Pkj7vvwRgDuvnIh28oaer2t0+vH77f8dUsZnV4//9hagctlQq+7sLia1Ws2h8Z3PBlr7XF/0vdj0aJFVht4i4jIqaKvWafgDFFftw9U9zDSU8j53LlTKJjm5u29tdz3xj4sMH/CCN7eV8s3L57FDed6KCyu5saHNuC3sGhyNq/tqg7d9sK2Cv7jsc34/JYxWamU17eTk55MVXMHiS5Dpy8yk6QmuejyWcZmpVLe0M6IYYnUtXaRlpxAS4cPgIzURFo6vGSmJtHY3kVGSiJN7V4AkhKOfkwAAxgDy2aM5p2Sem5fOT/i+A0lY8xGa+2iHm9TCBMREYm+wYal8HBU4HFHLKcVeNwUFlf3OvMUvL2/pcDg8//mpd383792c8aEEWw5WI8FRg1PoaKxnf7iQqLLMD03g10VzfitJfzuBshKS6K+tSt0X6/fkpc9jFljhnOgppXdh5s5a2oO6SkJvLjjMGdPy2FSTjqv7qqipK6NiSPTmDc+k71VLeysaGL22OFYCzsrmpg5Zjizxwxne3kjuyqbmTN2OC6XYWtpI2dNzSE/L4s1bx/kQ7NG88L2Sj4yN5cd5U3srGjithXT+NJHZh7z73Og+gphKswXERGJgu4F4cElvG/8pQg4Eq7y87L6LB4v8Li5feV8bn10E9c/uJ6nt5RF3M/6wefz8+dNpfz93XJ+/OxOrimYHApsNz28kfy8LPLzsli9ZjP3vFrMT/++g/rWTn787E7qWzr52ztlXHrHG/zyhV10+Szr99eRmGDITE2kvKGd2WMy+eoFM7lw3hgArlo6id9fvYjstCQ+OX88qUku0lMS2FHexPjsYVw6fzwZKYl87PRxpCUnMCknjfrWLpZOGcn3PjaH4amJ3LZiGq2dPpZMHUlNSye3rZjG1rIG1u2r5bYV09hR3sQUdxqtnT5uWzGN5g4v8yeO4HBTB7etmEZJXRul9W3ctmIaVU0dzB2fSXWz8ziH6to4UNMaesw1bx/k7isX8uvL5nPP1Yt4ccfh0M8+su7gUcf+eFFNmIiISBQEQ0/4Upe1lqe3lJGRksgjbx3kSx+ZHrpt9ZrN3Lx8Kj5/5M8CLJyUzbCkBF7acRiA3MwUrr1/PdNzM9hW2hiadXonMOP1yxd2cc+re2n3+khOcOZb5k/IpmDqSH703M6Icd716l7AWZ5bOnUk28saufqsyTzw5n66/DYUVJISTCgg3V+4nye3lIZm42aPG86Pn93JpfPH8+KOSl7YUcnvrnJuu+e14ojbtpU3hn5u+LDEiFqu+wv3A7DUkxNx2w3negZ83+63VTV38ExR+VG/n0vyx/Klj8xkqSfnqN/T8aLlSBERkSgpLK7m5kc2MTknjW1ljRigy+987k51p1PT0smdVyzgrKk5/PDZ7dz7+n7mjMukvL6NOy5fQIHHjbWWK36/jjeKa1g+YxRv769lZu5wikrr8flh2ugMLpg3hkffOsAnzhjPExtLmOROY2up83wWSDCQEKi5ykxNpLHdy/lzc7notLE89245z2+r5BNnjOPV3dV91oQFA1H3MwzDA+SBmpYB3RY8ASDaZ0euWuaJSg3dQKkmTEREpA/RaLlw7dmTueNfe/jNv/YAMC4rldMnjODV3VVMHJnGjvImDJCYYEhOcNHS6Qs9fmqii69fNItrCqbwxce38OTmUpbPGMVST05oWROcZcGH3joAEAo2wZmnT8wfz8vvHeYTZ4znhe0VlNa3c+50N9vKGrliyUQeWXeQm5dP5c5X9nLFkonc89o+vvSR6dxwrqffsyMHe3zCbzve7SBipa8QpuVIERGJW8dr1iJ8+a+opOF9t1xYuWQCF//mdfYcbgbg4tPG8uruKl7fU809Vy2iwOPmmaIyvvLHdzAGWjp9zJ84gn3VLSyfMZq/binle09v55l3ytlwoI78vEyKShu4cdnU0JgvyR/Lf14wi9rWztByW2FxNXe+spdvXjwLnx8+vSgvFNgunT+OpzaX9bi8d8O5ntCy3NxxWRHHtrclugKPe8hvO9UohImISNzqXlcVrZ5OBR43371kDtfev55pozPYXtbINy6cxdIpOdz3xj7aO3384e1DoaW8VR+YSoHHzd/eKaPL5+eJjSWsfa+K8sZ2LPDoukMMS3Jqsb7VbQkv6JL8cZQ3tPHLf+4+KiB97Iyx3PTQRjYcqCMvexiHao8sT961tjg06wXwk0/m89HTx4VmnsJrm4IF58EA+c2LM7nzlb3MHZeFz08orAWPQTCEKiQdH1qOFBGRuBasqzpvzmhe3lnVawH1scyadb/vjvJGLvvdWzS0dYXuY8DpR9XpIzXRxcScNHZVNoduT0100e51EkxSgqHLZ5k1ZjgXzBvDtrJGXtheyaXzx/GrzxwJjOHjCQ+U3WffgoHzf/7xHpsP1Q+6jcLxrn+So6lFhYiIHFdDuV/fWVNzSEl08eeNpXx6YV6vszTBWbPg8/bVAiK8XcTb+2r55G/foKGti/TkBK47ezLpyQmMzx5GS6eP8+eO4a4rFobaH4wYlkT++CzavX7One7mx5fOY3hKEretmMbhpg7SUxLYeKCO21ZMY+2u6ojnLfC4Q+GnqKQhFLhWLfNww7meUCALOlDb+r7aKKxa5jnqeIWPQWJLy5EiIjLkgoHol58+nSVTc9h8qC4063OsszP/++JuDjd1APDIWwdYNnNUxNY6wcc5a2oOH56dy+W/X8e4rGG0dnpDS3hAZBsCC36/5clNpfxx/SH8FtJTErkn0FZh7IjUiLYKhcXVR7VVCN625VB9jy0Xwuusepq96+m1BuulujdojWUbBYkehTARERlyBR43t392Plfd9zYpiS4SE1zceUXPgaivOq/Xdlfxm3/tZvyIVBJcLrKGJYbuG14vlpTg4ttPvst7lc0MS3JRWt9GRnICLe2+iPFc98B6khNcNLV7Izq6L56czRc/PCM0nvDi9tQkV6+F7+G3AUNWZxU+S/Z+Hkfim0KYiMgp5njVCXX6/Hj9Fm+nj6QEP5WN7cCRQHHDQxuYNDKdisb2Xmd41qw7iN/Cdz86l62lDdz+8h7+77PzQ2O9feV8bnxoI80dXoyBC+bmsm5fLefNzuVvReXc9MgG7rxiIdNHZ/Cj53bQ3uWnvcvPvPGZLJmSw582HOLKsybxh7cPhZ6zewACei18D7+tt2W+wZwNOFSPI/FNIUxE5BRzPM44tNbyg2e24zLw6UUTeHz9Ib70+Du4MCyZmsO9r+2jpcPH9vJGLl8yMSJcBEPigonZbD5Yz/yJI8hISaSty4ffQnlDeyikLJyUjQn83EdPG8frxdWhJci544v56d93surhjRgDfgspiS5uOHcqD765nwM1raFlxLOnuUPH4FgDkMKRDJZCmIjIKabA4+aX/346V977NhfOG0Nhcc2Q1xrd+UoxxVUt3HDuFL518RxOz8vim09u5QuPb8FlwGedMwp9fssTG0u4OH9s6PmDIfGCublUNLZzwwemsPoPTkBat6+WJzeX8rlznX5ZP3xmB00dXj5xxjie31YZsQ3QqmUeEgz89O878VnnbMb7rl1MgcdNTUvkVjZa7pNY0NmRIiIxNJRnER6L4alJ+PyWZ4rK+cyiI2ccDsV4rLU89NYB3BnJfPWCWQCsXDKJH186j+REgy8wI/XgdWdy0WljSTBw66ObQs9b4HHzs3/L5w9vH2JSThp3vFwcComfOGM828oa2V3ZxEs7KnnkrQPMHZfJry+bz73XLOLOV/ZGjH/u+CyyhiVztieHpMQjH3k/+WQ+d1+5MOJMRJ01KMebQpiInLJiFYDC9dVWoS99jX0gr+uZorLQ9/cX7g/d/1jGE/48we8Li6v5zyfeoaKhnU/OH8+9r+8L3X+yO530lCTO9uSQHAhEK5dMpLXLz2WLJ0YEojeLa7DAgZpWrghbrqxr7cRl4Kktpdz3xj4s8NkzJ3DX2uKI2azwsd9++XwevWEpd1+5MOK1KXRJrCmEicgpa7ABaCgVeNz8v0/nc+W9b1Pwk5e48aENR3U87ykU9jX2/l5XYXE1a9YdJD05gU+cMQ6vz3LzI85MVIHHzc8/lc81963nit+vi6gd66vXVvD7mx7eyMYD9YwensKfNpZEPGfwscIDERamuNNZv782FIj2HG7igcJ9pCS6juqRdZYnhwSXYc26g2w+WM+SKSP55Qu7Q8/TWx+u4G3d+3CJxJJqwkTklBX8UL7l0U184ozxPP1OWUz6MPn84PNbyhraMUB5fRsQGVx6G/vnHtxAenICnT4b0QLi9pXzufXRTVySP5Zn362IeF1FJQ3kZQ9j9PBU8rLTSHAZpo3OoKikgUk56Xz/6W10+vy8vqeaawomH1WrFXys+tZOOr1+/rj+EM+8U05ThxeApnYv6ckJ3HP1oojn7C0QffbMCfz4uZ3sqmxi+ugMvvj4FqyF/73sDC6YN/aoHlk3nDuV375SjAF2VjRFvO5wOsNQ4p1mwkTklFbgcZOZmsQDhfsjlr36M5RLmY+vd9ojXDhvDBb48p+K+NyDG/ptzlngcTMyPZmq5k6aO7zsqmwK3TZ6eCpYePitg5w/NzfiMT53zhRK6tqYMy6Tgmk5uIxh44E6XAYu/t/XOFTXRmqSC5eBP7x9MGL57vaV81n18EY+8POXueXRzXR4/YzMSKapw8tp47NYMHEEANeePTniOfvq3N7c4SMxMLv10o7DvFvayHmzc9lf0xrxvMEZrFs/OI3kBIMFrj5rkkKVnLAUwkTkhDOUAeiFbRUcrG0lNcl1TFvDDNVSZuGeal7aWcmCiSO484qF3LFyPgZ4cUcliydn9xkwCvdUU1rXxsSRafj9lu89vZ1fv7CL+9/Yx/m/WktdYB/Ev24pi3hd+6pb6PD6mTM201lmvHIBLgM/fm5naOue+65ZzNUFk+ny+UNLlQCTc9Lp9Pk5WNvK4knZ/OJT+fj8cNuKaeyvaWH34WZuWzGNNW8fGvCxXDp1JC5jeHz9QX7w7HbGjUhlw4HaiGMZvsz4Tkk9aSmJrP6gZ9Db+YjEA4UwETnhDFkAKq7mC49vAaDD6+d/Lzsj4nH7EuzAftPDG/nF8zsHvaXMy+8dxm/h0gV5AGSnJ5ORmkhmaiLPb6tkzboDvY795kc3YYEbPzCVu65ciMvAr1/azff/th2fhc9/aBpT3enMzB0e8bq2lzcCMGdcJgDLZozm0gXjAVg0KTu0jHjrB6eRmpTA7LHDKSppoL3Lx5W/X0d7l58rlk5kZ2UT339mO7evnM9ST05obEs9Ody+cv4xHcuvXjCTti4/B2paaWqP3G6o++tevWYzv718AV85f9YxPY9IvFEIE5ETTnB56voHNnDh/77Kdfev5+blU4/qAN/fzFhRSQOzxjpBxFqYOWb4MRVul9S10dTu5faXi49pKTPc2KxhACyfMSoUMO6+ciH/+spy0pJcfOvJrfxj65F+VsHXVVTSwLUFkwE4bXwW588dw31XL2ZsVioAl84fxxc/PJMlU0ey53Azv7nsyOvaXtZIcoKLaaMzQo/58s4qblsxjeLqltBzuTNSOGPCCN7aW8u5093c8uhGiqtb+NTCPPKy07gkf2zEsbz7yoWhtg/HWgR//TlTcGckA3BtweRej6WK7eVkohAmIiekAo+bBBfsKG+i3evn5/94j4ff3A9Ezoz1tXR51VmT2FrawLhAcKlu6gwte/W35Fnb0sn3n9kGwIzcjEEvi72yq4qpo9KZMDItImC4M1L49iVzsMB/PbWVTq8/4nWtWuahy+8nwWWYOWY4AMlJLjq8fm5bMY21u5x2EUun5tDU4WVEWlJoOW97eSMzxmSQlOCKKP7/0kdmHjWzdO3ZUzDAVfeu4187qyjw5PCvnYfJz8uK6LUVrPkKXzY8lhYQb+6twee3/S4x9lVbJnKiUQgTkbjUXwh6eWclzR0+zpgwgmFJCSQYw3/9dRu3PLopYmmwr6XLV3dV0eH1c9mZEwGoaekIPVd/S55ffHwzLR0+po9Op7q5k/+77NiXxdq7fKzbW8OyGaOAowPGyiWTWP1BD1XNnXz2nreOWvLcWtrI9NEZpCYl9BqmEl3Opj5v7a0BnEaq28samROYAexvZunDc3L5zOIJ1LR0MWHkMHaWNx11//cbgIJjv0NLjHKKUQgTkbjQVw8qiAxBhcXVfP6xLQB87twp3HvNIoYlJ5CRksBz75Zz+ZkTI0LCrz5zOlfd+zafufvNiCDz/LZKstOSuHDeGACqm4+EsGAYufXRTdz08Aauf+DIkue6vTWs3VXNWVNzmDkmk9qWTrLSknpcFusrTL65t4YOr5/lM0f3ely+cv4sprrT2XigLmKPRWstW0sbmDfeCYW9halDdW1MzkkLhbDKxg5qWjpDIWwgM0vf+9hcls0YxaHaNq5YOrhl175oiVFOVQphIieweOj43pdjGV/3mScs+P2WpzaX8ZPndkSEp6KSBq49ewrgNPos8Li55YMeOr0WiOwAD7C9rAmv37JuXy0fmO4smXV6/by4o5IPzc4lN7AcWdPcGTGmAo+bueOyeH5bJdbCr1/YzSvvHebbT23FnZHMzopGLjrNCXCv7q7qcVaorxm1te9VkZrkYsmUkb0ew8Liaiob2wF4MOx1VTS2U9PSybxx/YeppVNzeHtfLT6/ZXu5E2zmjh/4SQybDtbxbmnDUY1Th4qWGOVUpRAmMRfvQSKexUPH974cy/iCsx83PbyRFf/vFS7//TpaOn20dfm4+9W9EYXvq5Z5QtveTM5Jp7C4mjtf2ct91yxmqjud7LQkVj/qPO/hxnb+98VdJCUYstOS+Os7ZTy/rYK39tbQ1O7l/LljGJ6SSHKCi6qwmbDgeNfvryXBBV1+S0unj2vvX8/uw820d/m54/IFXHTaOGaPzeS1XT0Hk/AZtZX3vMUtj24Khcm1u6pYOjWH1KSEHn82eLz+59P5AJw/d0zoeG4tdc5wPG0Av+ulU3NobPeyo7yR7WXOz80K1JH1p7+aMREZPIWwU0C8h5x4DxLxLPwD/voH1g+6TUK0BMd38yObuPz3b0Us6QV1fy+2dHjZW9XCjDHDufbsybgMeEalHzUDs7eqhdzMFNJTEkPLWedMd7N6xTQO1bVxzdmTKSpp4Mt/escp3P9UPp+YPx4DrF6ziQcL95OWnEBSguHuV/fizkimuqkzYlyr12zm9LwsJuek8/B1Z5KRkkh6ihOYrgtrRvqB6W42HKiltdPb63GYkJ1GYbET/N7YXc2Bmhb2VbeEzors6d9j8HVddNo48vOy2FPVHFqm21ragDEwO7Cs2JclU52Ztrf21rC9vJFJOWkMT03q9+fCx6ClQpGhpxB2Coj3kFPgcfN/lzntBv77b9viLkjEuwKPm5ljhvPSzsN8ZE5u3B23Ao+b8SNSeWNPDV6f5Vcv7Oa13VXctbaYe14rDr0Xi0rqufret7EWbjx3CmX1bTyxsYTFk7Px+e1RMzD7a1qY4k4HIpezKhrayc1M4Z/bKzhzykhe211NgSeHysYOPjwnl9SkBLp8lpffO8xp4zP50h/fIT8vi5yMlIjC/GD46PJbxmSlUjDNzefPmxZqTBoeCs+dPooun2Xd3toej8GTm0soKm3AMyoday13vFLMv91ZCEDmsKRe/z2Gv64Vs0az5VA9M3OHs2qZh21lDXhGZZCW3P/uc2OzhjEpJ411+2rZXtbI3HH9B7eexhCkpUKRoaEQdgoI32Pu638pisuQk56aSFuXj/veOLatY04k0ZqRLCyuZsOBOgCefqcs7paJCourea+yGXdGMhZo7fRx9X1v81xRGT9+dic3L5tKbmYqn7n7Tbr8lts+NI1vXjwn1INqwsh0DtS2cnreiIgZmH3VLUxxZxz1fGdMHEFzu5etpY3c+NBGRgxLYkd5I/l5WRR43Nxz5SISEwx+C1vLGsNaQiRHFOYHw0dlQzu5mamhJc97r1l01LLcosnZpCS6eHV3VY+v/2tPvEtSgmHNDUt56LolpCS6qG7uZERaEj98dseA/j2umDUaa2HtLuc5tpY2hurBBmLJlJG8WVzD/prWUFG+iMSWQtgp4vS8EbR2+njs7UNRCTnvN2D8aYOzd97sscNP2m1IojEjWVhcza2PbiLBOG0I5k8YEVf1OsHxGeDfFuTxUGBJLyMlkaLSRizOVjmX/OY12rr83Lzcwxc/PBMg1IPKby3Wwq7KptAMTENrF7UtnUxxpx31nM42PE73+OrmDrr8/oju62dPd3PNWZMBuPqsI0uK7oyUowrz/X7L4aYOxmSm9rksl5qUwJKpOby2++jj/sL2Sjp9fm441wmb50x3c/+1i1k0KZv61q4B/3ucNy4Ld0YK/9p5mKqmDioa20NnRvbnrrXF5GSk0BzYYHvOuMy4KkkQOVUphJ0intpcCkBuZkpUQk4wYLyxu5r2Lh/f+EsRNz28MSJg9PZHv7C4mj8GQlhKYkLEDEO817Mdi+CH9o0PbeRnfx/8Njfhikoa+MKHZ9Dh9TM8JZGDda1xVa9TVNLA1y6YhddvA5tFO0t6XT7L1WdNIj05gbzsYbR1+bls8QS+dsGsiJ8v8Lj54nkzANhZcWRz6n01Tlf3nmbCwFke/PgZzjY81xVMOaoG7S+bS7ltxTQeW39kf8OcQAiz1obuW93SgTewHNnfstwHprvZc7iZsvq2iPftvuoWsoYlMX9idsT7dm91yzGdbehyGT44cxSv7qrinUP1AAMOYfl5Wfxh3cHQ5bZOX1yVJIicqhTCTgGFxdX8+O87AGjp8HH7Z4f+7KbQNjIPrWfOd/7BnzaU4PNbsEfG0Nsf/XcO1YfODiuuauasqTmhIBHv9WzHKjstmeYOL3euHfw2N+FWLfPQ1ukD4PKlkzhU28a0URlxU6+zapmHpATnz8zcwOxLcEnv+x+fxxc+PJ2SujYunT+ef26v7PE9OX7EMDJSEtkZ2O8QYF91M0CoJqy7wuJq1u5ytuF59O2DR71/ejrTz52RTKfPT2P7keL6ygZneTI3M7XP13nX2uJQofvru6tDPc6uf2A9r7xXxUWnjeFrfy4K9Tgb7NmGK2aNprHdy4OBnQHmDHA5ssDj5rdXOJt0D0ty8V9/3RZ3JQkipyKFsFNAUUkD583OBaC5w8tkd3pUZksKPG6y05LxW3AZp/bninvX8ZFfreWmhzdG/NEPn8368Jxcmtq9zBmbSVO7l6qmjtAMQzDc3fDgBr74+JaI2aMTcZbsycCM5FmenCGbkVy3twbPqHTOn+v8jjcdrHvfjzmUtpc3kprkYoo7I2JJLxjIvnnxrNCejT2FEVdgW54d4TNhVS24DEwcefRyZF8hp68lRXdGChDZsLUi0J9rTD8hLD8vi5/9YycjhiXx7LvloRnhf713mPTkBP6xtSKix9lgzzbcVdlEggte213N5Jw0MlOTBvyeL/C4+cCMUbR1+U/aukuRE41C2CkgOFsSKBuKqK0ZSm/srqa8oZ388VmkpyRy8zIPORnJ7KpspqXDS32LU2/TfTZr/X4nNHz2zAkA7KlqjnjcJVNy6PT6eXJzacSHR7zNkvUXCguLq3mwcD8Ac8dmDngGpK/H9fktG/bXsWRqDnPHZZGc6GLjgfcfwoYy4G4ra2DmmEwSXCZiSS8YRm441xMRuHsKI7PGDGdneWNoqXBfTSt52WmhXmHh+go5fS0pBkNYeF1YKIRl9R3Cgs/R2uVj7a4q7nilGL+1jExLpqXTx5VLJ0X0OBvs2YaLp4zE4PxDnjs+65je88EQGq2GqyJy7BTCThF7qppZPMnpFbS7srmfex+7wuJqblmzCYDPLpnIHZcv4JF1B2jv8nPONDd+C7es2cxND284qhZq/b5a3BnJnDfHmckpPhw5vic3ldDlt6QkuiI+PMJ7UK0KLP3014MqmkKhcE81Da1dR31AFpU0MCnHmbmpbe0c8AxIX2Fze1kjTR1elkwZSXKii9PzsoYkhA1VwA3uU9hTS4RjCSOzxmbS2O6lvMEJRfuqm3tdihxsyMnJSAYiZ8IqG9pJcJlQQOtLgcfNlUudPShXLpnIA9eeieXodhbvR4HHzWWLnf9ZqWvpHHBdoRquisQnhbBTQKfXz4GaVhZNzsadkcLuw039/9AxKipp4PpznG1kwptHXpI/lkc+t4RffDofAzy/rZIVs0ZFfGisP1DLokkjGZOZSnpyAsVVLaHbCour+e7ftgHQ4fXz3UvmRHx4FHjc5KQn849tFRjg1y/u5vXdQzszNtBZodDS6cMbmP+Df3LLI5siPiBvPHcqpXVtANQGZgUHEg7Cw+aX/xi5JLtun7Mf4JIpOQAsmJTN1tJG2rt87+s1B59z1cMbueXRjYM+iaC0vo3GwFLz+zE70N19Z4UzG7avqqXXEDZYR2bCIpcjR2WkkBDYBLsvhcXVPLm5jNtWTONv75SFOuMPdei5+YPTyExNpLC4ZsDLimq4KhKfFMLiWF8f/seyXHSgpgWf3zJtdAYzcjPYFYWZsFXLPFgLxsCMXKf25+4rF/KTTzrbrYwbMYz0lEQSXPDXLUd6WVU0tHOots1ZZjEGz+gM9oTNhBWVNHDutCMfMsmJrogPj8I91eyrbmFMZgodPktLh4+r7lvHt598d8j6oR3r1jsjhjl1cYsmj4x47v01LbQEiuiDIWygCjxuUhNd/HlTKRfNGxN63Lf21jIpJy20XLZwYjadPj/byt7/h+uCidn4reW5dyu4dP64QR3H4BY5Ay0g782MQAjbUd5EVVMHLZ2+IQ9h2WlJGANVYcuRlY3toX0l+9J9pinY4yxoKEPPgZoWEhNcxzTDpoarIvFJISyO9fXhH7ztxR2VdHh9fQaDYKiZPno40wMhJ/w0/KGys6KRyTnppCUnRvzRD47td1ct5JblTnuCmx/ZFNqXD2Dx5GwApo2KDGGrlnmobu5k3vhMEl2GrWUNoQ+PwuJqbn50Exa4dcV0HrrW6UFlLTyy7uCgi4+7B9wCj5uP5o/lqnvf5sfP7ugz3D1bVEZpvTPb9eruqojH2RoIJDNzhx9zCHv4zf1UNjkzNI+tP8Qbu6vx+y3r99dGbP68YJJzHHtakjzWOq//fmYbzR1OaHx0kMtp28oacRmYPeb9hbDM1CTysoexs6KJfdXB9hRDG8ISE1yMTIts2FrR0M6YzP6XIrvPNAV7nIWHrqEIPVpWFDm5KITFkZ4+/JfNcHP5Peu48aHIWqoCj5tffDqfGx/awPL/eYXVj/YeDIKhxjM6nem5w2nu8FIWqK05lvFA3x/aO8obe9wUOPwD6qqzJpOc6GLRpGyKShpYv7+WtOSE0HKVZ3QGFY3toaaSXp+fbWWNLJo0kum5w0ObFgcfN1iDs2DiCM6e7vSgApg2OmPAswTdX2ewvcA3/lLEodpWPvO7N3nwzQN4/Zbfvba313BXWFzNfz5RBMCcsZlkpiZGfEBuK2sgKcGwZOrIYwphhcXV/ODZHaQmufjOJXPw+i03PLyBx9cfoqGtK7QUCc6S2uSctB5D2LHM6L2wrZI/rDtEfl4WF8wdQ4Ix3PropmP+sN9e3sgUdzrDknveoPpYzBqTyc7yxqiFMAg2bI1cjuzvzEg4fjNNWlYUObkohMWR8A/Jlg4vV9+3jic3lwHwz+2VfHL++Ig/9Bv21+G3OGckTsjqddZnT1Uz40cMIy05kRm5TkjaVdl/XdixfGi3dHg5UNvKrB5mPMI/oEYNT+HfFozn9T3VfHphHuv317FgYjaJgV5SnlFO881gcX5xVQttXT7y87KYNy6TraUNoVm8Vcs8NLV7SUtOYGbu8FDLg1ljhzNiWNKAZwnCX+fhxna2lTbQ6fXzxMYSlv3Py6zbW8s509wkuAy5w3tvdltU0kBuZiqn52Xx74vyqG7u5LuXzAl9QG4va2RG7nByM1Np7fQNuG7rzeIafH4/ly2eyLVnT2biyDS8Pj9Pv+O0u1gydWREOF4wKZuNB+qPmu0M3+z7+/3s0fn71/digZ9+Mp8bl02lpdPHJfnjjvnDfntZI3PGDc3ZqrPHDmdvdQs7K5pITnAxbsSwIXnccDkZyVQHliNbO700tXsHtBx5vGhZUeTkEtUQZoy5wBjznjFmjzHm6z3cPtEY87IxZrMxpsgYc1E0xxPvwouhC376Emt3VXO2J4fhqYkY4KG3DkR04b771WKSE13kZQ9j7XtVPFtU1uPj7q5sxjPaCTfTA//dM4C6sOB4bnl0E6vXbOrzQ3tXZRPWOh+U/clISaTD6+eOl4vZWdHIosnZoRAxLTC+4kCbiqKSesAJSvPGZ1HT0kll45GZik0H68nPyyIxwRWaJZiZO5yKxvYBzxKE76155o9f4kfP7aTD68cAfgsXzM1le3kjn1mcR2VTB/918ewew92H5+Syr7qFj50xnqUeZ3aqy28D9XKWraUNzB2XSU66cxbeQGfDkhNc+Pxw5VmTMMbwhfOm0+mzvLm3lvEjhnGwtjUUju9aW0x2WhLVzR0cqnWWRcMDWoHHzcSRadz/xn7Omz069LsMnw2saGhny6F6zpnm5tXdVSyYmM3iydn8a+fh0MkXA1Hf2klpfdsxbRbdl1ljMvH5LS9sr2RSTtqAiuWPVfhMWEXDwHqEiYgMVtRCmDEmAbgDuBCYA3zWGDOn292+DfzRWjsfuAz4bbTGEyvdl7ruWlvMPa8VRyzphX9Itnf5aO300dDm5dzpbnZUNHHXlQtZtdxDp9fPTQ9vpHBPNV94fDM+P/z6M6fz8PVLMAY+/9iWiOdyZob2sLe6mWmBGabs9GTcGSkDmgkD50N7qjudZ4rK+fTCvF5n24JbyswewFlwH5w1mqQEw31v7MNaSE9OCIWISTlpJLpMaAl1a2kD6ckJTHFnMG+889jvljqhqq3Tx47yRhZMdOqggrMEuVmpHG7swFo74FmCAo+btMCS2aXzx3PHyvlkpCZx24pprN1Vzc3Lp7L6g9MB54y/nsLdX7eUYQx8NH8sM0YPJzstibf2Omcvlje0U9faxbzxWWQfQwjr8vl5ZN0Bzp3uDs0SfnJBHp9Z5LQpGB5Y8gyG4/y8LJ7YWALAxoO1R81evryzMjTuJzaW8ML2SiByNvDXL+7C57e8W9oQ+rmJI9MorW/juXfLQ2Prr/3H9kCH+6HaLHpWIOCX1rdFZSkSImfCBtqoVURksKI5E3YmsMdau9da2wk8Bny8230sEPwLnQX0PJVzAuu+pJfggh8/u5ME15FAFvyQ/OP6Q1z/wAZ81nL9OVPYsL8u1PcqLSkBd3oymamJ/GnjId451MB5s0ZzsNb5QPrUwjy8fsvdYY1BV6/ZzLisYbR3+Zmee2SPvRm5Gew63P9MWPBx3gl8aIfvs9fdjvJGMlISGT+AJaICj5uvX+jsEWiA375SHAoRSQkuJuWkHZkJK21g7vgsElyG2WMzcRknmIETxrx+GwphQWMyU+n0+Y+p7urPGw9R3dzJOdPcvLijkq//5d1Q8fO91yzizlf2sr+mhcWTs3n6nbKjwp21lqe3lFLgyWF0Zioul2HJlJxQCNsWKMoPnwmrCRtfeFgPfl9YXM1X/vQOlY0dLJ0yMiLw/PDSecwbn8nOiqaIGrUCj5s7Vi7EAPe+vi8ioAXfExb4+oWzsBZWPbKBwj3VR7ademADj60/RKLLcOcVRza9/uT8PFwGfvnPXVhrB9T+Y6jOjAyanJNOSqA5a7RCmDuwyXV7l4/KQAiLp+VIETm5RDOEjQcOhV0uCVwX7nvAFcaYEuA54D96eiBjzI3GmA3GmA1VVVXRGGvUHOnxtJFP3P4Gv3h+F/MnjuAXz+/iyU0l/OjZnZw/N5e39tby1T8XYYGvfGQG/3XJnNCHf2FxNQsnZ9Pu9VNa384zReWMG5HKxoN1oQ/BH37iNPKyU3ltdzU//fuRM/gyhzn72QWX+QBm5A5nT2VTv2dIFhZXc+ujm0L3u3m5p9caq53lTcwcMxzXAJeIrjt7CmMyU7AQ0U0cnLqwPYeb6fL52V7WSH5gk+K05EQ8ozJC7ReC2/PMnzgi4rGDMxflAzj5IPg6v/2U04vs55/K77O9wMdOH8euymbeC8z8BQNTUUkD+2ta+fjp40MzREunjqSkro1Dta1sLW3AGGemcGQghNWFhbBgWH+uqIys1CRufGgjNz20kffKmxg9PIV7X98fEXjW76+lrL69xzYF50x3M2VUOltLG/nUgiN1hEUlDYzPHsZUdzo3fWAqly+ZiM8Pv/jne1Q1dfDrF3bRFqhTu/KsyN/J2dPdfO6cKRyobeXzj23pc2k6eEy2lzWSm5mCOyPlfTfNvWttMev21TAzcOLHFHd6VBrxusMatlYE9o3UTJiIREusC/M/Czxgrc0DLgIeNsYcNSZr7e+stYustYtGjRp13Af5fhV43ORlp7ElUN+0v6YVl4H3AnVZf3j7EL95aTcJLsM3LpzFrYFlr/AP/wKPm99dtZCkBEOXz9LU7uWOy4/MVCQnuvj1Z+bjt3DX2iNn8AWX9YLLkeAEspZOX79nSBaVNHDLcg/+QFYbmZ7c4zKctZYdFY0DqgcLenNvDR1ePzeeO/WoEDFtdAYHalrZUd5Ih9fPaWHhY974rNAZkpsO1DE5J42cbt3Mgz2zgjMZ/XH2DUxm0aRsxo0Y1md7gQtPG0uCy4SK4oPh6a5XiklOcJGdnhyaIQrWha3bV8u2skamup32HSN7mAkL/q5X/2Ez33jyXZo7vDR1eNlZ2URzh5fbL4/cd7OvNgWFxdWhuqaH3jxSR3je7NHsqmzmM4snYIzhB5+Yx4KJ2Ww6WM85P/sXb++vIyXRxeoPevjzptKjwvaXz59J1rBEnn6njEvyx/a6NB08Juv31zJnbOaQNM0NPubINOd/Klo6vVHZoip866LKxnaGpySSnpI4pM8hIhIUzRBWCkwIu5wXuC7c9cAfAay1bwKpwEm3q2xhcTW7KpvISU929lRcPpVhyYms/uA0RgxL4gPTnZd8y3IPN3WrXwpf9irwuLnubKcw+tqCyUd9CHb6/CQlGFISXTwcKOLfc7iZnPTkUB0SMOAzJFct89Da6Q/tOVnR0N5jjVVZQztN7d4ez4zs7XisXrOZOy5fwDcvnn1UiPCMysDrtzxT5NQfnTb+yAft3HGZVDS2U9XUwaaD9UctRcKREFYRFsL6ardx3uzRHKpr4+KwGbDeasncGSkUeHL42zvloZqzr184i39sqyBv5DC+9uei0AxReF3Y9rIG5gbOEsxMTSLBZaht6Yh47DMnjwwF3uUzRnFmoHfaVd1mpfpqUxA8tndesZDLFk+gy+8P9WR77G1nmfHfFuYBYIzh/msWM2JYEh1ePymJLu6/djFfOX9Wj2eWOm0vDAkuw8NvHuCFbRWhYxte51jgcXP9OZM5VNdGU7t3SJrmBl/j2/vrcBn4v3/tGZJGvN3lhG3iXdEwsEatIiKDFc0Qth6YboyZYoxJxim8f7rbfQ4CHwIwxszGCWEn1npjP4IfipNz0pg9NpObl0/lx8/u5OblU/nK+TO5dYWH13ZXc+n88f02xCwsruZPG0t6XIIKPs9/XTKHDq+fC+aNYfWazWw6WBs6MzIoeIbk7gEU56/fX8usMc4SWkUvM0s7ArU/A50J66/XUXDp9K9bShmeksjknCP1P/MCgewfW8upbu44aikSYFRGCi7j7PsX1Fe7jWeKyjEGLjpt7FGP1d1da4uZPTaTg7WtvFPSwB/ePsi3nnyXBJdhb1VLRH1WsC7sXzsPU9bQHjqxwOUyZKclU9vSFfHYz211Qs2KWaPZeLCOHRVN3LZiGn/cUBLxu+6rTUH4sf3SR2aQkpjAjNwMNh2s48+bSvjwnNyIfRC3lTfgMoazPTkRm2F3/50cCXcL+NZFs7HATY9sZO2uwxF1jl0+Pz98dju/+OcuADYcqBt009zunP8JmYzfwlXdlrCHSsRy5AB7hImIDFbU5tmttV5jzGrgeSABuM9au80Y89/ABmvt08CXgXuMMV/EKdK/xkajlXsMBT8Uv/GXd8nJSMbnh29ePAufP3j24t7Q5U8vyut11iB8CarA42apJyficviH74s7DvPPbZX8v0/ns+qRTSwOa+YJ4WdI9l2c7/X52XSwjk8vzOPt/XURoSbczopAJ/gBzoT1NMMUbEALMHWUE7oqGzs4a2pORJ1ZsN3Bw28dAGB+DzNhiQku3BkpEaExGCquvX89o4c7xdfB5dzv/nUbiyePJHcAH7j5eVnc+ugmEl2GGx7aQFVTBy4XpCUlct3Zk3lk3UGWenIo8Li5a20xuZkpoRME5o7LCtWPjUxPipgJKyyu5ltPvgvAmVNGhnYSWOrJOep33ZfwYzt6eCoLJ2Xz2u5q5ozNpK61i8vOnBgaQzCYBpc6u7/Hwn8n4e+vAo+b2pYObn+5mC88toX2Lj/LZozil//czR3/Kqa+rYtxI1JpavdybUHkMXk/CourWfP2odD/hAzFY3bnDs2EOcuRnigEPRGRoKjWhFlrn7PWzrDWeqy1Pwpc951AAMNau91ae7a19nRr7RnW2n9GczyxEJy1qGnuJCc9hVXLPNxwridi1iJ4ua++Vv3NHoXPjnz+Q9Ooaenkrb21dHj9oZkvOLIsNyM3g92BerHeCpy3lTXS2ulj8ZSRjM1K7bXQfUdFExNHppExBLUzd60t5t3ShtAMRH5eVsT4hqcmMcWdzq7KZtKSE3rs0A/OkmRFY+RyX4HHTUqii0N1bXj9lo4uH+9VNLH7cDMfzR87oELvAo+bOy5fgDFQ3dRBcoKL9OREfnfVwqPqs/Lzsnhqy5EV+NaOI3VMI9OTI87eLCpp4OqzJgHO49595cJQXdr76Yp+/dlTMAYefPMA40cMI9GY0BiOpft699m3r5w/i7M9OdS1dtHW5eOVXVW0dfmob+ti6dSRtHf5ufvKo4/JYB2v7XpSkxLISEnkcGM7h5s6GJPV/5ZFIiKDFevC/FNCe5eP5g4vORnJEdcfS/frY7nv+v11zBuXyT2v7QWc5b1gwAjOfgxPTWRPZROFe3ovmj6yr6MzS9S90D0Y6HaGbVf0fs9YC44vuCyUkuSKGF9wdil438QEV4/PmZuZetTM3SvvHaax3cuYzBSa2r187qGN/OalXbiMMwMy0ELvAo+ba8+egsXZ8/LuKxf2GGQKPG5+G2gXMTwlka8F2l4UeNzkpKdEhLBVyzxkpDpF51/48IzQjFN4PeBguqIvnzWaVR9wfi4vexj/8diRma730329sLiaHRVN3LLcQ3ZaEl+/cBbZaUn8x4ppvHOoIdRapfsxGazjuV2POyOZnRVN+PxWy5EiElUKYcdB8Cw4d7cQFi35eVkcqG0NFXk3tHWFAkbww+u13dW0dPq46ZGNvS5zrd9fy8SRaeRmpjImM5Walk46vL6I57n10U3srWph1hCdBRccX3CW7sHCAxHjy8/L4p1DzgfvgonZvT7nmMzUiOXIwuJqPv/YFgC+cdFsbl3uwee3PPtuBeNGDONbT20dcKF3YXE1TwRq83ZUHF1XFx5kzp7uZunUHJo6vBG1UdnpSUf1MSurbyMzNXFIZhTD/ef5Mzl/7hjW7asdkvqs8Fmpr14wi1s+6OFnf9/JLR/08OWwvmrd90F9P1vrHM/tenIyUkKNZgeyRC0iMlgKYcdBsF1ATvrxWdoo8Li5+8qFJLoMyQmG7/x121GzCCvPdDa+bunwhnpeBQU77W/YX8fiySMBGBs4S+xw2BJfgcfNVz4yE4tT5D8UZ8EFH/fjZ4wDjj4zsMDj5ivnzwCgpK6t1+cck5VKQ1sXbZ1OaCwqaeDagsmAc/blf14wi5uXe0KPM9BwcqzLYoXF1bxX2XTUyRQj01Oob+vC5z9SAllW3x6V/RDf2lfD+v21PZ7QMRjdZ6XC6xzhxN9U2p2RTFO7s4H8GJ0dKSJRpBB2HNQEtkHpvhwZTQUeN1efNYlOnz0qYBQWV/OXzaVcd/ZkrIXv/207976+N3Tb6jWbGZWRQk1LJ4sDbRJye2j7AJCW4mz18/etFUN2FlxhcTUv7jjMbSum9XjG6HVnT+HfF+Xx9DtlvT5ncAYjON5VyzwkJjgF/sHC/3Onu8lOSzqmcHIsy2J9Bbac9GSsdfZXDCpvaAuF3aESjVqq7rNS4XWOQdGapToewvvOaTlSRKJJIew4qA7MhLkzjl+Rb2FxNU9uKTsqYIR/KH/no3O5Y+V8DPCDZ3bwhcc2h27rCszQLJ7izIQFP4wqutVZvbrL6Shy63LPkMyyDCQ0vLm3JhTSenvOnsa7t6qFsVmppCUnRvQqO5ZwcizLYn0Ftp72jyyrbxvymbDjWUt1sgj+O01wmaMaAYuIDCWFsOMgWBN2vGbC+goy3T+UL8ofx28vX8CwJBdPbSkj0QWNbV2s31dLTnoyUwPbw/x9q9M4NTzUFBZX82xROWMyU/nPC3pu8Hms+gsNA53Z6alrfnF1S2gW7HiEk74CW/f9I9s6fdS1dg15CDuetVQni2Dt5ujhKSQMcBsuEZHB0H4cx0FNcwfDkhJISz4+h7uvgNHTh29WWhKpSQnMHpvJpoP1rHpkE2lJLs6dMYo399Y4oeez8xmWlBCxHFlU0kBuZmqoA3/3MwMHo78eYn29tvDn7N4131rL3qpmLp0/fkDPE23d948sb2gDGPLlSDl2wZkwFeWLSLQphB0HNc2dx7Ue7FgCRviyXIHHzRMbDvHVvxTR2uWnvcsfUfju9N46EsJuOHcqv3xhFxfMG9Pv8wyVgb62jBTnLMPgzF1VcwdN7V6mutOP+vlY6L5/ZFm9M86xWUNfmC8Dd9daZw9QOLKkHZxB1uyhiAw1LUceB9UtnXFbW9J9ZulTiybwyHVLOD0vi7W7qiIK38dkpkYsR5bVt9Hp9eMZldHjY8dabmZKaDlyb1ULAFPjZKzZaZE1YWWBmbDxUTg7UgYuPy+L/31pN+DMpg5F2xURkd4ohB0HNc0duNOP30zYseipZggDh+rajip8H5MVGcL2VDm9vOIl2HQXPnN3JITFx0xYcqKL4amJoRBWHpgJy1WH9pgq8Lj55b+fDkBxVfOQtV0REemJQthxcLyXI9+Pvgrfx2SlcripHX/gzMlgsPHESbDpLrxrfnFVM6lJLsbF0XJf+NZFZfVtuDNSSElMiPGo5EOzc7ls8QRe2109ZG1XRER6ohAWZdZaalo64nY5sru+Ct/HZKbS5bOhOqbiqmayhiWF6pvizZjMVA43deD3O0X5U9wZEZuBx1pECGtoY/wIFYLHg8Liav65vXLImtuKiPRGhflR1tjupctnQy0J4l1fhe//2FoBOG0fRg1PYW9VM55R6RgTP8Em3JisVLx+S3VLB3urW5g3Pr7qenLSk0MF+eUN7UyL02XdU0n4THCBx81ST46WJEUkajQTFmU1MWjUGi3B9gkVoSW+lritB4MjLQYO1rRyqLYVT5ycGRmUnebMhFlrKatvY6xmwmJOzW1F5HjSTFiUHe9GrdEU7L1V3thOY3sXVU0dcVPo3pNgi4F1+2rx2/g7gWBkhhPCGtu8tHb6dGZkHIh1/zgRObVoJizKjvfm3dHkznA6iFc2tIcV5cdXsAkXnLl7s7gGiL+x5qQn0+nzs/uws4G6eoSJiJxaFMKirDqwebf7JJgJS3AZRg9PoaKxnb2B9hTxemYkOBsxJ7gM6/fXAjAlzsYa7BW2tdRZ6tJypIjIqUUhLMpqAiEs+wQpzO9PbqBha3FVMwkuw8SR8RVswgVDY4fXT25mChkp8bX6Hlyi3lrWCBBX7TNERCT6FMKirKalgxFpSSQlnByHekxmamAmrIWJI9NITozv1xUszp/qjq+lSICRgSXqraUNJLoMo4af+EvWIiIycPH9CXoSqGnujNs+WoMxJis1VBMWz0uRd60tdhrMBkKYZ3Q6hcXV3LW2OMYjO2JkYDlyz+FmcjNTSYijHmYiIhJ9CmFRVt3cgfskKMoPGpOVSlOHl+Kq5rg72zBcfl4Wq9dsxuJ09zeYuNsDcGRgOdLrt4xTPZiIyClHISzKalpOnC2LBiJ4xqHXb+N6JizY3+nVXU6386c2l8Zdw8305ITQcu44tacQETnlKIRFWU1zx0kVwoI1VhB/fbe6K/C4+cjcXAD+fXFeXAUwAGNMaElS7SlERE49CmFR5PX5qWvtOil6hIFTZ1Ve3xa6PNUdf3VW4QqLq3ltdzW3LPfw5OayuNwDMFgvqOVIEZFTj0JYFNW2njw9wsCps/rvZ7YDMCItifcqmuKuzioofA/Ar14wi9tXzmf1ms1xE8SCJw4EZ0nHZQ2L60ArIiJDTyEsioI9wnJOgn0jwVneu+PyBRggNdHF6j/E78bG8b4HYPDEAZ/fOXHgcFN73AZaERGJDoWwKAqFsJOoRUWBx83sscOpaOzgiiUT4zKAgbMHYPexFXjcPe4NGAvBULjxQB0AP3/+vbgNtCIiEh0KYVFU0xLYN/IkmQkDZ5mvorGD21ZM45F1B+Nmee9EVOBx8+E5zokDVy6dpAAmInKKUQiLopNp30iIrLP60kdmxl2d1YmmsLiawuIablsxjUcVaEVETjkKYVFU09xBosuQmZoU66EMiXivszqRKNCKiEh87Wh8kgluWeQ6Sbaj6ameqsDj1jLaIPQVaHU8RURODQphUVTT0nFS1YPJ0FGgFRERLUdGUXVz50lTDyYiIiJDSyEsimpaOk6q9hQiIiIydBTCoiDYDb2muTO0HKlu6CIiIhJOISwK8vOyuPXRTbR2+sjJSA6dCadu6CIiIhKkEBYFBR433/voXAA27q8LtSJQ0bWIiIgEKYRFSWaa0xvspZ2H43p7HxEREYkNhbAoefjN/QDcstyj7X1ERETkKAphUfD67mpe3lnFWZ4cvnrBLHVDFxERkaMohEXBc++WY4ErlkwCtL2PiIiIHE0d86MgJclFcqKL5TNHha5TN3QREREJp5mwIWat5fmtFXxg+ijSU5RxRUREpGcKYYMQbMYaLtiM9d3SBsoa2rlg3pgYjU5EREROBAphg5CflxVRaB/ejPUfWytIcBnOmz06xqMUERGReKb1skEIFtrf8sgmZo/NZGdFI3dcvoCzpubw7Se3ctbUHEakac9IERER6Z1mwgapwONm5pjhvLm3hpYOL9vKGthzuJm91S2cP2+M9ooUERGRPimEDVJhcTWbD9WTnGDo8ll+9OxOrntgPcZATlqS9ooUERGRPimEDUKwBmzaqAxOyxvB/dcsJsllOFTXxpjMVL79123aK1JERET6pBA2CEUlDdy+cj5NHV3kZQ9j+azRPHj9meSPz6K8oV17RYqIiEi/FMIGYdUyD2dOHkl5fTt52cNC15fUt3HbimnaK1JERET6pRA2SJVNHXj9lrzstNDy5O0r5/Olj8zUXpEiIiLSL4WwQSqpbQUgL3tYaHkyuASpvSJFRESkP+oTNkgldW0A5GWnce70UUfdrr0iRUREpC+aCRukYAgbNyI1xiMRERGRE5FC2CCV1LWSm5lCSmJCrIciIiIiJyCFsEEqrW8jLzst1sMQERGRE5RC2CCV1LVFtKcQERERORYKYYPg81vK6tsYP0IhTERERAZHIWwQKhvbQz3CRERERAZDIWwQjrSn0EyYiIiIDI5C2CCU1B1p1CoiIiIyGAphg3CkR5hCmIiIiAxOVEOYMeYCY8x7xpg9xpiv93KffzfGbDfGbDPGrInmeIZKSV0ro4enkJqkHmEiIiIyOFHbtsgYkwDcAXwYKAHWG2OettZuD7vPdOAbwNnW2jpjzOhojWcoqT2FiIiIvF/RnAk7E9hjrd1rre0EHgM+3u0+NwB3WGvrAKy1h6M4niHjhDCdGSkiIiKDF80QNh44FHa5JHBduBnADGPMG8aYt4wxF/T0QMaYG40xG4wxG6qqqqI03IEJ9gjTTJiIiIi8H7EuzE8EpgPLgc8C9xhjRnS/k7X2d9baRdbaRaNGjTq+I+xGPcJERERkKEQzhJUCE8Iu5wWuC1cCPG2t7bLW7gN24YSyuBU8M3K8ZsJERETkfYhmCFsPTDfGTDHGJAOXAU93u89TOLNgGGPcOMuTe6M4pvdNPcJERERkKEQthFlrvcBq4HlgB/BHa+02Y8x/G2M+Frjb80CNMWY78DLwn9bammiNaSiEZsLUI0xERETeh6i1qACw1j4HPNftuu+EfW+BLwW+Tgglda2MUo8wEREReZ9iXZh/wrhrbTGFxdURPcIKi6u5a21xjEcmIiIiJyKFsAHKz8ti9ZrN7DncTF52GoXF1axes5n8vKxYD01EREROQAphA1TgcfOby+ZzuKmDsvo2Vq/ZzO0r51Pgccd6aCIiInICUgg7BjPGZACw8UAdVyyZqAAmIiIig6YQdgxeec/p1n/B3DE8su4ghcXVMR6RiIiInKgUwgaosLiaHzzj7D1+xdJJ3L5yPqvXbFYQExERkUEZUAgzxvzFGHOxMeaUDW1FJQ187pypAIxIS6LA4+b2lfMpKmmI8chERETkRDTQUPVbYCWw2xjzU2PMzCiOKS6tWuZhdGYKANnpyYBTrL9qmSeWwxIREZET1IBCmLX2RWvt5cACYD/wojGm0BhzrTEmKZoDjCd1rZ0AZKedMi9ZREREomTAy4vGmBzgGuBzwGbgf3FC2QtRGVkcqm/tIjnRxTB1yxcREZH3aUDbFhljngRmAg8DH7XWlgduetwYsyFag4s3dS2dZKclYYyJ9VBERETkBDfQvSN/Y619uacbrLWLhnA8ca2utYvstORYD0NEREROAgNdjpxjjBkRvGCMyTbG3BKdIcWv+tZORqgeTERERIbAQEPYDdba+uAFa20dcENURhTH6lo7GZmumTARERF5/wYawhJMWCGUMSYBOOXSSH1rFyO0HCkiIiJDYKA1Yf/AKcK/O3D5psB1pwxrLfVtXWpPISIiIkNioCHsazjB6+bA5ReA30dlRHGqsd2Lz29VmC8iIiJDYkAhzFrrB+4MfJ2S6lqcRq1ajhQREZGhMNA+YdOBnwBzgNTg9dbaqVEaV9xRt3wREREZSgMtzL8fZxbMC3wQeAh4JFqDikf1rV2AZsJERERkaAw0hA2z1r4EGGvtAWvt94CLozes+KOZMBERERlKAy3M7zDGuIDdxpjVQCmQEb1hxZ+6wEyYCvNFRERkKAx0JuzzQBpwG7AQuAK4OlqDikf1rZ24DGQO00yYiIiIvH/9zoQFGrN+xlr7FaAZuDbqo4pDda2dZA1LIsGlzbtFRETk/et3Jsxa6wPOOQ5jiWvavFtERESG0kBrwjYbY54G/gS0BK+01v4lKqOKQ9q8W0RERIbSQENYKlADrAi7zgKnTAirbeliXFZq/3cUERERGYCBdsw/JevAwtW3djJnbGashyEiIiIniYF2zL8fZ+YrgrX2uiEfUZyqa+1UjzAREREZMgNdjnwm7PtU4FKgbOiHE5/au3y0d/nJTldhvoiIiAyNgS5H/jn8sjHmD8DrURlRHAp2y1dhvoiIiAyVgTZr7W46MHooBxLP6lqcbvkj1aJCREREhshAa8KaiKwJqwC+FpURxaH60EyYQpiIiIgMjYEuRw6P9kDiWWjfyHQtR4qIiMjQGNBypDHmUmNMVtjlEcaYT0RtVHEmWBOmjvkiIiIyVAZaE/Zda21D8IK1th74blRGFIfqWlSYLyIiIkNroCGsp/sNtL3FCa+utYu05ARSEhNiPRQRERE5SQw0hG0wxvzSGOMJfP0S2BjNgcWT+tZOLUWKiIjIkBpoCPsPoBN4HHgMaAdujdag4k2dNu8WERGRITbQsyNbgK9HeSxxq661SzNhIiIiMqQGenbkC8aYEWGXs40xz0dtVHGmvrVTWxaJiIjIkBrocqQ7cEYkANbaOk6ljvmtXdq8W0RERIbUQEOY3xgzMXjBGDOZyA76Jy2f39LY3qVu+SIiIjKkBtpm4lvA68aYtYABzgVujNqo4khDWxfWopkwERERGVIDLcz/hzFmEU7w2gw8BbRFcVxxQ93yRUREJBoGuoH354DPA3nAFmAp8CawImojixPqli8iIiLRMNCasM8Di4ED1toPAvOB+mgNKp6ENu/WTJiIiIgMoYGGsHZrbTuAMSbFWrsTmBm9YcUPLUeKiIhINAy0ML8k0CfsKeAFY0wdcCBag4on9YEQNiJdy5EiIiIydAZamH9p4NvvGWNeBrKAf0RtVHHgrrXF5OdlUdfaRaLLMDwlkcLiaopKGli1zBPr4YmIiMgJbqDLkSHW2rXW2qettZ3RGFC8yM/LYvWazbxX0ciItGTe3FvD6jWbyc/LivXQRERE5CRwzCHsVFHgcXP7yvm8trsan9/P6jWbuX3lfAo87lgPTURERE4CCmF9KPC4GZs1jLrWLq5YMlEBTERERIaMQlgfCourKatvY1xWKo+sO0hhcXWshyQiIiInCYWwXhQWV7N6zWamuNOYOiqD21fOZ/WazQpiIiIiMiQUwnpRVNLA7Svnk5yYQEqiK1QjVlTSEOuhiYiIyElgoH3CTjnBNhSdXj/JiU5WLfC4VRcmIiIiQ0IzYf3o8PpJSdRhEhERkaGldNGP8JkwERERkaGidNGPDq+PlMSEWA9DRERETjIKYf3o1HKkiIiIRIHSRT86tBwpIiIiUaB00Qef3+L1Wy1HioiIyJCLaggzxlxgjHnPGLPHGPP1Pu73b8YYa4xZFM3xHKtOrx9AM2EiIiIy5KKWLowxCcAdwIXAHOCzxpg5PdxvOPB5YF20xjJYHV4fgGrCREREZMhFM12cCeyx1u611nYCjwEf7+F+PwB+BrRHcSyDopkwERERiZZopovxwKGwyyWB60KMMQuACdbaZ/t6IGPMjcaYDcaYDVVVVUM/0l50BEKYZsJERERkqMUsXRhjXMAvgS/3d19r7e+stYustYtGjRoV/cEFhJYjk1SYLyIiIkMrmiGsFJgQdjkvcF3QcGAe8IoxZj+wFHg6norzgzNhyQmaCRMREZGhFc10sR6YboyZYoxJBi4Dng7eaK1tsNa6rbWTrbWTgbeAj1lrN0RxTMcktByZpBAmIiIiQytq6cJa6wVWA88DO4A/Wmu3GWP+2xjzsWg971AKFuanaCZMREREhlhiNB/cWvsc8Fy3677Ty32XR3Msg6GZMBEREYkWpYs+dHQF+4SpMF9ERESGlkJYHzp96hMmIiIi0aF00YeOLvUJExERkehQuuiDZsJEREQkWpQu+qCaMBEREYkWhbA+aCZMREREokXpog+qCRMREZFoUbroQ4fXj8tAosvEeigiIiJyklEI60Onz09yogtjFMJERERkaCmE9aGjy6eifBEREYkKhbA+BGfCRERERIaaEkYfOrr8KsoXERGRqFDC6EOHVyFMREREokMJow8dXj/JqgkTERGRKFAI60OH16eZMBEREYkKJYw+dHpVmC8iIiLRoYTRB9WEiYiISLQoYfSh0+tXnzARERGJCoWwPqgmTERERKJFCaMPWo4UERGRaFHC6IMK80VERCRalDD6oJkwERERiRYljD5oJkxERESiRQmjF9baQGG+zo4UERGRoacQ1guv3+K3aDlSREREokIJoxedXj+AliNFREQkKpQwetERCGGaCRMREZFoUMLoxZGZMNWEiYiIyNBTCOtFh9cHaCZMREREokMJoxfBmbCUJB0iERERGXpKGL0I1oQlJ+gQiYiIyNBTwuhFaDkySTVhIiIiMvQUwnqhmTARERGJJiWMXnSoJkxERESiSAmjF53qEyYiIiJRpITRCzVrFRERkWhSwuhFR1ewT5gK80VERGToKYT1otOnvSNFREQkepQwetHRpeVIERERiR4ljF5oJkxERESiSQmjF8GZMPUJExERkWhQwuhFp89HosuQqBAmIiIiUaCE0YuOLr+WIkVERCRqlDJ60eH1qyhfREREokYpoxedXs2EiYiISPQoZfSiw+tTo1YRERGJGoWwXnT6tBwpIiIi0aOU0QsV5ouIiEg0KWX0QoX5IiIiEk1KGb1QYb6IiIhEk1JGL1SYLyIiItGkENaLDs2EiYiISBQpZfSiUzVhIiIiEkVKGb1wCvO1HCkiIiLRoRDWCy1HioiISDQpZfTCKczX4REREZHoUMrohWrCREREJJqUMnpgrVWzVhEREYkqpYwedPksAClJKswXERGR6FAI60GH1wdAcoIOj4iIiESHUkYPOrx+AFKSdHhEREQkOpQyetAZCGGaCRMREZFoUcrogWbCREREJNqimjKMMRcYY94zxuwxxny9h9u/ZIzZbowpMsa8ZIyZFM3xDFRwJkwd80VERCRaohbCjDEJwB3AhcAc4LPGmDnd7rYZWGStzQeeAH4erfEcCxXmi4iISLRFM2WcCeyx1u611nYCjwEfD7+DtfZla21r4OJbQF4UxzNgWo4UERGRaItmyhgPHAq7XBK4rjfXA3/v6QZjzI3GmA3GmA1VVVVDOMSeqTBfREREoi0uUoYx5gpgEfA/Pd1urf2dtXaRtXbRqFGjoj6e4HKkmrWKiIhItCRG8bFLgQlhl/MC10UwxpwHfAtYZq3tiOJ4BkwzYSIiIhJt0UwZ64Hpxpgpxphk4DLg6fA7GGPmA3cDH7PWHo7iWI6JasJEREQk2qKWMqy1XmA18DywA/ijtXabMea/jTEfC9ztf4AM4E/GmC3GmKd7ebjjKhTCtIG3iIiIREk0lyOx1j4HPNftuu+EfX9eNJ9/sIIhLFkhTERERKJEKaMHHV2Bwnw1axUREZEoUQjrQadPy5EiIiISXUoZPejo0tmRIiIiEl1KGT3o9PlJTnDhcplYD0VEREROUgphPejo8qsoX0RERKJKSaMHHV6f6sFEREQkqpQ0etDp1UyYiIiIRJeSRg86vH7NhImIiEhUKWn0oNPrV48wERERiSqFsB50eH1ajhQREZGoUtLoQadPy5EiIiISXUoaPVCLChEREYk2JY0eqDBfREREok1JowdqUSEiIiLRpqTRA6dZq86OFBERkehRCOtBp5YjRUREJMqUNHrQoeVIERERiTIljR50qFmriIiIRJlCWA9UmC8iIiLRpqTRjd9v1axVREREok5Jo5tOnx+AlCQdGhEREYkeJY1uOrxOCEtO0KERERGR6FHS6KbTG5wJU2G+iIiIRI9CWDcdXh8AKZoJExERkShS0uimw6uaMBEREYk+JY1uOlUTJiIiIseBkkY3mgkTERGR40FJI+CutcUUFlcfKcxPTKCwuJq71hbHeGQiIiJyMlIIC8jPy2L1ms1sPlgHwK7KJlav2Ux+XlaMRyYiIiInI4WwgAKPm9tXzuf//rUHgF+9sIvbV86nwOOO8chERETkZKQQFqbA4+aCebkAXDp/vAKYiIiIRI1CWJjC4mr+tbOK21ZM429F5RQWV8d6SCIiInKSUggLKCyuZvWazdy+cj5f+shMbl85n9VrNiuIiYiISFQohAUUlTRE1IAFa8SKShpiPDIRERE5GRlrbazHcEwWLVpkN2zYEOthiIiIiPTLGLPRWruop9s0EyYiIiISAwphIiIiIjGgECYiIiISAwphIiIiIjGgECYiIiISAwphIiIiIjGgECYiIiISAwphIiIiIjGgECYiIiISAwphIiIiIjFwwm1bZIypAg5E+WncgHbu7puOUd90fPqnY9Q3HZ/+6Rj1Tcenf8fjGE2y1o7q6YYTLoQdD8aYDb3t8yQOHaO+6fj0T8eobzo+/dMx6puOT/9ifYy0HCkiIiISAwphIiIiIjGgENaz38V6ACcAHaO+6fj0T8eobzo+/dMx6puOT/9ieoxUEyYiIiISA5oJExEREYkBhbBujDEXGGPeM8bsMcZ8PdbjiTVjzARjzMvGmO3GmG3GmM8Hrh9pjHnBGLM78N/sWI811owxCcaYzcaYZwKXpxhj1gXeS48bY5JjPcZYMcaMMMY8YYzZaYzZYYw5S++hSMaYLwb+jW01xvzBGJN6Kr+HjDH3GWMOG2O2hl3X43vGOH4TOE5FxpgFsRv58dPLMfqfwL+zImPMk8aYEWG3fSNwjN4zxpwfk0EfRz0dn7DbvmyMscYYd+ByTN5DCmFhjDEJwB3AhcAc4LPGmDmxHVXMeYEvW2vnAEuBWwPH5OvAS9ba6cBLgcunus8DO8Iu/wz4lbV2GlAHXB+TUcWH/wX+Ya2dBZyOc5z0HgowxowHbgMWWWvnAQnAZZza76EHgAu6Xdfbe+ZCYHrg60bgzuM0xlh7gKOP0QvAPGttPrAL+AZA4O/2ZcDcwM/8NvCZdzJ7gKOPD8aYCcBHgINhV8fkPaQQFulMYI+1dq+1thN4DPh4jMcUU9bacmvtpsD3TTgfnuNxjsuDgbs9CHwiJgOME8aYPOBi4PeBywZYATwRuMspe4yMMVnAB4B7Aay1ndbaevQe6i4RGGaMSQTSgHJO4feQtfZVoLbb1b29Zz4OPGQdbwEjjDFjj8tAY6inY2St/ae11hu4+BaQF/j+48Bj1toOa+0+YA/OZ95Jq5f3EMCvgK8C4UXxMXkPKYRFGg8cCrtcErhOAGPMZGA+sA7ItdaWB26qAHJjNa448Wucf9T+wOUcoD7sj+Gp/F6aAlQB9weWa39vjElH76EQa20p8Auc/zMvBxqAjeg91F1v7xn97e7ZdcDfA9/rGAHGmI8Dpdbad7rdFJPjoxAmA2KMyQD+DHzBWtsYfpt1TrE9ZU+zNcZcAhy21m6M9VjiVCKwALjTWjsfaKHb0qPeQyYb5//EpwDjgHR6WEaRI07190x/jDHfwikneTTWY4kXxpg04JvAd2I9liCFsEilwISwy3mB605pxpgknAD2qLX2L4GrK4NTtYH/Ho7V+OLA2cDHjDH7cZawV+DUQI0ILC3Bqf1eKgFKrLXrApefwAlleg8dcR6wz1pbZa3tAv6C877SeyhSb+8Z/e0OY4y5BrgEuNwe6UOlYwQenP/ReSfw9zoP2GSMGUOMjo9CWKT1wPTAGUnJOEWMT8d4TDEVqG26F9hhrf1l2E1PA1cHvr8a+OvxHlu8sNZ+w1qbZ62djPOe+Ze19nLgZeBTgbudssfIWlsBHDLGzAxc9SFgO3oPhTsILDXGpAX+zQWPkd5DkXp7zzwNXBU4w20p0BC2bHlKMcZcgFMa8TFrbWvYTU8DlxljUowxU3AK0N+OxRhjxVr7rrV2tLV2cuDvdQmwIPA3KjbvIWutvsK+gItwzigpBr4V6/HE+gs4B2fKvwjYEvi6CKfm6SVgN/AiMDLWY42HL2A58Ezg+6k4f+T2AH8CUmI9vhgelzOADYH30VNAtt5DRx2j7wM7ga3Aw0DKqfweAv6AUx/XhfNheX1v7xnA4JzZXgy8i3OWacxfQ4yO0R6c2qbg3+u7wu7/rcAxeg+4MNbjj8Xx6Xb7fsAdy/eQOuaLiIiIxICWI0VERERiQCFMREREJAYUwkRERERiQCFMREREJAYUwkRERERiQCFMRGQAjDHLjTHPxHocInLyUAgTERERiQGFMBE5qRhjrjDGvG2M2WKMudsYk2CMaTbG/MoYs80Y85IxZlTgvmcYY94yxhQZY54M7OGIMWaaMeZFY8w7xphNxhhP4OEzjDFPGGN2GmMeDXS3FxEZFIUwETlpGGNmA58BzrbWngH4gMtxNsTeYK2dC6wFvhv4kYeAr1lr83G6ZAevfxS4w1p7OlCA03UbYD7wBWAOTjf7s6P8kkTkJJbY/11ERE4YHwIWAusDk1TDcDZ59gOPB+7zCPAXY0wWMMJauzZw/YPAn4wxw4Hx1tonAay17QCBx3vbWlsSuLwFmAy8HvVXJSInJYUwETmZGOBBa+03Iq405r+63W+w+7V1hH3vQ39DReR90HKkiJxMXgI+ZYwZDWCMGWmMmYTzt+5TgfusBF631jYAdcaYcwPXXwmstdY2ASXGmE8EHiPFGJN2PF+EiJwa9H9xInLSsNZuN8Z8G/inMcYFdAG3Ai3AmYHbDuPUjQFcDdwVCFl7gWsD118J3G2M+e/AY3z6OL4METlFGGsHOysvInJiMMY0W2szYj0OEZFwWo4UERERiQHNhImIiIjEgGbCRERERGJAIUxEREQkBhTCRERERGJAIUxEREQkBhTCRERERGJAIUxEREQkBv4/Y+W+rfV7gKsAAAAASUVORK5CYII=\n",
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
   "execution_count": 25,
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
   "execution_count": 26,
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
   "execution_count": 27,
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
   "execution_count": 28,
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
