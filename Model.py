import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as I
import numpy as np
import math
from pysc2.lib import static_data

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class ResidualModel(torch.nn.Module):
    def __init__(self, convChannels, residualDepth, inputChannels=49, flatten=True, residualPadding=(0, 0), softmax=False):
        super().__init__()

        # Generating convolution layers for deciding which action to take
        self.mainLayers = []
        inChannels = inputChannels
        for resIndex in range(len(convChannels)):
            resUnit = []
            for layerIndex in range(residualDepth):
                resUnit.append(nn.Conv2d(inChannels, convChannels[resIndex], 3,
                                         stride=(2 if layerIndex == 0 and resIndex != 0 else 1) if flatten else 1,
                                         padding=residualPadding if layerIndex == 0 and not flatten else (1, 1)))
                inChannels = convChannels[resIndex]
            self.mainLayers.append(resUnit)

        self.flatten = flatten

        self.p = nn.ModuleList()
        for c in self.mainLayers:
            for l in c:
                I.xavier_normal(l.weight)
                self.p.append(l)

        self.softmax = softmax

    def forward(self, x):
        xMain = x
        index = 0
        for res in self.mainLayers:
            xMain = F.leaky_relu(res[0](xMain))
            xMainRes = xMain
            for layer in res[1:]:
                xMain = F.leaky_relu(layer(xMain))
            if index != len(self.mainLayers) - 1:
                xMain = F.leaky_relu(xMain + xMainRes)
            else:
                xMain = xMain + xMainRes
            index += 1

        # if self.debugNaN:
            # print(torch.isnan(xMain).any())

        if self.flatten:
            maxPoolMain = F.max_pool2d(xMain, kernel_size=xMain.size()[2:])
            maxPoolMain = torch.flatten(maxPoolMain)

            if not self.softmax:
                return maxPoolMain
            else:
                return F.softmax(maxPoolMain, dim=-1)
        else:
            if not self.softmax:
                return torch.squeeze(xMain)
            else:
                xMain = torch.squeeze(xMain)
                flat = torch.flatten(xMain)
                sm = F.softmax(flat, dim=-1)
                return torch.reshape(sm, xMain.shape)


class CGRULayer(torch.nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, returnSequence=False):
        super().__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.returnSequence = returnSequence
        self.Wr = nn.Conv2d(inChannels + outChannels, outChannels, kernel_size=(kernelSize, kernelSize), stride=(1, 1), padding=int((kernelSize - 1)/2))
        I.xavier_normal(self.Wr.weight)
        self.Wu = nn.Conv2d(inChannels + outChannels, outChannels, kernel_size=(kernelSize, kernelSize), stride=(1, 1), padding=int((kernelSize - 1)/2))
        I.xavier_normal(self.Wu.weight)
        self.Wc = nn.Conv2d(inChannels + outChannels, outChannels, kernel_size=(kernelSize, kernelSize), stride=(1, 1), padding=int((kernelSize - 1)/2))
        I.xavier_normal(self.Wc.weight)

    def forward(self, xSeq):
        h = torch.zeros(xSeq[0].shape[0], self.outChannels, xSeq[0].shape[2], xSeq[0].shape[3])
        hSeq = []
        for x in xSeq:
            concat = torch.cat((x, h), dim=1)
            r = torch.sigmoid(self.Wr(concat))
            u = torch.sigmoid(self.Wu(concat))
            rh = r * h
            c = F.leaky_relu(self.Wc(torch.cat((x, rh), 1)))

            h = u * h + (1 - u) * c
            if self.returnSequence:
                hSeq.append(h)

        return torch.stack(hSeq) if self.returnSequence else h


class CGRUPolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.CGRUScreen1 = CGRULayer(27, 16, kernelSize=5, returnSequence=True)
        self.CGRUScreen2 = CGRULayer(16, 32, kernelSize=3, returnSequence=False)
        self.CGRUMM1 = CGRULayer(11, 16, kernelSize=5, returnSequence=True)
        self.CGRUMM2 = CGRULayer(16, 32, kernelSize=5, returnSequence=False)
        self.fullyConnected = nn.Linear(64 * 64 * (32 + 32 + 11), 256)

        self.valueEstimate = nn.Linear(256, 1)
        self.actionScores = nn.Linear(256, 573)

        self.argLayers = nn.ModuleList()

        for _ in range(3):
            self.argLayers.append(nn.Conv2d((32 + 32 + 11), 1, (1, 1), stride=(1, 1), padding=(0, 0)))

        for outLen in [2, 5, 10, 4, 2, 4, 500, 4, 10, 500]:
            self.argLayers.append(nn.Linear(256, outLen))

        for p in self.parameters():
            if len(p.shape) > 1:
                I.xavier_normal_(p)
            else:
                I.normal_(p, std=0.1)

    def forward(self, x):
        featureScreen = x[0]
        minimap = x[1]
        playerData = x[2]

        screenRepresentation = nn.ReLU()(self.CGRUScreen2(self.CGRUScreen1(featureScreen)))
        MMRepresentation = nn.ReLU()(self.CGRUMM2(self.CGRUMM1(minimap)))
        stateRepresentation = torch.cat((screenRepresentation, MMRepresentation, playerData), dim=1)
        flattenRepresentation = nn.ReLU()(self.fullyConnected(torch.flatten(stateRepresentation)))

        value = self.valueEstimate(flattenRepresentation)
        policy = nn.Softmax(dim=-1)(self.actionScores(flattenRepresentation))

        args = []
        for i in range(3):
            convPolicy = self.argLayers[i](stateRepresentation)
            convPolicy = torch.squeeze(convPolicy)
            flatConvPolicy = torch.flatten(convPolicy)
            sm = nn.Softmax(dim=-1)(flatConvPolicy)
            args.append(torch.reshape(sm, convPolicy.shape))

        for i in range(3, len(self.argLayers)):
            FCPolicy = self.argLayers[i](flattenRepresentation)
            FCPolicy = nn.Softmax(dim=-1)(FCPolicy)
            args.append(FCPolicy)

        return value, policy, args


class UnifiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.convScreen1 = nn.Conv2d(27, 16, (5, 5), stride=(1, 1), padding=(2, 2))
        self.convScreen2 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 1))
        self.convMM1 = nn.Conv2d(11, 16, (5, 5), stride=(1, 1), padding=(2, 2))
        self.convMM2 = nn.Conv2d(16, 32, (5, 5), stride=(1, 1), padding=(2, 2))
        self.fullyConnected = nn.Linear(64 * 64 * (32 + 32 + 11), 256)

        self.valueEstimate = nn.Linear(256, 1)
        self.actionScores = nn.Linear(256, 573)

        self.argLayers = nn.ModuleList()

        for _ in range(3):
            self.argLayers.append(nn.Conv2d((32 + 32 + 11), 1, (1, 1), stride=(1, 1), padding=(0, 0)))

        for outLen in [2, 5, 10, 4, 2, 4, 500, 4, 10, 500]:
            self.argLayers.append(nn.Linear(256, outLen))

        for p in self.parameters():
            if len(p.shape) > 1:
                I.xavier_normal_(p)
            else:
                I.normal_(p, std=0.1)

    def forward(self, x):
        featureScreen = x[0]
        minimap = x[1]
        playerData = x[2]

        screenRepresentation = nn.ReLU()(self.convScreen2(nn.ReLU()(self.convScreen1(featureScreen))))
        MMRepresentation = nn.ReLU()(self.convMM2(nn.ReLU()(self.convMM1(minimap))))
        # stateRepresentation = nn.ReLU()(self.conv2(nn.ReLU()(self.conv1(x))))
        stateRepresentation = torch.cat((screenRepresentation, MMRepresentation, playerData), dim=1)
        flattenRepresentation = nn.ReLU()(self.fullyConnected(torch.flatten(stateRepresentation)))

        value = self.valueEstimate(flattenRepresentation)
        policy = nn.Softmax(dim=-1)(self.actionScores(flattenRepresentation))

        args = []
        for i in range(3):
            convPolicy = self.argLayers[i](stateRepresentation)
            convPolicy = torch.squeeze(convPolicy)
            flatConvPolicy = torch.flatten(convPolicy)
            sm = nn.Softmax(dim=-1)(flatConvPolicy)
            args.append(torch.reshape(sm, convPolicy.shape))

        for i in range(3, len(self.argLayers)):
            FCPolicy = self.argLayers[i](flattenRepresentation)
            FCPolicy = nn.Softmax(dim=-1)(FCPolicy)
            args.append(FCPolicy)

        return value, policy, args
