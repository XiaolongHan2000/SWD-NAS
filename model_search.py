import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class AttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(AttentionModule, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        attention_out = x * y.expand_as(x)

        op_attention = []
        op_channel = c // 8  # Number of channels per operation
        for i in range(8):
            temp = y[:, i * op_channel:op_channel * (i + 1), :, :]  # The attention weights of i-th operation
            op_i_atten = torch.sum(temp)  # Attention weights summation
            op_attention.append(op_i_atten.item())

        return attention_out, op_attention

class ChannelAttention(nn.Module):
    def __init__(self, channel=16, reduction=2):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = max_out + avg_out
        output = self.sigmoid(output)
        return output

class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.stride = stride

        self.mp = nn.MaxPool2d(2, 2)
        self.k = 16
        self.ca = ChannelAttention(C)

        if self.stride == 2:
            self.auxiliary_op = FactorizedReduce(C, C, affine=False)
        else:
            self.auxiliary_op = Identity()

        self.channel = C // self.k
        for primitive in PRIMITIVES:
            op = OPS[primitive](C // self.k, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))
            self._ops.append(op)

        self.attention = AttentionModule(C * 8 // self.k, ratio=8)

    def forward(self, x):

        dim_2 = x.shape[1]
        num_list = self.ca(x)
        auxiliary_op = self.auxiliary_op(x)
        x = x * num_list
        slist = torch.sum(num_list, dim=0, keepdim=True)
        values, max_num_index = slist.topk(dim_2 // self.k, dim=1, largest=True, sorted=True)
        max_num_index = max_num_index.squeeze()
        num_dict = max_num_index
        xtemp = torch.index_select(x, 1, max_num_index)


        out = 0
        temp = []
        for op in self._ops:
            temp.append(op(xtemp))
        temp = torch.cat(temp[:], dim=1)  # Concatenate feature maps in channel dimension

        attention_out, op_attention = self.attention(temp)  # Calculate attention weights

        for i in range(8):  # Integrate all feature maps by element-wise addition
            out += attention_out[:, i * self.channel:self.channel * (i + 1):, :, :]

        # concat feature maps
        if out.shape[2] == x.shape[2]:
            x[:, num_dict, :, :] = out[:, :, :, :]
        else:
            x = self.mp(x)
            x[:, num_dict, :, :] = out[:, :, :, :]

        x += auxiliary_op
        return x, op_attention


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        op_Attention = []
        for i in range(self._steps):

            s = 0
            for j, h in enumerate(states):
                temp, op_attention = self._ops[offset + j](h)
                s += temp

                op_Attention.append(op_attention)  # 14*8 attention weight matrix

            offset += len(states)
            states.append(s)

        if self.reduction != True and self.reduction_prev != True:
            states.append(s1)
        return torch.cat(states[-self._multiplier:], dim=1), op_Attention  # self._multiplier=4


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier + 2

        C_curr = stem_multiplier * C  # 48
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr  # 16 16*4


        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        op_Attention_normal_all = []
        op_Attention_reduce_all = []
        for i, cell in enumerate(self.cells):

            if cell.reduction:
                s2, op_Attention_reduce = cell(s0, s1)
                op_Attention_reduce_all.append(op_Attention_reduce)  # Different cell topologies are various
            else:
                s2, op_Attention_normal = cell(s0, s1)
                op_Attention_normal_all.append(op_Attention_normal)  # Different cell topologies are various

            s0, s1 = s1, s2
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, op_Attention_normal_all, op_Attention_reduce_all

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)


