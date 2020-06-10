import torch
from flow_resnet import *
from objectAttentionModelConvLSTM import *
import torch.nn as nn


class twoStreamAttentionModel(nn.Module):
    def __init__(self, flowModel='', frameModel='', stackSize=5, memSize=512, num_classes=61):
        super(twoStreamAttentionModel, self).__init__()
        self.flowModel = flow_resnet34(False, channels=2*stackSize, num_classes=num_classes)
        if flowModel != '':
            self.flowModel.load_state_dict(torch.load(flowModel))
        self.frameModel = attentionModel(num_classes, memSize)
        if frameModel != '':
            self.frameModel.load_state_dict(torch.load(frameModel))
        self.fc2 = nn.Linear(512 * 2, num_classes, bias=True)           # as stated in the paper, results of the two networks are joined with a fc layer
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(self.dropout, self.fc2)

    def forward(self, inputVariableFlow, inputVariableFrame):
        _, flowFeats = self.flowModel(inputVariableFlow)                # the features coming from the two models are concatenated and fed to the classifier
        _, rgbFeats = self.frameModel(inputVariableFrame)               # that is a fc with 512 * 2 beacuse every model compute 512 features (last layer before classification)
        twoStreamFeats = torch.cat((flowFeats, rgbFeats), 1)
        #return self.classifier(twoStreamFeats)
        return twoStreamFeats, self.classifier(twoStreamFeats)
