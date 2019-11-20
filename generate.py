import torch
import torchvision.models as models
import torch.nn as nn
import torchvision

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class modifyNet(nn.Module):
    def __init__(self, model):

        super(modifyNet, self).__init__()

        #读取已经训练好的模型的层数
        self.conv1 = model._modules['conv1']
        self.bn1 = model._modules['bn1']
        self.relu = model._modules['relu']
        self.maxpool = model._modules['maxpool']
        self.layer1 = model._modules['layer1']
        self.layer2 = model._modules['layer2']
        self.layer3 = model._modules['layer3']

        del model #删除加载的已经训练好的模型
    def forward(self, data):

        x = self.conv1(data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)#这里可以定义想要的层数

        return x

if __name__ == "__main__":
   model = models.resnet18(pretrained=True).cuda() #加载已经训练好的模型
   print(model)#查看模型

   modifyModel = modifyNet(model)#重新构建模型，定义输入层为指定输出层
   modifyModel = modifyModel.eval()#测试模式
   example = torch.rand(1, 3, 224, 224).cuda() # 注意，我这里导出的是CUDA版的模型，因为我的模型是在GPU中进行训练的
   traced_script_module = torch.jit.trace(modifyModel, example)
   output = traced_script_module(torch.ones(1,3,224,224).cuda())
   traced_script_module.save('resnet-descriptor-modify-trace.pt')#保存重新构建的模型
