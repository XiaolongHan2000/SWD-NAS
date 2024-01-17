import torch
# from ptflops import get_model_complexity_info
# from model import NetworkCIFAR as Network
from thop import profile
from model_search import Network as Network
from model import NetworkCIFAR as Network_CIFAR
import genotypes

# torch.cuda.set_device(0)
init_channels = 36
CIFAR_CLASSES = 10
layers = 20
auxiliary = True
input = torch.randn(1, 3, 32, 32)
# input = input.cuda()
genotype = eval("genotypes.%s" % 'SWD_NAS')

# genotype = eval("Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))")

criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.cuda()

with torch.cuda.device(0):
    model = Network_CIFAR(init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)
    # net = Network(init_channels, CIFAR_CLASSES, layers, criterion)
    # model = model.cuda()
    model.drop_path_prob = 0.2
    flops, params = profile(model, inputs=(input,))
    # macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
print(f"FLOPs: {flops / 1e6} M FLOPs")
print(f"Number of Parameters: {params}")