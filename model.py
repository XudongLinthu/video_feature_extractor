import sys
import torch as th
import torchvision.models as models
from videocnn.models import resnext
from torch import nn

sys.path.insert(0, "/private/home/bkorbar/torch_projects/fe_h21M/")
from VTC.models.video_classification import r2plus1d_152, r2plus1d_34


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])


class AdaptivePool_bruno(nn.Module):
    def __init__(self, d=2):
        super(AdaptivePool, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, d, d))

    def forward(self, x):
        return self.pool(x).view(-1, 2048)
    
class AdaptivePool(nn.Module):
    def __init__(self, d=1):
        super(AdaptivePool, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, d, d))

    def forward(self, x):
        #print(x.shape)
        #print(self.pool(x).view(-1, 512).shape)
        return self.pool(x).view(-1, 512)

def get_model(args):
    assert args.type in ["2d", "3d", "ig"]
    if args.type == "2d":
        print("Loading 2D-ResNet-152 ...")
        model = models.resnet152(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        model = model.cuda()
    elif args.type == "3d":
        print("Loading 3D-ResneXt-101 ...")
        model = resnext.resnet101(
            num_classes=400,
            shortcut_type="B",
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            last_fc=False,
        )
        model = model.cuda()
        model_data = th.load(args.resnext101_model_path)
        model.load_state_dict(model_data)
    else:
        model = r2plus1d_34()
        checkpoint = th.load(args.ig_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model = th.nn.Sequential(*list(model.children())[:-2], AdaptivePool())
        model = model.cuda()

    model.eval()
    print("loaded")
    return model
