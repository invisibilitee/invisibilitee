from .darknet import Darknet
from .darknet2 import Darknet2

# mingyu
from .darknet3 import Darknet3

from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hourglass2 import HourglassNet2
from .hourglass3 import HourglassNet3

from .hrnet import HRNet
from .hrnet2 import HRNet2
from .regnet import RegNet
from .regnet2 import RegNet2
from .res2net2 import Res2Net2
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .ssd_vgg2 import SSDVGG2
from .ssd_vgg3 import SSDVGG3

from .trident_resnet import TridentResNet

__all__ = [
    'RegNet', 'RegNet2', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'SSDVGG2','SSDVGG3', 'HRNet', 'HRNet2', 'Res2Net', 'Res2Net2',
    'HourglassNet', 'HourglassNet2','HourglassNet2', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet', 'Darknet2','Darknet3',
    'ResNeSt', 'TridentResNet'
]
