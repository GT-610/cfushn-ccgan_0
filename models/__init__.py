from .sngan import sngan_generator, sngan_discriminator

from .ResNet_embed import *
from .resnetv2 import *

cnn_dict = {
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
}