# N-Channeled-Input-UNet-Fastai
This Repository contains a modified version of unet_learner function from fastai library which you can use to define a unet with more/less number of channels than 3 (Default in all ResNet like networks). Sometimes datasets contain non ImageNet like images, sometimes they don't even contain images as inputs, the input dimension might change in such cases. 

I have tried to change the original function as little as possible. You should checkout the notebook in this repo for more complete example. You can just copy the code below and then use this `unet_learner` function with your dataset. It will detect the number of channels present in you input data and set the `input_channel` of the model to be those many channels. 

* **Note**  that if the number of channels is not 3 then the network won't be frozen as the first layer contains untrained weights. And I don't think there is much need of pretrianing in this case as the usage of this function would be in scenarios where the images/arrays are not Imagenet like*. 

Be sure to check the [notebook](https://github.com/navidpanchi/N-Channeled-Input-UNet-Fastai/blob/master/N-Channeled-Input-UNet%20.ipynb) out for testing. 

```
from fastai.vision import *
from fastai.vision.learner import cnn_config
import torch.nn as nn

def unet_learner(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on=None, blur:bool=False,
                 self_attention:bool=False, y_range=None, last_cross:bool=True,
                 bottle:bool=False, cut=None, **learn_kwargs:Any)->Learner:
    "Build Unet learner from `data` and `arch`."
    
    # I have defined size intentionally like this, so that it won't be a problem when 
    # the input is an image 
    size = next(iter(data.train_dl))[0].shape[-2:]
    n_input_channels = next(iter(data.train_dl))[0][0].size(0)    
        
    meta = cnn_config(arch)
    body = create_body(arch, pretrained, cut)
    
    # changing the first layer to suit our input
    if not n_input_channels == 3: 
        prev_layer = body[0]
        body[0] = nn.Conv2d(n_input_channels, prev_layer.out_channels, 
                      kernel_size=prev_layer.kernel_size, 
                      stride=prev_layer.stride, 
                      padding=prev_layer.padding, 
                      bias=prev_layer.bias)

    model = to_device(models.unet.DynamicUnet(body, n_classes=data.c, img_size=size, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle), data.device)
    learn = Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained and n_input_channels != 3: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn
```
