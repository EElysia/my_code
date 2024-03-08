# Classifier-free Guidence Diffusion for Single Image
## Config
加入了EMA来更新参数
步数 t 的编码方式为不学习的固定位置编码

条件 c 的编码方式为可学习的编码

条件 pos 另外加入mask图像的位置，对于一张被mask的图像，编码[start_w/w, new_w/w, start_h/h, new_h/h]
然后经过固定位置编码输出维度为特定维度的1/4，最后flatten，变成特定维度。

融合方式是在Unet的上采样中，(x * c + t + pos)，然后与对应的下采样层的输出相融合。