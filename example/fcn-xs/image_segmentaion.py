# pylint: skip-file
# -*- coding: utf-8 -*- 
import numpy as np
import mxnet as mx
from PIL import Image  # import python image library
from timer import Timer

pallete = [ 0,0,0,
            128,0,0,
            0,128,0,
            128,128,0,
            0,0,128,
            128,0,128,
            0,128,128,
            128,128,128,
            64,0,0,
            192,0,0,
            64,128,0,
            192,128,0,
            64,0,128,
            192,0,128,
            64,128,128,
            192,128,128,
            0,64,0,
            128,64,0,
            0,192,0,
            128,192,0,
            0,64,128 ]
# 调色板，对sematic segmentation的结果进行不同颜色的着色
# 21类加上背景类

img = "img/031.jpg"
seg = img.replace("jpg", "png")
model_prefix = "model_pascal/FCN8s_VGG16"
epoch = 19
ctx = mx.gpu(0) # context 设置，用于计算单元配置
#ctx = mx.cpu()

def get_data(img_path):
    """get the (1, 3, h, w) np.array data for the img_path"""
    mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)  数据是从imagenet来的么？
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)   # 强制转化为float32型数据
    reshaped_mean = mean.reshape(1, 1, 3)  #将RGB mean 转化为三维数据
    
    img = img - reshaped_mean                    
    img = np.swapaxes(img, 0, 2)                #维度转换，第1维和第3维互换
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)  #在零维增加一维新的？用意是什么？
    return img

def main():
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    # load the saved module parameters 返回的第一个参数是symbol
    fcnxs_args["data"] = mx.nd.array(get_data(img), ctx) 
    # 在GPU(0)中载入数据 第一次往GPU中载入数据很慢，但是之后载入的速度就快很多
    data_shape = fcnxs_args["data"].shape
    label_shape = (1, data_shape[2]*data_shape[3])
    #print data_shape,label_shape
    fcnxs_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
    exector = fcnxs.bind(ctx, fcnxs_args ,args_grad=None, grad_req="null", aux_states=fcnxs_args)
    # 为symbol的计算申请空间 ctx指明需要使用的计算单元，fcnxs_args指明需要使用的显式参数，不计算任何变量的梯度
    exector.forward(is_train=False)
    output = exector.outputs[0]
    print output.shape
    out_img = np.uint8(np.squeeze(output.asnumpy().argmax(axis=1)))
    print out_img.shape
    # squeeze函数会去掉数组中长度为1的维度数据
    # argmax函数返回相应坐标轴下最大值的索引，数组维度下降一维
    out_img = Image.fromarray(out_img)
    out_img.putpalette(pallete)
    out_img.save(seg)   

if __name__ == "__main__":
    """
    for i in np.arange(4):
        img = "img/cat{:01d}.jpg".format(i+3)
        seg = img.replace("jpg", "png")
        print i
    """
    main()
