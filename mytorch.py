import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        ## Create the torch.nn.Parameter for the weights and bias (if bias=True)
        ## Be careful about the size
        # ----- TODO -----
        self.W = nn.Parameter(
            torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size) * 0.01
        )
        if self.bias:
            self.b = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.b = None
            
    
    def __call__(self, x):
        
        return self.forward(x)


    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """

        # call MyFConv2D here
        # ----- TODO -----
        B, C_in, H_in, W_in = x.shape
        
        # 1) unfold 输入 => [B, C_in*k*k, L]
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )
        # x_unfold shape: [B, C_in*k*k, L]
        
        # 2) 计算 H_out, W_out (也可从 x_unfold.shape[2] 推断)
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # 3) 卷积核 W reshape => [out_channels, C_in*k*k]
        W_reshape = self.W.view(self.out_channels, -1)  # (oc, C_in*k*k)
        
        # 4) x_unfold shape 改变以做 batch 矩阵乘法 => [B, L, C_in*k*k]
        #    然后 (B, L, C_in*k*k) x (oc, C_in*k*k)^T => (B, L, oc)
        x_unfold_t = x_unfold.permute(0, 2, 1)  # => [B, L, C_in*k*k]
        
        out_temp = torch.matmul(x_unfold_t, W_reshape.t()) # => [B, L, out_channels]
        
        # 5) 加上 bias
        if self.bias:
            out_temp += self.b  # broadcasting => shape [out_channels]

        # 6) reshape => [B, out_channels, H_out, W_out]
        out = out_temp.permute(0, 2, 1).view(B, self.out_channels, H_out, W_out)
        
        return out
        ## ----- unfold method -------
       
        # ## ------- for for for for method---------
        # if self.padding > 0:
        #     x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        # B, C_in, H_in, W_in = x.shape
        # H_out = (H_in - self.kernel_size) // self.stride + 1
        # W_out = (W_in - self.kernel_size) // self.stride + 1

        # # 输出 shape = (B, out_channels, H_out, W_out)
        # out = torch.zeros(B, self.out_channels, H_out, W_out, device=x.device)

        # # for loop 
        # for b in range(B):
        #     for oc in range(self.out_channels):
        #         for oh in range(H_out):
        #             for ow in range(W_out):
        #                 # 取 region
        #                 h_start = oh * self.stride
        #                 w_start = ow * self.stride
        #                 region = x[b, :, 
        #                            h_start : h_start + self.kernel_size,
        #                            w_start : w_start + self.kernel_size]
        #                 # region shape: (in_channels, kernel_size, kernel_size)
        #                 # W[oc] shape: (in_channels, kernel_size, kernel_size)
        #                 val = (region * self.W[oc]).sum()
        #                 if self.bias:
        #                     val += self.b[oc]
                        
        #                 out[b, oc, oh, ow] = val
        # return out
class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size

        ## Take care of the stride
        ## Hint: what should be the default stride_size if it is not given? 
        ## Think about the relationship with kernel_size
        # ----- TODO -----
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride



    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """
        
        ## check the dimensions
        self.batch_size = x.shape[0]
        self.channel = x.shape[1]
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]
        
        ## Derive the output size
        # ----- TODO -----
        ## -------------for loop method--------------

        # self.output_height = (self.input_height - self.kernel_size) // self.stride + 1
        # self.output_width  = (self.input_width  - self.kernel_size) // self.stride + 1

        # self.output_channels = self.channel
        # self.x_pool_out = torch.zeros(
        #     (self.batch_size, self.output_channels, self.output_height, self.output_width),
        #     device=x.device
        # )
        # ## Maxpooling process
        # ## Feel free to use for loop
        # # ----- TODO -----
        # for b in range(self.batch_size):               # 第1重：batch
        #     for c in range(self.channel):             # 第2重：channel
        #         for oh in range(self.output_height):   # 第3重：输出高度
        #             for ow in range(self.output_width):# 第4重：输出宽度
        #                 # 计算在输入 x 中的起始坐标
        #                 h_start = oh * self.stride
        #                 w_start = ow * self.stride
        #                 # 取出 kernel_size x kernel_size 的小区域
        #                 region = x[b, c,
        #                           h_start : h_start + self.kernel_size,
        #                           w_start : w_start + self.kernel_size]
        #                 # region 里取最大值
        #                 self.x_pool_out[b, c, oh, ow] = region.max()

        # return self.x_pool_out
        B, C, H_in, W_in = x.shape
        
        # 用 unfold 把 x 切成patch => [B, C*k*k, L]
        x_unfold = F.unfold(
            x, 
            kernel_size=self.kernel_size, 
            stride=self.stride,
            padding=0  # 通常池化不需要padding
        )
        
        # 计算输出高宽
        H_out = (H_in - self.kernel_size)//self.stride + 1
        W_out = (W_in - self.kernel_size)//self.stride + 1
        L = H_out * W_out
        
        # 重塑 => [B, C, k*k, L]
        x_unfold = x_unfold.view(B, C, self.kernel_size*self.kernel_size, L)
        
        # 在 k*k 维度上取 max => [B, C, L]
        out_pooled, _ = x_unfold.max(dim=2)
        
        # reshape => [B, C, H_out, W_out]
        out = out_pooled.view(B, C, H_out, W_out)
        return out        

if __name__ == "__main__":

    ## Test your implementation!

    ## ----- TEST CASE 1 MyConv2D --------
    x_conv1 = torch.tensor([[
        [[1.0,  2.0,  3.0,  4.0],
        [5.0,  6.0,  7.0,  8.0],
        [9.0, 10.0, 11.0, 12.0],
        [0.0,  1.0,  2.0,  3.0]]
    ]], dtype=torch.float)   # shape (1,1,4,4)

    print("Input shape:", x_conv1.shape)

    # MyConv2D: in_channels=1, out_channels=1, kernel_size=3
    my_conv = MyConv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

    # kernel
    W_test = torch.tensor([[[[ 1.0,  0.0, -1.0],
                            [ 0.5,  0.0, -0.5],
                            [ 1.0,  1.0,  1.0]]]], dtype=torch.float)
    my_conv.W.data = W_test  # (out_channels=1, in_channels=1, k=3, k=3)

    # MyConv2D result
    out_my = my_conv(x_conv1)
    print("MyConv2D Output shape:", out_my.shape)
    print("MyConv2D Output:\n", out_my)

    # nn.Conv2d result
    conv_ref = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
    conv_ref.weight.data = W_test.clone()  
    out_ref = conv_ref(x_conv1)
    print("Builtin Conv2d Output:\n", out_ref)
    print("Difference:", (out_my - out_ref).abs().max())


    ## ----- TEST CASE 2 MyMaxPool2D(kernel_size=2, stride=2) --------
    # input: shape (1, 1, 4, 4)
    x_test1 = torch.tensor([[
        [[ 1.0,  2.0,  5.0,  3.0],
        [ 4.0,  8.0,  9.0, 10.0],
        [ 2.0,  3.0,  1.0,  1.0],
        [ 6.0,  7.0,  0.0, -1.0]]
    ]], dtype=torch.float)

    print("Input shape:", x_test1.shape)
    print("Input:\n", x_test1)

    # MyMaxPool2D
    my_pool = MyMaxPool2D(kernel_size=2, stride=2)
    out_my_pool = my_pool(x_test1)
    print("MyMaxPool2D Output:\n", out_my_pool)

    # nn.MaxPool2d
    import torch.nn as nn
    pool_ref = nn.MaxPool2d(kernel_size=2, stride=2)
    out_ref = pool_ref(x_test1)
    print("PyTorch MaxPool2d Output:\n", out_ref)
    print("Is close to PyTorch builtin?", torch.allclose(out_my_pool, out_ref))
    
    