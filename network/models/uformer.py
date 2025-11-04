import torch
import torch.nn as nn
import torch.nn.functional as F

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.conv11 = nn.Conv2d(in_channels, out_channels, 1)

#     def forward(self, x):
#         skip = self.conv11(x)
#         x = self.block(x)
#         x = x + skip
#         return x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

    def flops(self, H, W): 
        flops = H*W*self.in_channel*self.out_channel*(3*3+1)+H*W*self.out_channel*self.out_channel*3*3
        return flops

class UNet(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(UNet, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out

    def flops(self, H, W): 
        flops = 0
        flops += self.ConvBlock1.flops(H, W)
        flops += H/2*W/2*self.dim*self.dim*4*4
        flops += self.ConvBlock2.flops(H/2, W/2)
        flops += H/4*W/4*self.dim*2*self.dim*2*4*4
        flops += self.ConvBlock3.flops(H/4, W/4)
        flops += H/8*W/8*self.dim*4*self.dim*4*4*4
        flops += self.ConvBlock4.flops(H/8, W/8)
        flops += H/16*W/16*self.dim*8*self.dim*8*4*4

        flops += self.ConvBlock5.flops(H/16, W/16)

        flops += H/8*W/8*self.dim*16*self.dim*8*2*2
        flops += self.ConvBlock6.flops(H/8, W/8)
        flops += H/4*W/4*self.dim*8*self.dim*4*2*2
        flops += self.ConvBlock7.flops(H/4, W/4)
        flops += H/2*W/2*self.dim*4*self.dim*2*2*2
        flops += self.ConvBlock8.flops(H/2, W/2)
        flops += H*W*self.dim*2*self.dim*2*2
        flops += self.ConvBlock9.flops(H, W)

        flops += H*W*self.dim*3*3*3
        return flops

# class Uformer(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super().__init__()
        
#         # Encoder
#         self.ConvBlock1 = ConvBlock(in_channels, 32)
#         # downsample by factor 2 to match decoder upsampling (stride=2)
#         self.pool1 = nn.Conv2d(32, 32, kernel_size=2, stride=2)

#         self.ConvBlock2 = ConvBlock(32, 64)
#         self.pool2 = nn.Conv2d(64, 64, kernel_size=2, stride=2)

#         self.ConvBlock3 = ConvBlock(64, 128)
#         self.pool3 = nn.Conv2d(128, 128, kernel_size=2, stride=2)

#         self.ConvBlock4 = ConvBlock(128, 256)
#         self.pool4 = nn.Conv2d(256, 256, kernel_size=2, stride=2)

#         self.ConvBlock5 = ConvBlock(256, 512)
        
#         # Decoder
#         self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.ConvBlock6 = ConvBlock(512, 256)
        
#         self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.ConvBlock7 = ConvBlock(256, 128)
        
#         self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.ConvBlock8 = ConvBlock(128, 64)
        
#         self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.ConvBlock9 = ConvBlock(64, 32)
        
#         self.conv10 = nn.Conv2d(32, out_channels, 3, padding=1)
        
#     def forward(self, x):
#         # Encoding
#         conv1 = self.ConvBlock1(x)
#         pool1 = self.pool1(conv1)
        
#         conv2 = self.ConvBlock2(pool1)
#         pool2 = self.pool2(conv2)
        
#         conv3 = self.ConvBlock3(pool2)
#         pool3 = self.pool3(conv3)
        
#         conv4 = self.ConvBlock4(pool3)
#         pool4 = self.pool4(conv4)
        
#         conv5 = self.ConvBlock5(pool4)
        
#         # Decoding
#         up6 = self.upv6(conv5)
#         # If spatial sizes mismatch due to rounding, center-crop both to the
#         # minimum common size so concatenation works robustly for arbitrary image sizes.
#         if up6.size(2) != conv4.size(2) or up6.size(3) != conv4.size(3):
#             th = min(up6.size(2), conv4.size(2))
#             tw = min(up6.size(3), conv4.size(3))
#             uh = up6.size(2); uw = up6.size(3)
#             ch = conv4.size(2); cw = conv4.size(3)
#             up6 = up6[:, :, (uh - th)//2:(uh - th)//2 + th, (uw - tw)//2:(uw - tw)//2 + tw]
#             conv4 = conv4[:, :, (ch - th)//2:(ch - th)//2 + th, (cw - tw)//2:(cw - tw)//2 + tw]
#         up6 = torch.cat([up6, conv4], 1)
#         conv6 = self.ConvBlock6(up6)
        
#         up7 = self.upv7(conv6)
#         if up7.size(2) != conv3.size(2) or up7.size(3) != conv3.size(3):
#             th = min(up7.size(2), conv3.size(2))
#             tw = min(up7.size(3), conv3.size(3))
#             uh = up7.size(2); uw = up7.size(3)
#             ch = conv3.size(2); cw = conv3.size(3)
#             up7 = up7[:, :, (uh - th)//2:(uh - th)//2 + th, (uw - tw)//2:(uw - tw)//2 + tw]
#             conv3 = conv3[:, :, (ch - th)//2:(ch - th)//2 + th, (cw - tw)//2:(cw - tw)//2 + tw]
#         up7 = torch.cat([up7, conv3], 1)
#         conv7 = self.ConvBlock7(up7)
        
#         up8 = self.upv8(conv7)
#         if up8.size(2) != conv2.size(2) or up8.size(3) != conv2.size(3):
#             th = min(up8.size(2), conv2.size(2))
#             tw = min(up8.size(3), conv2.size(3))
#             uh = up8.size(2); uw = up8.size(3)
#             ch = conv2.size(2); cw = conv2.size(3)
#             up8 = up8[:, :, (uh - th)//2:(uh - th)//2 + th, (uw - tw)//2:(uw - tw)//2 + tw]
#             conv2 = conv2[:, :, (ch - th)//2:(ch - th)//2 + th, (cw - tw)//2:(cw - tw)//2 + tw]
#         up8 = torch.cat([up8, conv2], 1)
#         conv8 = self.ConvBlock8(up8)
        
#         up9 = self.upv9(conv8)
#         if up9.size(2) != conv1.size(2) or up9.size(3) != conv1.size(3):
#             th = min(up9.size(2), conv1.size(2))
#             tw = min(up9.size(3), conv1.size(3))
#             uh = up9.size(2); uw = up9.size(3)
#             ch = conv1.size(2); cw = conv1.size(3)
#             up9 = up9[:, :, (uh - th)//2:(uh - th)//2 + th, (uw - tw)//2:(uw - tw)//2 + tw]
#             conv1 = conv1[:, :, (ch - th)//2:(ch - th)//2 + th, (cw - tw)//2:(cw - tw)//2 + tw]
#         up9 = torch.cat([up9, conv1], 1)
#         conv9 = self.ConvBlock9(up9)
        
#         out = self.conv10(conv9)
#         return out
    
