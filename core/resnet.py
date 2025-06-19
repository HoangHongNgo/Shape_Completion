import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
# from vis_utils.data_utils import CameraInfo
# from vis_utils.depth2pci import create_point_cloud_from_depth_image_tensor
from core.modules import ResidualConv, Upsample, Upsample_interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from generate_6DCM.utils.utils import CameraInfo

class Decoder(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, input):
        return self.fc(input)

class Encoder(nn.Module):
    def __init__(self, input_size, output_size, stride) -> None:
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, stride, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, input):
        return self.fc(input)
    
    
class ModifiedResNet18(nn.Module):
    def __init__(self, in_channel = 3, pretrained=False):
        super(ModifiedResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)  # Load pre-trained ResNet18
        self.resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)


    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x2s = self.resnet.relu(x)
        x = self.resnet.maxpool(x2s)

        x4s = self.resnet.layer1(x)
        x8s = self.resnet.layer2(x4s)
        x16s = self.resnet.layer3(x8s)
        x = self.resnet.layer4(x16s)

        # print(f"x shape: {x.shape}")

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.resnet.fc(x)

        return x2s, x4s, x8s, x16s, x

 

    
class SSD_rn18_net(nn.Module):
    def __init__(self, fcdim=512, s8dim=128, s4dim=64, s2dim=32, t0_dim=32, t1_dim=32, raw_dim2=16, class_num=41):
        super(SSD_rn18_net, self).__init__() 

        self.encoder = ModifiedResNet18(in_channel=4)

        self.conv16s = Decoder(512, 256)

        self.conv8s = Decoder(256, 128)

        self.conv4s = Decoder(128, 64)

        self.conv2s = Decoder(64, 64)

        self.conv1s = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True)
        )

        self.conv0s =  nn.Conv2d(32, class_num+1, 1, 1)    # You can change predicted channel num at this line 

        self.fusion16s = nn.Conv2d(512, 256, 3, 1, 1)
        self.fusion8s = nn.Conv2d(256, 128, 3, 1, 1)
        self.fusion4s = nn.Conv2d(128, 64, 3, 1, 1)
        self.fusion2s = nn.Conv2d(128, 64, 3, 1, 1)

    def forward(self, input):
        # print(f"input shape: {input.shape}")
        x2s, x4s, x8s, x16s, xfc = self.encoder(input)
        print(f"x2s shape: {x2s.shape}")
        print(f"x4s shape: {x4s.shape}")
        print(f"x8s shape: {x8s.shape}")
        print(f"x16s shape: {x16s.shape}")
        print(f"xfc shape: {xfc.shape}")

        fm1 = self.conv16s(xfc)
        # print(f"fm1 shape after conv16: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x16s.size(2), x16s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv8s(self.fusion16s(torch.cat([fm1, x16s], 1)))
        # print(f"fm1 shape after conv8: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x8s.size(2), x8s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv4s(self.fusion8s(torch.cat([fm1, x8s], 1)))
        # print(f"fm1 shape after conv4: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x4s.size(2), x4s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv2s(self.fusion4s(torch.cat([fm1, x4s], 1)))
        # print(f"fm1 shape after conv2s: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x2s.size(2), x2s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv1s(self.fusion2s(torch.cat([fm1, x2s], 1)))
        # print(f"fm1 shape after conv1s: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [input.size(2), input.size(3)])

        fm1 = self.conv0s(fm1)

        return fm1


class SM_rn18(nn.Module):
    def __init__(self, class_num=41):
        super(SM_rn18, self).__init__() 

        self.encoder = ModifiedResNet18(in_channel=4)

        self.conv16s = Decoder(512, 256)

        self.conv8s = Decoder(256, 128)

        self.conv4s = Decoder(128, 64)

        self.conv2s = Decoder(64, 64)

        self.conv1s = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True)
        )

        self.conv0s =  nn.Conv2d(32, class_num, 1, 1)    # You can change predicted channel num at this line 

        self.fusion16s = nn.Conv2d(512, 256, 3, 1, 1)
        self.fusion8s = nn.Conv2d(256, 128, 3, 1, 1)
        self.fusion4s = nn.Conv2d(128, 64, 3, 1, 1)
        self.fusion2s = nn.Conv2d(128, 64, 3, 1, 1)

    def forward(self, input):
        # print(f"input shape: {input.shape}")
        x2s, x4s, x8s, x16s, xfc = self.encoder(input)
        # print(f"x2s shape: {x2s.shape}")
        # print(f"x4s shape: {x4s.shape}")
        # print(f"x8s shape: {x8s.shape}")
        # print(f"x16s shape: {x16s.shape}")
        # print(f"xfc shape: {xfc.shape}")

        fm1 = self.conv16s(xfc)
        # print(f"fm1 shape after conv16: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x16s.size(2), x16s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv8s(self.fusion16s(torch.cat([fm1, x16s], 1)))
        # print(f"fm1 shape after conv8: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x8s.size(2), x8s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv4s(self.fusion8s(torch.cat([fm1, x8s], 1)))
        # print(f"fm1 shape after conv4: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x4s.size(2), x4s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv2s(self.fusion4s(torch.cat([fm1, x4s], 1)))
        # print(f"fm1 shape after conv2s: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x2s.size(2), x2s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv1s(self.fusion2s(torch.cat([fm1, x2s], 1)))
        # print(f"fm1 shape after conv1s: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [input.size(2), input.size(3)])

        fm1 = self.conv0s(fm1)

        return fm1
    
class SM_rn18_v2(nn.Module):
    def __init__(self, class_num=41):
        super(SM_rn18_v2, self).__init__() 

        self.encoder = ModifiedResNet18(in_channel=3, pretrained=True)

        self.conv16s = Decoder(512, 512)

        self.conv8s = Decoder(512 + 256, 256)

        self.conv4s = Decoder(256 + 128, 128)

        self.conv2s = Decoder(128 + 64, 128)

        self.conv1s = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True)
        )

        self.conv0s =  nn.Conv2d(64, class_num, 1, 1)    # You can change predicted channel num at this line 

    def forward(self, input):
        # print(f"input shape: {input.shape}")
        x2s, x4s, x8s, x16s, xfc = self.encoder(input)
        # print(f"x2s shape: {x2s.shape}")
        # print(f"x4s shape: {x4s.shape}")
        # print(f"x8s shape: {x8s.shape}")
        # print(f"x16s shape: {x16s.shape}")
        # print(f"xfc shape: {xfc.shape}")

        fm1 = self.conv16s(xfc)
        # print(f"fm1 shape after conv16: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x16s.size(2), x16s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv8s(torch.cat([fm1, x16s], 1))
        # print(f"fm1 shape after conv8: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x8s.size(2), x8s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv4s(torch.cat([fm1, x8s], 1))
        # print(f"fm1 shape after conv4: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x4s.size(2), x4s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv2s(torch.cat([fm1, x4s], 1))
        # print(f"fm1 shape after conv2s: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [x2s.size(2), x2s.size(3)])
        # print(f"fm1 shape us: {fm1.shape}")

        fm1 = self.conv1s(torch.cat([fm1, x2s], 1))
        # print(f"fm1 shape after conv1s: {fm1.shape}")
        fm1 = Upsample_interpolate(fm1, [input.size(2), input.size(3)])

        fm1 = self.conv0s(fm1)

        return fm1
    
class FeatureProcessorMLP(nn.Module):
  """
  MLP for processing fused RGB and Depth features.
  """
  def __init__(self, in_channels, hidden_sizes, output_size):
    super(FeatureProcessorMLP, self).__init__()
    self.layers = nn.Sequential(
        nn.Linear(in_channels, hidden_sizes[0]),
        nn.ReLU(),  # Activation function
        # Add more hidden layers with ReLU activation as needed
        *[nn.Linear(h_in, h_out) for h_in, h_out in zip(hidden_sizes[:-1], hidden_sizes[1:])],
        nn.Linear(hidden_sizes[-1], output_size)  # Output layer
    )

  def forward(self, x):
    input = x.view(x.size(0), -1)   
    return self.layers(input)
    

class SSD_ff_v2(nn.Module):
    def __init__(self, class_num = 41) -> None:
        super(SSD_ff_v2, self).__init__()

        self.intrinsic = np.load("/mnt/sda1/grapsnet/scenes/scene_0000/realsense/camK.npy") 

        self.encoder = ModifiedResNet18(in_channel = 3, pretrained = True)

        self.mlp1 = FeatureProcessorMLP(in_channels=128*90*160, hidden_sizes=[64], output_size=42*90*160)

        self.mlp2 = FeatureProcessorMLP(in_channels=1024*12*20, hidden_sizes=[64], output_size=42*12*20)

        self.mlp3 = FeatureProcessorMLP(in_channels=84*90*160, hidden_sizes=[128, 64], output_size=42*90*160)

        self.camera = CameraInfo(640, 360, self.intrinsic[0][0] / 2, self.intrinsic[1][1] / 2, self.intrinsic[0][2], self.intrinsic[1][2], 1000)
    def forward(self, input):
        rgb_input = input[:, :3, :, :]
        depth_input = input[:, 3, :, :].unsqueeze(1).squeeze(1)

        rgb_f1, rgb_f2, rgb_f3, rgb_f4, rgb_f5 = self.encoder(rgb_input)

        # Generate pci code
        point_clouds = torch.empty(depth_input.shape[0], 3, 360, 640, device=rgb_input.device)
        for i in range(depth_input.size(0)):
            depth_image = depth_input[i].permute(1,0)
            point_cloud = create_point_cloud_from_depth_image_tensor(depth = depth_image, camera = self.camera)
            point_clouds[i] = point_cloud.permute(2,1,0)

        # point_cloud_np = point_clouds[0].cpu().numpy()
        # point_cloud_np_transposed = np.transpose(point_cloud_np, (1, 2, 0))
        # point_cloud_o3d = o3d.geometry.PointCloud()
        # point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np_transposed.reshape(-1, 3))
        # o3d.visualization.draw_geometries([point_cloud_o3d])

        # print(f"point_clouds shape: {point_clouds.shape}")

        pci_f1, pci_f2, pci_f3, pci_f4, pci_f5 = self.encoder(point_clouds)
        
        fm2 = torch.cat([rgb_f2, pci_f2], 1)

        # print(f"fm2 shape: {fm2.shape}")

        fm2 = self.mlp1(fm2)

        fm2 = fm2.view(fm2.size(0), -1, 90, 160) 

        # print(f"fm2 shape: {fm2.shape}")

        fm5 = torch.cat([rgb_f5, pci_f5], 1)

        # print(f"fm5 shape: {fm5.shape}")

        fm5 = self.mlp2(fm5)

        fm5 = fm5.view(fm5.size(0), -1, 12, 20)

        fm5 = Upsample_interpolate(fm5, [fm2.size(2), fm2.size(3)])

        # print(f"fm5 shape: {fm5.shape}")

        x = torch.cat([fm5, fm2], 1)

        x = self.mlp3(x)

        x = x.view(x.size(0), -1, 90, 160)

        x = Upsample_interpolate(x, [input.size(2), input.size(3)])

        # print(f"x shape: {x.shape}")

        return x

    
class SSD_ff_v3(nn.Module):
    def __init__(self, class_num=41):
        super(SSD_ff_v3, self).__init__()

        self.intrinsic = np.load("/mnt/sda1/grapsnet/scenes/scene_0000/realsense/camK.npy")
        self.camera = CameraInfo(640, 360, self.intrinsic[0][0] / 2, self.intrinsic[1][1] / 2, self.intrinsic[0][2], self.intrinsic[1][2], 1000)

        self.encoder = ModifiedResNet18(in_channel = 3, pretrained = True)

        self.conv1 = Encoder(128, 128, 2)
        self.conv2 = Encoder(128, 256, 2)
        self.conv3 = Encoder(256, 512, 2)

        self.dec1 = Decoder(1024 + 512, 1024)
        self.dec2 = Decoder(1024, 512)
        self.dec3 = Decoder(512, 256)
        self.dec4 = Decoder(256, 128)
        self.dec5 = Decoder(128, 64)
        
        self.conv4 = nn.Conv2d(64, 42, 1, 1)

    def forward(self, input):
        rgb_input = input[:, :3, :, :]
        # plt.imshow(rgb_input[0].permute(1,2,0).cpu().numpy().astype('uint8'))
        # plt.title('RGB Image')
        # plt.axis('off')  # Turn off axis
        # plt.show()
        depth_input = input[:, 3, :, :].unsqueeze(1).squeeze(1)

        rgb_f1, rgb_f2, rgb_f3, rgb_f4, rgb_f5 = self.encoder(rgb_input)

        # Generate pci code
        point_clouds = torch.empty(depth_input.shape[0], 3, 360, 640, device=rgb_input.device)
        for i in range(depth_input.size(0)):
            depth_image = depth_input[i].permute(1,0)
            point_cloud = create_point_cloud_from_depth_image_tensor(depth = depth_image, camera = self.camera)
            point_clouds[i] = point_cloud.permute(2,1,0)

        # print(f"point_clouds shape: {point_clouds.shape}")

        # point_cloud_np = point_clouds[0].cpu().numpy()
        # point_cloud_np_transposed = np.transpose(point_cloud_np, (1, 2, 0))
        # point_cloud_o3d = o3d.geometry.PointCloud()
        # point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np_transposed.reshape(-1, 3))
        # o3d.visualization.draw_geometries([point_cloud_o3d])

        pci_f1, pci_f2, pci_f3, pci_f4, pci_f5 = self.encoder(point_clouds)

        fm2 = torch.cat([rgb_f2, pci_f2], 1)
        fm2 = self.conv1(fm2)
        fm2 = self.conv2(fm2)
        fm2 = self.conv3(fm2)

        x = torch.cat([rgb_f5, pci_f5], 1)
        x = torch.cat([x, fm2], 1)

        x = self.dec1(x)
        x = Upsample_interpolate(x, [rgb_f4.size(2), rgb_f4.size(3)])
        x = self.dec2(x)
        x = Upsample_interpolate(x, [rgb_f3.size(2), rgb_f3.size(3)])
        x = self.dec3(x)
        x = Upsample_interpolate(x, [rgb_f2.size(2), rgb_f2.size(3)])
        x = self.dec4(x)
        x = Upsample_interpolate(x, [rgb_f1.size(2), rgb_f1.size(3)])
        x = self.dec5(x)
        x = Upsample_interpolate(x, [input.size(2), input.size(3)])

        x = self.conv4(x)

        return x

class SSD_ff_v4(nn.Module):
    def __init__(self, class_num=41):
        super(SSD_ff_v4, self).__init__()

        self.intrinsic = np.load("/mnt/sda1/grapsnet/scenes/scene_0000/realsense/camK.npy")
        self.camera = CameraInfo(640, 360, self.intrinsic[0][0] / 2, self.intrinsic[1][1] / 2, self.intrinsic[0][2], self.intrinsic[1][2], 1000)

        self.rgb_encoder = ModifiedResNet18(in_channel = 3, pretrained = True)
        self.depth_encoder = ModifiedResNet18(in_channel = 1, pretrained = True)

        self.conv1 = Encoder(128, 128, 2)
        self.conv2 = Encoder(128, 256, 2)
        self.conv3 = Encoder(256, 512, 2)

        self.dec1 = Decoder(1024 + 512, 1024)
        self.dec2 = Decoder(1024, 512)
        self.dec3 = Decoder(512, 256)
        self.dec4 = Decoder(256, 128)
        self.dec5 = Decoder(128, 64)
        
        self.conv4 = nn.Conv2d(64, 42, 1, 1)

    def forward(self, input):
        rgb_input = input[:, :3, :, :]
 
        depth_input = input[:, 3, :, :].unsqueeze(1)
        # print(depth_input.shape)

        rgb_f1, rgb_f2, rgb_f3, rgb_f4, rgb_f5 = self.rgb_encoder(rgb_input)

        depth_f1, depth_f2, depth_f3, depth_f4, depth_f5 = self.depth_encoder(depth_input)

        fm2 = torch.cat([rgb_f2, depth_f2], 1)
        fm2 = self.conv1(fm2)
        fm2 = self.conv2(fm2)
        fm2 = self.conv3(fm2)

        x = torch.cat([rgb_f5, depth_f5], 1)
        x = torch.cat([x, fm2], 1)

        x = self.dec1(x)
        x = Upsample_interpolate(x, [rgb_f4.size(2), rgb_f4.size(3)])
        x = self.dec2(x)
        x = Upsample_interpolate(x, [rgb_f3.size(2), rgb_f3.size(3)])
        x = self.dec3(x)
        x = Upsample_interpolate(x, [rgb_f2.size(2), rgb_f2.size(3)])
        x = self.dec4(x)
        x = Upsample_interpolate(x, [rgb_f1.size(2), rgb_f1.size(3)])
        x = self.dec5(x)
        x = Upsample_interpolate(x, [input.size(2), input.size(3)])

        x = self.conv4(x)

        return x        



class SM_SCM_rn18(nn.Module):
    def __init__(self, class_num = 41):
        super(SM_SCM_rn18, self).__init__()

        # Load SM net
        self.sm_net = SM_rn18_v2()

        # Load ResNet18 backbone
        self.enc = ModifiedResNet18(in_channel=2, pretrained=False)

        self.dec1 = Decoder(512 + 256, 256)
        self.dec2 = Decoder(256 + 128, 128)
        self.dec3 = Decoder(128 + 64, 128)
        self.dec4 = Decoder(128 + 64, 64)
        self.conv = nn.Conv2d(64, 1, 1, 1)

    def forward(self, input):
        rgb_input = input[:, :3, :, :]

        depth_input = input[:, 3, :, :].unsqueeze(1)

        sm_res = self.sm_net(rgb_input)
        probs = torch.softmax(sm_res, dim=1)
        class_masks = (probs > 0.6).float()
        segMask_pre = torch.argmax(class_masks, dim=1)
        segMask_pre = torch.unsqueeze(segMask_pre, 1)

        fm1, fm2, fm3, fm4, fm5 = self.enc(torch.cat([segMask_pre, depth_input, ], 1))
        
        # print(f"fm1 shape: {fm1.shape}")
        # print(f"fm2 shape: {fm2.shape}")
        # print(f"fm3 shape: {fm3.shape}")
        # print(f"fm4 shape: {fm4.shape}")
        # print(f"fm5 shape: {fm5.shape}")

        x = Upsample_interpolate(fm5, [fm4.size(2), fm4.size(3)])

        x = self.dec1(torch.cat([x, fm4], 1))

        x = Upsample_interpolate(x, [fm3.size(2), fm3.size(3)])

        x = self.dec2(torch.cat([x, fm3], 1))

        x = Upsample_interpolate(x, [fm2.size(2), fm2.size(3)])

        x = self.dec3(torch.cat([x, fm2], 1))

        x = Upsample_interpolate(x, [fm1.size(2), fm1.size(3)])

        x = self.dec4(torch.cat([x, fm1], 1))

        x = Upsample_interpolate(x, [input.size(2), input.size(3)])

        x = self.conv(x)

        x = torch.cat([sm_res, x], 1)

        # print(f"x shape: {x.shape}")

        return x
    
class SSC_SM_SCM_rn18(nn.Module):
    def __init__(self, class_num = 41):
        super(SSC_SM_SCM_rn18, self).__init__()

        # Load SM net
        self.sm_net = SM_rn18_v2()

        # Load ResNet18 backbone
        self.enc = ModifiedResNet18(in_channel=2, pretrained=False)

        self.dec1 = Decoder(512 + 256, 256)
        self.dec2 = Decoder(256 + 128, 128)
        self.dec3 = Decoder(128 + 64, 128)
        self.dec4 = Decoder(128 + 64, 64)
        self.conv = nn.Conv2d(64, 3, 1, 1)

    def forward(self, input):
        rgb_input = input[:, :3, :, :]

        depth_input = input[:, 3, :, :].unsqueeze(1)

        sm_res = self.sm_net(rgb_input)
        probs = torch.softmax(sm_res, dim=1)
        class_masks = (probs > 0.6).float()
        segMask_pre = torch.argmax(class_masks, dim=1)
        segMask_pre = torch.unsqueeze(segMask_pre, 1)

        fm1, fm2, fm3, fm4, fm5 = self.enc(torch.cat([segMask_pre, depth_input, ], 1))
        
        # print(f"fm1 shape: {fm1.shape}")
        # print(f"fm2 shape: {fm2.shape}")
        # print(f"fm3 shape: {fm3.shape}")
        # print(f"fm4 shape: {fm4.shape}")
        # print(f"fm5 shape: {fm5.shape}")

        x = Upsample_interpolate(fm5, [fm4.size(2), fm4.size(3)])

        x = self.dec1(torch.cat([x, fm4], 1))

        x = Upsample_interpolate(x, [fm3.size(2), fm3.size(3)])

        x = self.dec2(torch.cat([x, fm3], 1))

        x = Upsample_interpolate(x, [fm2.size(2), fm2.size(3)])

        x = self.dec3(torch.cat([x, fm2], 1))

        x = Upsample_interpolate(x, [fm1.size(2), fm1.size(3)])

        x = self.dec4(torch.cat([x, fm1], 1))

        x = Upsample_interpolate(x, [input.size(2), input.size(3)])

        x = self.conv(x)

        x = torch.cat([sm_res, x], 1)

        # print(f"x shape: {x.shape}")

        return x
    


        
        


