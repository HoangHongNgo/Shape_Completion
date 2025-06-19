from torchvision import transforms
from dataset import SSCM_dataloader
from utils import metrics
from dataset.reader import *
from dataset.augment import resizer, resizer_input
from core.res2net_v2 import SSD_net, SSC_net
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import os
import random
from eval import ssc2pointcloud

object_list = SSCM_dataloader.object_list
mapping = SSCM_dataloader.mapping

def one_hot_decoding(one_hot):
    segMask = np.zeros(one_hot.shape[:2])
    single_layer = np.argmax(one_hot, axis=-1)
    for k in mapping.keys():
        segMask[single_layer == k] = mapping[k]
    segMask = np.asarray(segMask, dtype='int')
    return segMask

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("-checkpoints", type=str, required=True, help="checkpoint weight file")
    parser.add_argument("-checkpoints", type=str, default="/home/dsp520/grasp/6DCM_Grasp/6DCM/checkpoints/SSC/SSC_checkpoint_0019.pt", help="checkpoint weight file")
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--gn", default="/media/dsp520/Grasp_2T/graspnet", type=str, help="gn_root")
    parser.add_argument("--camera", default="realsense", type=str, help="camera(kinect/realsense)")
    parser.add_argument("--scene", default=140, type=int, help="scene_num")
    parser.add_argument("--frame", default=0, type=int, help="frame_num")
    parser.add_argument("--name", default="testing", type=str, help="testing dump name")
    parser.add_argument("--result", default="results", type=str, help="dump dir")
    parser.add_argument("--RESNET_PLUS_PLUS", default=False, type=bool)
    args = parser.parse_args()

    scene = random.randint(100, 129)
    frame = random.randint(0, 255)
    scene = 10
    frame = 20
    print("Testing on Scene: %3d, Frame: %3d" % (scene, frame))

    if args.RESNET_PLUS_PLUS:
        model = SSD_net().cpu().eval()
    else:
        # model = ResUnet(4).cpu().eval()
        model = SSC_net().cpu().eval()

    checkpoint = torch.load(args.checkpoints)
    model.load_state_dict(checkpoint["state_dict"])

    criterion_SS = metrics.SS_Loss()
    criterion_CM = metrics.COMAP_Loss()

    rgb = SSCM_dataloader.read_rgb_np(os.path.join(args.gn, "scenes", "scene_"+str(scene).zfill(4), args.camera, "rgb", str(frame).zfill(4)+".png"))
    depth = SSCM_dataloader.read_depth_np(os.path.join(args.gn, "scenes", "scene_"+str(scene).zfill(4), args.camera, "depth", str(frame).zfill(4)+".png"))
    comap = None
    try:
        comap = read_comap_np_old(os.path.join(args.gn, 'comap', 'scene_%04d' % scene, args.camera), frame)
        flag = True
    except:
        print('no comap ground truth')
        flag = False
    segMask = SSCM_dataloader.read_mask_np(os.path.join(args.gn, "scenes", "scene_" + str(scene).zfill(4), args.camera, "label",str(frame).zfill(4) + ".png"))
    rgbd = np.append(rgb, np.expand_dims(depth, axis=2), axis=2)

    try:
        rgbd, segMask, comap = resizer(rgbd, segMask, comap)    
    except:
        rgbd, segMask = resizer_input(rgbd, segMask)
    input_tensor = torch.from_numpy(rgbd).permute(2,0,1).float()
    # print('input tensor: ', input_tensor.shape[1]/2)
    # input_tensor=transforms.functional.resize(input_tensor, (360, 640))
    input_tensor=torch.unsqueeze(input_tensor, 0).cpu()
    # print("V_shape: ", input_tensor.shape)
    for i, cls in enumerate(mapping):
        segMask = np.where(segMask == mapping[i], i, segMask)

    flag=False
    if flag:
        gt = np.append(np.expand_dims(segMask, axis=2), comap, axis=2)
        labels = torch.from_numpy(gt).permute(2, 0, 1).float()
        labels = torch.unsqueeze(labels, 0).cpu()

    with torch.no_grad():
        output_tensor = model(input_tensor).cpu()
        print("output_tensor : ", output_tensor.shape)

    if flag:
        SS = output_tensor[:, :len(object_list), ...]
        CM = output_tensor[:, len(object_list):, ...]
        SS_L = labels[:, 0, ...]
        # CM_L = labels[:, 1:, ...]
        l_ss = criterion_SS(SS, SS_L)
        l_cm = criterion_CM(CM, labels, use_mask=True)
        print('L_total: %4f, l_ss: %4f, l_cm: %4f'%(l_ss+l_cm, l_ss, l_cm))

    print(output_tensor.shape)
    probs = torch.softmax(output_tensor[:, :len(object_list), ...], dim=1)
    class_masks = (probs > 0.3).float()     # threshold = 0.5
    segMask_pre = torch.argmax(class_masks, dim=1)
    segMask_pre = torch.squeeze(segMask_pre).detach().numpy()
    for k in mapping.keys():
        segMask_pre[segMask_pre == k] = mapping[k]
        segMask[segMask == k] = mapping[k]

    # test 2023.12.18, If run SSD, add unsqueeze(0).
    # print(output_tensor.shape)
    # comap_pre = torch.squeeze(output_tensor[:,len(object_list):,...]).unsqueeze(0).permute(1,2,0).detach().numpy()

    map_pre = output_tensor[:, len(object_list):, ...]
    comap_pre = ((torch.squeeze(output_tensor[:,len(object_list):,...]).permute(1,2,0).detach().numpy()))
    map_pre = torch.squeeze(map_pre).permute(1, 2, 0).cpu().numpy()
    map_pre = map_pre * np.expand_dims(np.asarray(segMask_pre > 0), axis=2)
    rear_pt_pre = ssc2pointcloud(segMask_pre, map_pre)
    print(rear_pt_pre.min(), rear_pt_pre.max())
    # print(comap_pre.max())
    # print("comap_pre : ", comap_pre.shape)
    # print(output_tensor.shape)
    # output_tensor = transforms.functional.resize(output_tensor, (int(rgb.shape[0]), int(rgb.shape[1]))).permute(1,2,0)
    # print(output_tensor.shape)
    # output = output_tensor.detach().numpy()
    # print(output.shape)
    
    os.makedirs("{}/{}".format(args.result, args.name), exist_ok=True)
    
    # segMask_pre = one_hot_decoding(torch.squeeze(output_tensor[:,:len(object_list),...]).permute(1,2,0).detach().numpy())

    comap_pre = comap_pre * np.expand_dims(np.asarray(segMask>0), axis=2)
    print("comap : ", comap.max())
    

    # show front view comap
    if comap is not None:
        
        # fig, axis = plt.subplots(1, 4)
        # axis[0].imshow(rgbd[...,:3])
        # axis[0].axis('off')
        # axis[0].set_title('RGB image')
        # axis[1].imshow(comap_pre[...,:3])
        # axis[1].axis('off')
        # axis[1].set_title('Front predict')
        # axis[2].imshow(comap[..., :3])
        # axis[2].axis('off')
        # axis[2].set_title('Front GT')
        # axis[3].imshow(segMask_pre)
        # axis[3].axis('off')
        # axis[3].set_title('seg')

        plt.show()

        # show rare view comap
        print(comap_pre[..., -1].mean(),comap_pre[..., -2].min(),comap_pre[..., -3].min())

        print(comap[..., -1].mean(),comap[..., -2].min(),comap[..., -3].min())

        fig, axis = plt.subplots(1, 3)
        axis[0].imshow(rgb) #Need to reshape the image
        axis[0].axis('off')
        axis[0].set_title('RGB image')
        axis[1].imshow( rear_pt_pre)
        axis[1].axis('off')
        axis[1].set_title('Rear predict')
        axis[2].imshow(comap[..., 3:])
        axis[2].axis('off')
        axis[2].set_title('Rear GT')
        plt.show()

        # show sefMask
        fig, axis = plt.subplots(1, 3)
        axis[0].imshow(rgb)
        axis[0].axis('off')
        axis[0].set_title('RGB image')
        axis[1].imshow(segMask_pre)
        axis[1].axis('off')
        axis[1].set_title('Mask predict')
        axis[2].imshow(segMask)
        axis[2].axis('off')
        axis[2].set_title('Mask GT')
        plt.show()
    else:
        fig, axis = plt.subplots(1, 4)
        axis[0].imshow(rgbd[..., :3])
        axis[0].axis('off')
        axis[0].set_title('RGB image')
        axis[1].imshow(segMask_pre)
        axis[1].axis('off')
        axis[1].set_title('Mask predict')
        axis[2].imshow(comap_pre[...,:3])
        axis[2].axis('off')
        axis[2].set_title('Front predict')
        axis[3].imshow(comap_pre[..., 3:])
        axis[3].axis('off')
        axis[3].set_title('Rear predict')
        plt.show()
