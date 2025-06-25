import gc
import time
import os
import argparse
from torch.utils.data import DataLoader, RandomSampler
import torch
from utils.logger import MyWriter
from core.resnet import SM_SCM_rn18
from core.res2net_v2 import SSC_net, SSD_net
from utils import metrics
from dataset import SSCM_dataloader
from tqdm import tqdm
from utils.hparams import HParam
import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning, RuntimeWarning))

# from torchvision import transforms
pred_vec = False

def ssd2pointcloud_torch(cloud, mask, diff, format='img_rear'):
    """
    cloud: torch.Tensor of shape [H, W, 3]
    mask: torch.Tensor of shape [H, W]
    diff: torch.Tensor of shape [H, W]
    """
    # Ensure types
    cloud = cloud.float()
    mask = mask.float()
    diff = diff.float()

    mask_obj = (mask > 0).unsqueeze(2).float()  # shape [H, W, 1]

    # Compute unit vectors
    uv = torch.sqrt(torch.sum(cloud ** 2, dim=2, keepdim=True))  # [H, W, 1]
    uv = cloud / uv.clamp(min=1e-6)  # Avoid division by zero
    uv[uv != uv] = 0  # Remove NaNs

    # Compute front view shift
    fr = uv * diff.unsqueeze(2)  # [H, W, 3]
    cr = (cloud + fr) * mask_obj
    cloudm = cloud * mask_obj

    obj_pt = torch.cat([cr.reshape(-1, 3), cloud.reshape(-1, 3)], dim=0)  # Not used unless you want it

    if format == 'img_rear':
        return cr
    elif format == 'cloud_rear':
        return cr.reshape(-1, 3)

def main(hp, num_epochs, resume, name):

    checkpoint_dir = "{}/{}".format(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hp.log, name))

    # get model, PLUS just used to switch the model i want to use
    if hp.PLUS:
        model = SM_SCM_rn18()
        print('SM_SCM_rn18')
    else:
        model = SSC_net()
        print('SSC_net')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)
 
    criterion_SS = metrics.SS_Loss()
    criterion_CM = metrics.COMAP_Loss()
    criterion_PC = metrics.PC_Error()

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.1)

    # starting params
    best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint["epoch"]
            step = checkpoint["step"]

            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # get data
    if hp.PLUS:
        training_set = SSCM_dataloader.SSCM_dataset(
            hp.gn_root, hp.camera, rgb_only=False, pred_depth=True)
        
        validation_set = SSCM_dataloader.SSCM_dataset(
            hp.gn_root, hp.camera, split='valid', rgb_only=False, pred_depth=True)
    else:
        training_set = SSCM_dataloader.SSCM_dataset(
            hp.gn_root, hp.camera, rgb_only=False, pred_depth=False, pred_cloud=True)
        validation_set = SSCM_dataloader.SSCM_dataset(
            hp.gn_root, hp.camera, split='valid', rgb_only=False, pred_depth=False, pred_cloud=True)
    object_list = training_set.get_onject_list()
    print("Data length: ", len(training_set))

    # creating loaders
    train_sampler = RandomSampler(
        training_set, replacement=True, num_samples=int(len(training_set)/2))

    train_batch_sampler = SSCM_dataloader.ImageSizeBatchSampler(
        train_sampler, hp.batch_size, False, cfg=hp)

    train_dataloader = DataLoader(
        training_set, batch_sampler=train_batch_sampler, num_workers=hp.batch_size)

    val_dataloader = DataLoader(
        validation_set, batch_size=hp.val_batch_size, num_workers=0, shuffle=False)

    current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # 轉換為MB 2024.01.24
    print(f"Dataloader Current GPU Memory Usage: {current_memory:.2f} MB")

    if not resume:
        step = 0
    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()  # 2024.01.24
        # print("Epoch {}/{}".format(epoch, num_epochs - 1))
        # print("-" * 10)

        # run training and validation
        # logging accuracy and loss
        # train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()
        loss_ss = metrics.MetricTracker()
        loss_cm = metrics.MetricTracker()
        # iterate over data

        loader = tqdm(train_dataloader, desc="training")
        for idx, data in enumerate(loader):
            torch.cuda.empty_cache()  # 2024.01.24

            # get the inputs and wrap in Variable
            inputs = data["rgbd"].to(device)
            # print('input', inputs.shape)
            labels = data["gt"].to(device)
            # print('label', labels[:,6,:,:].shape)
            FR_CLOUD = data["FR_CLOUD"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)
            # print('output', torch.unsqueeze(outputs[:,6,:,:], 1).shape)
            # outputs = torch.nn.functional.sigmoid(outputs)

            SS = outputs[:, :len(object_list), ...]
            CM = outputs[:, len(object_list):, ...]
            SS_L = labels[:, 0, ...]
            # CM_L = labels[:, 1:, ...]
            if pred_vec:
                CM = FR_CLOUD + CM

            # print('cm shape', CM.shape)
            l_ss = criterion_SS(SS, SS_L)
            l_cm = criterion_CM(CM, labels, use_mask=True)

            loss = l_ss + l_cm*50

            # backward
            loss.backward()
            optimizer.step()

            # train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))
            loss_ss.update(l_ss.data.item(), outputs.size(0))
            loss_cm.update(l_cm.data.item(), outputs.size(0))

            # tensorboard logging
            if step % hp.logging_step == 0:
                writer.log_training(train_loss.avg, step)
                loader.set_description(
                    "Training Loss: {:.4f}, L_ss: {:.4f}, L_cm: {:.4f}".format(
                        train_loss.avg, loss_ss.avg, loss_cm.avg
                    )
                )
            step += 1

        # Validation
        torch.cuda.empty_cache()
        if epoch % hp.validation_interval == 0:
            valid_metrics = validation(
                val_dataloader, model, criterion_PC, writer, step, device
            )
            save_path = os.path.join(
                checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, epoch+1)
            )
            # store best loss and save a model checkpoint
            best_loss = min(valid_metrics["valid_loss"], best_loss)
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "arch": "ResUnet",
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            print("Saved checkpoint to: %s" % save_path)
        # step the learning rate scheduler
        lr_scheduler.step()


def validation(valid_loader, model, criterion, logger, step, device):

    # logging accuracy and loss
    # valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()
    pc_error = metrics.MetricTracker()
    object_list = SSCM_dataloader.object_list

    # switch to evaluate mode
    model.eval()
    torch.set_grad_enabled(False)

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc="validation")):

        # get the inputs and wrap in Variable
        inputs = data["rgbd"].to(device)
        labels = data["gt"].to(device)
        FR_CLOUD = data["FR_CLOUD"].to(device)

        outputs = model(inputs)
        # outputs = torch.nn.functional.sigmoid(outputs)
        SM = outputs[:, :len(object_list), ...]
        diff = outputs[:, len(object_list):, ...]
        if pred_vec:
            CM = FR_CLOUD + CM
        SS_L = labels[:, 0, ...]
        # CM_L = labels[:, 1:, ...]

        # l_ss = criterions[0](SS, SS_L)
        # l_cm = criterions[1](CM, labels, use_mask=True)
        # loss = l_ss + 50*l_cm

        probs = torch.softmax(SM, dim=1)
        class_masks = (probs > 0.5).float()
        segMask_pre = torch.argmax(class_masks, dim=1)
        segMask_pre = torch.squeeze(segMask_pre)

        diff = torch.squeeze(diff)
        diff = diff * (segMask_pre > 0).float()
        rear_pt_pre = ssd2pointcloud_torch(FR_CLOUD, segMask_pre, diff)

        l_pc = criterion()

        valid_loss.update(loss.data.item(), outputs.size(0))
        loss_ss.update(l_ss.data.item(), outputs.size(0))
        loss_cm.update(l_cm.data.item(), outputs.size(0))
        # if idx == 0:
        #     logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)
    logger.log_validation(valid_loss.avg, step)

    print("Valid Loss: {:.4f}, L_ss: {:.4f}, L_cm: {:.4f}".format(
        valid_loss.avg, loss_ss.avg, loss_cm.avg))
    model.train()
    torch.set_grad_enabled(True)
    return {"valid_loss": valid_loss.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic segmentation and 6DCM Prediction")
    # parser.add_argument("-c", "--config", type=str, required=True, help="yaml file for configuration")
    parser.add_argument("-c", "--config", type=str,
                        default='/media/dsp520/Grasp_2T/grasp/6DCM_Grasp/6DCM/configs/default.yaml', help="yaml file for configuration")
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--name", default="SSCM",
                        type=str, help="Experiment name")

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name)
