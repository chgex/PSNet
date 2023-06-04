

import os
import torch
import torch.optim as optim

from utils_a import get_circle_and_score, get_iou
from model.loss import reg_l1_loss, offset_l1_loss, iou_loss, bce_loss
from model.loss import compute_mask_loss, crop_mask_targets, paste_cropped_mask, compute_mask_iou

from model.IrisCenterNet import CenterNet
from config import Config as cfg

# for NICEII
from dataset.loader_niceii import load_data
from model.loss import focal_loss_niceii as focal_loss

# for MICHE, pad with -1
# from dataset.loader_miche import load_data
# from model.loss import focal_loss_miche as focal_loss

w_loc, w_reg, w_mask = 10, 1, 5


print("log info: ", cfg.log_info)
checkpoints_path = "./checkpoints/" + cfg.checkpoints
if not os.path.exists(checkpoints_path):
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    os.mkdir(checkpoints_path)
    print("mkdir checkpoints: ", checkpoints_path)

def create_model():
    model = CenterNet(cfg)
    return model

def load_model_and_weights(weight_path):
    print(weight_path)

    assert os.path.exists(weight_path)

    model = create_model()
    model.load_state_dict(torch.load(weight_path))
    return model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(model, optimizer, train_loader, device, epoch_info):
    model.train()
    print('Start Train')

    total_loss = 0
    total = 0
    for batch_idx, batch_all_data in enumerate(train_loader):

        batch_data = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch_all_data[:-1]]
        batch_masks = batch_all_data[-1].to(device)

        batch_images, batch_hms, \
        inner_batch_whs, inner_batch_regs, inner_batch_reg_masks, \
        outer_batch_whs, outer_batch_regs, outer_batch_reg_masks = batch_data

        print("---")
        outputs = model(batch_images)

        output, mask_logits, roi_coords = outputs[0], outputs[1], outputs[2]

        croped_mask_targets = crop_mask_targets(roi_coords, batch_masks)
        mask_loss, mask_iou = compute_mask_loss(mask_logits, croped_mask_targets)

        heatmap = output[0]
        inner_wh, inner_offset = output[1]["wh"], output[1]["offset"]
        outer_wh, outer_offset = output[2]["wh"], output[2]["offset"]

        c_loss = focal_loss(heatmap, batch_hms)

        inner_wh_loss = 0.1 * reg_l1_loss(inner_wh, inner_batch_whs, inner_batch_reg_masks)
        outer_wh_loss = 0.1 * reg_l1_loss(outer_wh, outer_batch_whs, outer_batch_reg_masks)
        inner_off_loss = offset_l1_loss(inner_offset, inner_batch_regs, inner_batch_reg_masks)
        outer_off_loss = offset_l1_loss(outer_offset, outer_batch_regs, outer_batch_reg_masks)

        inner_loss = inner_wh_loss + inner_off_loss
        outer_loss = outer_wh_loss + outer_off_loss
        reg_loss = inner_loss + outer_loss

        loss = w_loc * c_loss + w_reg * reg_loss + w_mask * mask_loss

        print("epoch:%d, batch_idx: %d, c_loss: %.5f, inner_branch_loss: %.5f, outer_branch_loss: %.5f, mask_loss: %.5f,  mask_iou: %.5f, lr: %.5f "
              % (epoch_info, batch_idx, c_loss.cpu().item(), inner_loss.cpu().item(), outer_loss.cpu().item(), mask_loss.cpu().item(), mask_iou, get_lr(optimizer)))

        total_loss += loss.cpu().item()
        total += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            with open(cfg.log_info + "-loss.txt", "a") as f:
                f.writelines("epoch:%d, batch_idx: %03d, c_loss: %.5f, inner_branch_loss: %.5f, outer_branch_loss: %.5f, mask_loss: %.5f, mask_iou: %.5f,lr: %.6f \n" %
                             (epoch_info, batch_idx, c_loss.cpu().item(), inner_loss.cpu().item(), outer_loss.cpu().item(), mask_loss.cpu().item(), mask_iou, get_lr(optimizer)))
            f.close()

    train_mloss = total_loss / total
    print("train_mean_loss: ", train_mloss)

    return train_mloss, get_lr(optimizer)


def eval_one_epoch(model, data_loader, device):
    model.eval()
    print('start validation')

    total_mask_iou = 0
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch_idx, batch_all_data in enumerate(data_loader):
            batch_data = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch_all_data[:-1]]
            batch_masks = batch_all_data[-1].to(device)

            batch_images, batch_hms, \
            inner_batch_whs, inner_batch_regs, inner_batch_reg_masks, \
            outer_batch_whs, outer_batch_regs, outer_batch_reg_masks = batch_data

            print("---")
            outputs = model(batch_images)
            output, mask_logits, roi_coords = outputs[0], outputs[1], outputs[2]

            croped_mask_targets = crop_mask_targets(roi_coords, batch_masks)
            mask_preds = paste_cropped_mask(mask_logits, roi_coords, cfg.input_size)
            mask_iou = compute_mask_iou(mask_preds, batch_masks)

            mask_loss, _ = compute_mask_loss(mask_logits, croped_mask_targets)
            total_mask_iou += mask_iou

            heatmap = output[0]
            inner_wh, inner_offset = output[1]["wh"], output[1]["offset"]
            outer_wh, outer_offset = output[2]["wh"], output[2]["offset"]

            c_loss = focal_loss(heatmap, batch_hms)

            inner_wh_loss = 0.1 * reg_l1_loss(inner_wh, inner_batch_whs, inner_batch_reg_masks)
            outer_wh_loss = 0.1 * reg_l1_loss(outer_wh, outer_batch_whs, outer_batch_reg_masks)
            inner_off_loss = offset_l1_loss(inner_offset, inner_batch_regs, inner_batch_reg_masks)
            outer_off_loss = offset_l1_loss(outer_offset, outer_batch_regs, outer_batch_reg_masks)

            inner_loss = inner_wh_loss + inner_off_loss
            outer_loss = outer_wh_loss + outer_off_loss
            reg_loss = inner_loss + outer_loss

            loss = w_loc * c_loss + w_reg * reg_loss + w_mask * mask_loss

            total_loss += loss.cpu().item()
            total += 1

            print("val, batch_idx: %d, c_loss: %.5f, inner_branch_loss: %.5f, outer_branch_loss: %.5f  mask_loss: %.5f, mask_iou: %.5f"%
                   (batch_idx, c_loss, inner_loss, outer_loss, mask_loss, mask_iou))

    val_mloss = total_loss / total
    mask_mIoU = total_mask_iou / len(val_loader)
    print('val loss: ', val_mloss)
    print(' finish validation...')

    return val_mloss, mask_mIoU


def eval_model_mIou(model, data_loader, device):

    print("===> eval model: compute mIoU")
    model.eval()

    total_iou = 0.0
    total = 0
    total_inner_iou, total_outer_iou = 0, 0

    det0_cnt, det1_cnt = 0, 0
    det0_list, det1_list = [], []
    low_score_list = []
    with torch.no_grad():
        for batch_idx, batch_all_data in enumerate(data_loader):
            batch_data = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch_all_data[:-1]]
            batch_masks = batch_all_data[-1].to(device)

            batch_images, batch_hms, \
            inner_batch_whs, inner_batch_regs, inner_batch_reg_masks, \
            outer_batch_whs, outer_batch_regs, outer_batch_reg_masks = batch_data

            print("---")
            outputs = model(batch_images)

            output = outputs[0]

            # mask_logits = outputs[1]
            # roi_coords = outputs[2]
            # croped_mask_targets = crop_mask_targets(roi_coords, batch_masks)
            # mask_loss, mask_iou = compute_mask_loss(mask_logits, croped_mask_targets)
            # # compute mask iou
            # mask_total_iou += iou

            heatmaps = output[0]
            inner_det_wh, inner_det_offset = output[1]["wh"], output[1]["offset"]
            outer_det_wh, outer_det_offset = output[2]["wh"], output[2]["offset"]

            # post process
            gt_inner_batch_info = get_circle_and_score(batch_hms, 0, inner_batch_whs, inner_batch_regs, 0.5, device)
            gt_outer_batch_info = get_circle_and_score(batch_hms, 1, outer_batch_whs, outer_batch_regs, 0.5, device)

            det_inner_batch_info = get_circle_and_score(heatmaps, 0, inner_det_wh, inner_det_offset, 0.01, device)
            det_outer_batch_info = get_circle_and_score(heatmaps, 1, outer_det_wh, outer_det_offset, 0.01, device)

            for gt_inner_info, det_inner_info, \
                gt_outer_info, det_outer_info in zip(gt_inner_batch_info, det_inner_batch_info, gt_outer_batch_info, det_outer_batch_info):

                gt_inner_circle, scores = gt_inner_info
                gt_outer_circle, scores = gt_outer_info

                det_inner_circle, scores = det_inner_info
                det_outer_circle, scores = det_outer_info

                inner_iou = get_iou(gt_inner_circle, det_inner_circle)
                outer_iou = get_iou(gt_outer_circle, det_outer_circle)

                # count
                if inner_iou == 0:
                    det0_cnt += 1
                    det0_list.append(total + 1)
                if outer_iou == 0:
                    det1_cnt += 1
                    det1_list.append(total + 1)
                if 0 < outer_iou <= 0.5:
                    low_score_list.append(total+1)

                total_inner_iou += inner_iou
                total_outer_iou += outer_iou

                iou = (inner_iou + outer_iou) / 2
                total_iou += iou
                total += 1
                print("index:%d, inner_iou:%f, outer_iou:%f, IoU:%f" % (total, inner_iou, outer_iou, iou))

    mIoU = total_iou / total
    mIoU_inner = total_inner_iou / total
    mIoU_outer = total_outer_iou / total

    print("mIou: ", mIoU)
    print("det 0 cnt: ", det0_cnt)
    print("det 1 cnt: ", det1_cnt)
    print("inner, mIoU: ", mIoU_inner)
    print("outer, mIoU: ", mIoU_outer)
    print("det0_list: ", det0_list)
    print("det1_list: ", det1_list)
    print("low_score_list: ", low_score_list)

    return mIoU, mIoU_inner, mIoU_outer, det0_cnt, det1_cnt


if __name__ == "__main__":

    model = create_model()
    # model = load_model_and_weights(checkpoints_path + "/" + "model-60.pth")
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    model.to(device)

    lr = 0.001
    start_epoch = 1
    end_epoch = 100

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # data loader
    loaders = load_data(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    for epoch in range(start_epoch, end_epoch + 1):
        train_mean_loss, cur_lr = train_one_epoch(model, optimizer, train_loader, device, epoch)

        if epoch < cfg.epoch_val:
            print("epoch: %d / %d, train_mloss: %.5f, lr: %.5f" % (epoch, end_epoch, train_mean_loss, cur_lr))
            continue

        val_mean_loss, val_mask_iou = eval_one_epoch(model, val_loader, device)
        lr_scheduler.step(val_mean_loss)  # update the learning rate

        if epoch < cfg.epoch_eval:
            with open(cfg.log_info + "-record.txt", "a") as f:
                f.writelines("epoch: %d / %d, train_mloss: %.5f, val_mloss: %.f, mask_iou: %.5f, lr: %.5f"
                             %(epoch, end_epoch, train_mean_loss, val_mean_loss, val_mask_iou, cur_lr))
                f.close()
            continue

        mIoU, mIoU_inner, mIoU_outer, det0_cnt, det1_cnt = eval_model_mIou(model, val_loader, device)
        with open(cfg.log_info + "-record.txt", "a") as f:
            f.writelines("epoch: %d / %d, train_mloss: %.5f, val_mloss: %.5f, "
                         "val_inner_mIoU: %.5f, val_outer_mIoU: %.5f, val_mIoU: %.5f, "
                         "val_mask_iou: %.5f, lr: %.5f,"
                         "det0_cnt: %d, det1_cnt: %d \n" %
                         (epoch, end_epoch, train_mean_loss, val_mean_loss,
                          mIoU_inner, mIoU_outer, mIoU,
                          val_mask_iou, cur_lr,
                          det0_cnt, det1_cnt))
            f.close()

        # save chechpoints
        if epoch % 5 == 0:
            assert os.path.exists(checkpoints_path)
            torch.save(model.state_dict(), checkpoints_path + "/" + "model-{}.pth".format(epoch))
            print("save checkpoints ...")
