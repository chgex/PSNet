

import torch


def paste_cropped_mask(mask_logits, roi_coords, mask_size):

    batch_size = mask_logits.shape[0]

    assert batch_size == len(roi_coords)
    h, w = mask_size

    batch_masks = torch.zeros(batch_size, 2, h, w)
    for i in range(batch_size):
        top, bottom, left, right = roi_coords[i]
        mask = mask_logits[i]
        batch_masks[i, :, top:bottom, left:right] = mask

    return batch_masks
