

import numpy as np


def normalise_iris_circle(image, boxes, crop_size=(64, 512)):
    # create normalized 2D grid
    r = np.linspace(0.0, 1.0, crop_size[0] + 2)
    theta = np.linspace(0.0, 2 * np.pi, crop_size[1])
    # r = np.concatenate([r1[1:p_r1-1], r2[0:p_r2-1]], 0)
    r = r[1:crop_size[0] + 1]
    theta = theta
    r_t, theta_t = np.meshgrid(r, theta) # 生成网格点坐标矩阵
    # flatten
    r_t_flat = np.reshape(r_t, [-1])
    theta_t_flat = np.reshape(theta_t, [-1])
    sampling_grid = np.stack([r_t_flat, theta_t_flat])
    # transform

    x_pupil, y_pupil, r_pupil, x_iris, y_iris, r_iris = boxes


    yp = ((y_pupil - np.cos(sampling_grid[1, :]) * r_pupil))
    xp = ((x_pupil + np.sin(sampling_grid[1, :]) * r_pupil))
    yi = ((y_iris - np.cos(sampling_grid[1, :]) * r_iris))
    xi = ((x_iris + np.sin(sampling_grid[1, :]) * r_iris))

    # yp = y_pupil - np.sin(sampling_grid[1, :]) * r_pupil
    # xp = x_pupil + np.cos(sampling_grid[1, :]) * r_pupil
    # yi = y_iris - np.sin(sampling_grid[1, :]) * r_iris
    # xi = x_iris + np.cos(sampling_grid[1, :]) * r_iris

    # print('new norm')

    x = xp + (xi - xp) * sampling_grid[0, :]
    y = yp + (yi - yp) * sampling_grid[0, :]

    batch_grids = np.stack([x, y], axis=0)
    # batch_grids = tf.convert_to_tensor(batch_grids)
    # reshape to (num_batch, 2, H, W)
    batch_grids = np.reshape(batch_grids, [2, crop_size[1], crop_size[0]])

    x = batch_grids[0, :, :]
    y = batch_grids[1, :, :]

    H = image.shape[0]
    W = image.shape[1]
    max_y = H - 1
    max_x = W - 1
    zero = np.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    # x = np.cast(x, 'float32')
    # y = np.cast(y, 'float32')
    # x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    # y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.floor(x)
    y0 = np.floor(y)

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = np.clip(x0, zero, max_x).astype(int)
    y0 = np.clip(y0, zero, max_y).astype(int)

    outH = x0.shape[0]
    outW = x0.shape[1]
    out = np.zeros((outH, outW))
    for i in range(outH):
        for j in range(outW):
            out[i, j] = image[y0[i, j], x0[i, j]]
    return out


def get_box(circle):
    circle = np.array(circle)

    assert len(circle) == 6

    if circle[2] < circle[5]:
        in_x, in_y, in_r = circle[0], circle[1], circle[2]
        ou_x, ou_y, ou_r = circle[3], circle[4], circle[5]
    else:
        in_x, in_y, in_r = circle[3], circle[4], circle[5]
        ou_x, ou_y, ou_r = circle[0], circle[1], circle[2]

    return (in_x, in_y, in_r, ou_x, ou_y, ou_r)


def norm_iris_mask(mask, circle, size=(64, 512)):

    mask = np.array(mask)
    box = get_box(circle)

    assert len(mask.shape) == 2 and len(box) == 6

    norm_iris = normalise_iris_circle(mask, box, size)
    norm_iris = np.transpose(norm_iris)

    return norm_iris


def normalise_iris_circle_image(image, boxes, crop_size=(64, 512)):
    # create normalized 2D grid
    r = np.linspace(0.0, 1.0, crop_size[0] + 2)
    theta = np.linspace(0.0, 2 * np.pi, crop_size[1])
    # r = np.concatenate([r1[1:p_r1-1], r2[0:p_r2-1]], 0)
    r = r[1:crop_size[0] + 1]
    theta = theta
    r_t, theta_t = np.meshgrid(r, theta) # 生成网格点坐标矩阵
    # flatten
    r_t_flat = np.reshape(r_t, [-1])
    theta_t_flat = np.reshape(theta_t, [-1])
    sampling_grid = np.stack([r_t_flat, theta_t_flat])
    # transform

    x_pupil, y_pupil, r_pupil, x_iris, y_iris, r_iris = boxes


    yp = ((y_pupil - np.cos(sampling_grid[1, :]) * r_pupil))
    xp = ((x_pupil + np.sin(sampling_grid[1, :]) * r_pupil))
    yi = ((y_iris - np.cos(sampling_grid[1, :]) * r_iris))
    xi = ((x_iris + np.sin(sampling_grid[1, :]) * r_iris))

    # yp = y_pupil - np.sin(sampling_grid[1, :]) * r_pupil
    # xp = x_pupil + np.cos(sampling_grid[1, :]) * r_pupil
    # yi = y_iris - np.sin(sampling_grid[1, :]) * r_iris
    # xi = x_iris + np.cos(sampling_grid[1, :]) * r_iris

    # print('new norm')

    x = xp + (xi - xp) * sampling_grid[0, :]
    y = yp + (yi - yp) * sampling_grid[0, :]

    batch_grids = np.stack([x, y], axis=0)
    # batch_grids = tf.convert_to_tensor(batch_grids)
    # reshape to (num_batch, 2, H, W)
    batch_grids = np.reshape(batch_grids, [2, crop_size[1], crop_size[0]])

    x = batch_grids[0, :, :]
    y = batch_grids[1, :, :]

    H = image.shape[0]
    W = image.shape[1]
    max_y = H - 1
    max_x = W - 1
    zero = np.zeros([], dtype='int32')

    x0 = np.floor(x)
    y0 = np.floor(y)

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = np.clip(x0, zero, max_x).astype(int)
    y0 = np.clip(y0, zero, max_y).astype(int)

    outH = x0.shape[0]
    outW = x0.shape[1]
    out = np.zeros((outH, outW, 3))
    for i in range(outH):
        for j in range(outW):
            out[i, j] = image[y0[i, j], x0[i, j]]
    return out

def norm_iris_image(image, circle, size=(64, 512)):

    image = np.array(image, dtype=int)
    box = get_box(circle)

    assert len(image.shape) == 3 and len(box) == 6

    norm_iris = normalise_iris_circle_image(image, box, size)
    norm_iris = np.transpose(norm_iris, (1, 0, 2))  # c h w

    return norm_iris

