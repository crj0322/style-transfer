import cv2 as cv


def squar_resize(img, dst_size):
    rows, cols = img.shape[:2]
    if rows > cols:
        row_start = (rows - cols)//2
        row_end = row_start + cols
        img = img[row_start:row_end, :]
    elif rows < cols:
        col_start = (cols - rows)//2
        col_end = col_start + rows
        img = img[:, col_start:col_end, :]

    return cv.resize(img, (dst_size, dst_size))

def center_crop(img, dst_size, row_offset=0, col_offset=0):
    rows, cols = img.shape[:2]
    assert rows > dst_size[0] and cols > dst_size[1]
    x, y = (rows + row_offset)//2, (cols + col_offset)//2
    left = y - dst_size[1]//2
    up = x - dst_size[0]//2
    return img[up:up+dst_size[0], left:left+dst_size[1], :]
