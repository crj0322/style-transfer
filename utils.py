import cv2 as cv


def squar_resize(img, dst_size):
    rows, cols = img.shape[:2]
    if rows > cols:
        row_start = (rows - cols)//2
        row_end = row_start + cols
        img = img[row_start:row_end, :]
    else:
        col_start = (cols - rows)//2
        col_end = col_start + rows
        img = img[:, col_start:col_end, :]

    return cv.resize(img, (dst_size, dst_size))