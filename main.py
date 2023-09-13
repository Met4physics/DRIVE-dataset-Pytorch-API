import numpy as np

from Ddataset import get_dataloader

mean = (0.709, 0.381, 0.224)
std = (0.127, 0.079, 0.043)

if __name__ == '__main__':
    t, tt = get_dataloader()

    for i, j in t:
        print(type(i))
        print(i.shape)
