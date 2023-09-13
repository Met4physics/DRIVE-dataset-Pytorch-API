# DRIVE Dataset Pytorch easy API

参考[[deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master)]

代码结构如图

```
  ├── DRIVE: DRIVE数据集
  ├── compute_mean_std.py: 计算每个通道的mean与std
  ├── Ddataset.py: 自定义dataset用于读取DRIVE数据集
  ├── transform.py: 包含变换方法，如镜像、剪切
  └── main.py: 测试文件
```

使用方法

```python
from Ddataset import get_dataloader

train_loader, test_loader = get_dataloader(path='./', batch_size=4, num_workers=4, shuffle=False, 
                                           pin_memory=False, mean=(0.709, 0.381, 0.224), 
                                           std=(0.127, 0.079, 0.043))
'''
path: DRIVE文件夹所在目录
mean与std由compute_mean_std.py计算
'''
```

如需自定义操作，请修改transform.py