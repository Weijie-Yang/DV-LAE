##DV-LAE是什么?

DV-LAE是一个基于 Python 语言开发的工具包，旨在通过对称函数分析和筛选势函数训练数据，实现对多样性的统计、筛选和可视化。该工具包能够帮助用户分析样本数据的多样性，同时有效节省训练成本。

##Requirements
```
ase(atomic simulation environment)
scipy (library for scientific computing)
plotly (interactive graphing library)
scikit-learn (machine learning library)
tqdm (progress bar library)
```

##Example

使用`example`中的数据进行分析并进行精简数据。如下示例所示：
```python
    form DV_LAE import DV_LAE
    functionpath = r'example/function.data'
    inputpath = r'example/input.data'
    interval = 0.05,
    data_2d, data_2d_path = DV_LAE(functionpath, reffunctionpath=None, intervelnum=10, mode='tsne', inputpath=inputpath,
                                   savename=None, refnum=1, num=-1, distancemode=0, interval=interval)

```
	
    运行结束后，在对应 functionpath 文件夹中会生成 HTML 文件，用于数据多样性可视化，并生成以 output 开头的文件，包含精简后的数据，用于后续势函数训练等。
	
##参数介绍

```
functionpath：对称函数文件目录
reffunctionpath：参考结构对称函数目录，默认为 None，使用 functionpath
intervelnum：使用直方图统计的区间个数
mode：降维方式，默认为 tsne，可选 pca、tsne
inputpath：势函数训练结构源文件
distancemode：不同的直方图统计模式，默认为 0，可选 0、1、2。选择模式 0 时，统计两个直方图区间有不同则将差异向量取 1，反之取 0；选择模式 1 时，统计两个直方图区间使用二者距离作为差异向量；选择模式 2 时，统计两个直方图区间有不同则将差异向量则加 1，反之加 0
savename：样本多样性可视化保存文件名，默认为 None，保存在 functionpath 下，命名格式为 [日期]_[源文件名]_[intervelnum]_[mode]_[distancemode].html
interval：使用降维后分布进行筛选时，取的网格大小，默认为 0.05
max_points_per_grid：使用降维后分布进行筛选时，每个网格最多保留样本的数量，默认为 1
output：自定义精简后数据文件名，默认为 None，保存在 functionpath 下，命名格式为 output_[日期]_[源文件名]_[interval]_[max_points_per_grid].data
```