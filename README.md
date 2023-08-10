白样本过多，原先下采样（随机）具有不稳定性，采样
更稳定方法
根据已有样本 梯度变化较大的白样本筛选出，然后增强(通过leaf差值排序)
把这些白样本（有代表性）+（剩下的无代表性的随机取一部分）
黑样本全量扩增
一起放到模型里，理想上会稳定一些且效果更好

after_sort
black_augmentation/white_augmentation

zengliang_test
group_data是原始文件
final_train_data/final_test_data是训练集和测试集
train_label_1_data.csv是原始文件中筛出的原始黑样本
