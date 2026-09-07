# Exploring Efficient Open-Vocabulary Segmentation in the Remote Sensing

## OBRSISBench目的
以前不同论文使用的数据集和实验设置不统一，需要建立一个统一的遥感开放词汇分割测试环境

## 实验设置
构建OBRSISBench，使用8个遥感数据集：
- DLRSD
- iSAID
- LoveDA
- Potsdam
- UAVid
- UDD5
- Vaihingen
- VDD
（其中DLRSD和iSAID作为训练集，其他数据集作为测试集）
（用于测试跨域迁移和开放词汇泛化）

## 验证普通开放词汇模型迁移到遥感后是否会下降
## 实验目的
普通自然图像开放词汇分割模型 vs 已有遥感开放词汇分割模型

## 结果比较方法
### 普通方法
- SCAN
- SAN
- SED
- Cat-Seg

### 已有OVRSIS方法
- OVRS
- GSNet

## 实验变量
1、训练数据
2、视觉骨干：ViT-B or ViT-L

## 指标
1、mIoU
2、mACC

## RSKT-Seg模型组成
1、三个预训练编码器：普通CLIP、RS-DINO、RS-CLIP
2、RS-CMA
3、RS-Fusion：SET & CET
4、RS-Transfer


