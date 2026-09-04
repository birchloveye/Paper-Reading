# Using knowledge graph embedding and meta-path transformers for geographic context-aware predictions of building functions

## 总结
论文把building、POI、TAZ、SVSP数据特征组成异构知识图谱，再结合KG嵌入与元路径Transformer，利用建筑周围的地理语境预测建筑功能；杭州试验达到93.05pct的OA，但是随机空间划分和不充分的组件消融使得泛化结论仍需考量。

## 研究问题
传统建筑功能预测通常使用建筑形态、遥感、街景或者POI特征，但容易把每栋建筑孤立处理；已有图模型多集中于BB，但是没有充分表达BP、BT、BS之间的多种空间和语义关系。

### 论文回答:
预测类别：居住、购物餐饮、商务办公、公共设施、教育、医疗、政府、酒店、工厂、工业园

## 核心贡献
- 01 构建BuildingKG：8类关系（B、BB、BP、BT、BS、BBP、BTT、BTS）
- 02 使用GIE知识图谱嵌入：将空间关系结构与实体属性拼接
- 03 提出MP-GTNN：对九条人工定义的元路径分别聚合邻居特征，统一投影后由多头自注意力融合，最后预测建筑功能

## 方法概要
```bash
建筑、POI、街景、TAZ
        ⬇️
空间/语义关系 -> BuildingKG
        ⬇️
GIE -> 每个实体的‘结构位置向量’
        ⬇️
沿9种元路径查找相关实体
        ⬇️
每条路径内做均值聚合
        ⬇️
MLP统一特征维度
        ⬇️
Transformer融合不同元路径
        ⬇️
Softmax预测建筑功能
```
## 实验结果

## 主要局限

## 评价
