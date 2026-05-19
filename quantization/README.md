# Quantization 运行说明

推荐从 `quantization` 目录运行，因为 `main.py` 默认会从 `./configs/` 读取配置文件。

```bash
cd /at2-data/mominghao/gen_rec/UniGenRec-A-universal-generative-recommendation-toolbox/quantization
```

## 训练 RQ-VAE

把 `<DATASET>` 和 `<EMBEDDING_MODEL>` 换成实际名称：

```bash
python main.py \
  --model_name opq \
  --dataset_name Beauty \
  --embedding_model text-embedding-3-large-pca512
```

示例：

```bash
python main.py \
  --model_name rqvae \
  --dataset_name Baby \
  --embedding_model sentence-t5-base
```

## 训练 RQ-VAE + FAISS 初始化

`rqvae_faiss` 会在模型初始化时使用全部 `item_embeddings` 训练 FAISS `ResidualQuantizer`，再把 FAISS 学到的每层 codebook 拷贝到 RQ-VAE 的量化层。

```bash
python main.py \
  --model_name rqvae_faiss \
  --dataset_name <DATASET> \
  --embedding_model <EMBEDDING_MODEL>
```

对应配置文件：

```text
configs/rqvae_faiss_config.yaml
```

## 输入文件要求

默认会读取：

```text
../datasets/<DATASET>/embeddings/<DATASET>.emb-text-<EMBEDDING_MODEL>.npy
```

例如：

```text
../datasets/Baby/embeddings/Baby.emb-text-sentence-t5-base.npy
```

## 输出位置

默认日志保存到：

```text
../logs/quantization/<DATASET>/rqvae/text-<EMBEDDING_MODEL>/
```

默认模型保存到：

```text
../ckpt/quantization/<DATASET>/rqvae/text-<EMBEDDING_MODEL>/
```

默认码本保存到：

```text
../datasets/<DATASET>/codebooks/<DATASET>.text.rqvae.npy
../datasets/<DATASET>/codebooks/<DATASET>.text.rqvae.codebook.json
```

## 常用自定义路径

```bash
python main.py \
  --model_name rqvae \
  --dataset_name <DATASET> \
  --embedding_model <EMBEDDING_MODEL> \
  --config_path ./configs/rqvae_config.yaml \
  --data_base_path ../datasets \
  --log_base_path ../logs/quantization \
  --ckpt_base_path ../ckpt/quantization \
  --codebook_base_path ../datasets
```

## 配置文件

RQ-VAE 的参数在：

```text
configs/rqvae_config.yaml
```

运行前重点确认：

- `common.device`：`cuda:0` 或 `cpu`
- `rqvae.model_params.num_levels`
- `rqvae.model_params.codebook_size`
- `rqvae.model_params.sk_epsilons` 的长度必须等于 `num_levels`
- `rqvae.training_params.batch_size`
- `rqvae.training_params.epochs`
- `rqvae.training_params.lr`

## 去重层开关

各量化配置的 `model_params.has_dup_layer` 控制是否在基础 semantic code 后追加一层去重 ID：

```yaml
model_params:
  has_dup_layer: true   # 追加去重层，最终码本列数 = semantic 层数 + 1
```

关闭时：

```yaml
model_params:
  has_dup_layer: false  # 不追加去重层，最终码本列数 = semantic 层数
```

推荐模型会从对应的量化配置读取同一个开关来校验码本列数，所以生成码本和训练推荐模型时要保持配置一致。
