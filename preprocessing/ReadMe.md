# UniGenRec preprocessing

当前这套脚本的主链路是：

1. 下载/准备 Amazon 原始数据
2. 运行 `process_data.py` 生成统一的交互、元数据、review 文件
3. 运行 `process_embedding.py` 生成 item 文本或 user-item review embedding

下面的命令默认在 `preprocessing/` 目录下执行。若从项目根目录执行，需要把脚本路径改成 `preprocessing/*.py`，并显式传入 `--output_dir datasets`、`--input_path datasets`、`--output_path datasets` 或 `--save_root datasets`。

## 1. 下载数据

Amazon 2014 数据可以直接下载并抽取 ratings：

```bash
python download_data.py \
  --source amazon \
  --dataset Beauty \
  --data_version 14 \
  --output_dir ../datasets
```

下载后会生成：

```text
../datasets/amazon14/Metadata/meta_<dataset>.json.gz
../datasets/amazon14/Review/<dataset>_5.json.gz
../datasets/amazon14/Ratings/<dataset>.csv
```

`download_data.py` 仍保留了 MovieLens 下载/初步转换逻辑，但当前 `process_data.py` 主入口只支持 `--dataset_type amazon`。MovieLens 尚未接入统一预处理主链路。

## 2. 数据预处理

运行 Amazon 统一预处理：

```bash
python process_data.py \
  --dataset_type amazon \
  --dataset Beauty \
  --input_path ../datasets \
  --output_path ../datasets
```

输出目录为 `../datasets/<dataset>/`，主要文件包括：

```text
<dataset>.inter.json
<dataset>.train.jsonl
<dataset>.valid.jsonl
<dataset>.test.jsonl
<dataset>.item.json
<dataset>.review.jsonl
<dataset>.user2id
<dataset>.item2id
```

其中 `.item2id` 是 `raw_item_id -> new_item_id`，`.item.json` 使用 `new_item_id` 作为 key。后续 item 级 embedding 的行号与 `new_item_id` 对齐。

## 3. 生成 Embedding

`process_embedding.py` 当前支持：

```text
text_local
text_api
review_local
review_api
```

输出默认写到：

```text
../datasets/<dataset>/embeddings/
```

如果 `--pca_dim > 0` 且原始维度大于目标维度，会额外保存 `-pca<pca_dim>.npy` 文件。原始 `.npy` 也会保留。

### 3.1 本地 item 文本 embedding

默认本地模型路径是 `preprocessing/emb_llm/Qwen3-Embedding-8B`。如未下载，可先运行：

```bash
python emb_llm/download_qwen3_embedding_8b.py
```

生成 item 文本 embedding：

```bash
python process_embedding.py \
  --embedding_type text_local \
  --dataset Beauty \
  --dataset_type amazon \
  --save_root ../datasets \
  --pca_dim 512 \
  --gpu_id 0
```

```bash
python process_embedding.py \
  --embedding_type text_local \
  --dataset Beauty \
  --dataset_type amazon \
  --save_root ../datasets \
  --pca_dim 512 \
  --gpu_id 0
```

注意：`text_local` 和 `review_local` 当前要求本地模型目录存在，并且需要 CUDA；CPU 会直接退出。

### 3.2 API item 文本 embedding

```bash
python process_embedding.py \
  --embedding_type text_api \
  --dataset Beauty \
  --dataset_type amazon \
  --save_root ../datasets \
  --sent_emb_model text-embedding-3-large \
  --pca_dim 512
```

如果使用兼容 OpenAI Embeddings API 的服务，替换 `--openai_base_url` 和 `--sent_emb_model` 即可。

### 3.3 Review embedding

Review embedding 是 user-item 交互级别，不是 item 级别。行号对应 `<dataset>.review.jsonl` 中的 review 行，脚本会额外保存同名 `.index.jsonl` 记录 `row/user/item/timestamp` 映射。

本地模型：

```bash
python process_embedding.py \
  --embedding_type review_local \
  --dataset Beauty \
  --save_root ../datasets \
  --review_fields summary reviewText \
  --pca_dim 512 \
  --gpu_id 0
```

API：

```bash
python process_embedding.py \
  --embedding_type review_api \
  --dataset Beauty \
  --save_root ../datasets \
  --sent_emb_model text-embedding-3-large \
  --openai_api_key "$OPENAI_API_KEY" \
  --openai_base_url "https://api.openai.com/v1" \
  --pca_dim 512
```

如果后续量化或推荐代码期望 item 级 embedding，不要直接把 `review-ui` embedding 当作 item embedding 使用。

## 4. 下游量化文件名

embedding 文件名格式为：

```text
<dataset>.emb-<modality>-<model_tag>.npy
```

常见示例：

```text
Beauty.emb-text-Qwen3-Embedding-8B.npy
Beauty.emb-text-text-embedding-3-large.npy
Beauty.emb-review-ui-Qwen3-Embedding-8B.npy
```

下游 `quantization/main.py` 会按 `--embedding_modality` 和 `--embedding_model` 拼接文件名，默认查找不带 `-pca<pca_dim>` 后缀的原始 embedding 文件。若要量化 PCA 文件，需要先让文件名与该拼接规则一致，或改下游加载逻辑。

在 `quantization/` 目录下执行示例：

```bash
python main.py \
  --model_name rqvae \
  --dataset_name Beauty \
  --embedding_modality text \
  --embedding_model Qwen3-Embedding-8B \
  --data_base_path ../datasets
```
