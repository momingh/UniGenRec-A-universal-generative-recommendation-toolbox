import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(token) for token in value)
    return str(value)


def _ordered_texts(item_text_list):
    if not item_text_list:
        return []

    max_item_id = max(int(item_id) for item_id, _ in item_text_list)
    texts = [""] * (max_item_id + 1)
    for item_id, text in item_text_list:
        texts[int(item_id)] = _to_text(text)
    return [text if text.strip() else "N/A" for text in texts]


def _batched(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield start, values[start : start + batch_size]


def _fit_rows(embeddings: np.ndarray, expected_rows: int, label: str) -> np.ndarray:
    if embeddings.shape[0] == expected_rows:
        return embeddings.astype(np.float32, copy=False)

    print(f"[WARN] {label} 嵌入数量 ({embeddings.shape[0]}) 与预期 ({expected_rows}) 不符。")
    if embeddings.shape[0] > expected_rows:
        return embeddings[:expected_rows].astype(np.float32, copy=False)

    padding = np.zeros((expected_rows - embeddings.shape[0], embeddings.shape[1]), dtype=np.float32)
    return np.concatenate([embeddings.astype(np.float32, copy=False), padding], axis=0)


def _guess_api_dim(model_name: str) -> int:
    if "large" in model_name:
        return 3072
    if "small" in model_name:
        return 1536
    return 0


def generate_local_text(args, item_text_list) -> np.ndarray:
    print(f"使用本地模型生成文本嵌入: {args.model_name_or_path}")
    device = getattr(args, "device", torch.device("cpu"))
    model = SentenceTransformer(args.model_name_or_path, device=str(device))
    model.eval()

    texts = _ordered_texts(item_text_list)
    chunks = []
    with torch.no_grad():
        for batch_id, batch_texts in tqdm(
            _batched(texts, args.batch_size),
            total=(len(texts) + args.batch_size - 1) // args.batch_size,
            desc="Local Text Encoding",
        ):
            try:
                batch_emb = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                chunks.append(np.asarray(batch_emb, dtype=np.float32))
            except Exception as exc:
                print(f"[WARN] 本地编码批次 {batch_id // args.batch_size} 失败: {exc}")
                emb_dim = model.get_sentence_embedding_dimension()
                chunks.append(np.zeros((len(batch_texts), emb_dim), dtype=np.float32))

    if not chunks:
        raise RuntimeError("未能生成任何本地文本嵌入。")

    embeddings = _fit_rows(np.concatenate(chunks, axis=0), len(texts), "本地文本")
    print(f"本地文本嵌入维度: {embeddings.shape}")
    return embeddings


def generate_api_text(args, item_text_list) -> np.ndarray:
    print(f"使用 API 模型生成文本嵌入: {args.sent_emb_model}")
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    texts = _ordered_texts(item_text_list)
    api_emb_dim = args.api_emb_dim or _guess_api_dim(args.sent_emb_model)
    print(f"[INFO] 预期/猜测的 API 维度: {api_emb_dim if api_emb_dim > 0 else '自动检测'}")

    chunks = []
    for batch_id, batch_texts in tqdm(
        _batched(texts, args.batch_size),
        total=(len(texts) + args.batch_size - 1) // args.batch_size,
        desc="API Text Encoding",
    ):
        try:
            response = client.embeddings.create(model=args.sent_emb_model, input=batch_texts)
        except Exception as exc:
            raise RuntimeError(
                f"API 请求批次 {batch_id // args.batch_size} 失败，已停止生成 embedding。原始错误: {exc}"
            ) from exc

        batch_embeddings = [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
        if api_emb_dim <= 0 and batch_embeddings:
            api_emb_dim = len(batch_embeddings[0])
            print(f"\n[INFO] 实际检测到 API 嵌入维度: {api_emb_dim}")
        chunks.extend(batch_embeddings)

    if not chunks:
        raise RuntimeError("未能生成任何 API 文本嵌入。")

    try:
        embeddings = np.stack(chunks, axis=0)
    except ValueError as exc:
        dims = {emb.shape for emb in chunks if isinstance(emb, np.ndarray)}
        raise RuntimeError(f"API 返回的嵌入维度不一致: {dims}") from exc

    args.api_emb_dim = api_emb_dim
    embeddings = _fit_rows(embeddings, len(texts), "API 文本")
    print(f"API 文本嵌入维度: {embeddings.shape}")
    return embeddings
