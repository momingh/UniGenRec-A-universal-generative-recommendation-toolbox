import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


# 这个模块是 preprocessing 和下游训练代码之间的共享划分层。
# 预处理阶段只写出每个用户完整的、按时间排序的交互序列；
# 具体如何生成 train/valid/test 样本，由下游任务在读取时决定。


@dataclass(frozen=True)
class UserSequence:
    """一个用户在 ID 重映射之后的时间序交互。"""

    user: int
    items: List[int]
    # 可选的 Unix 时间戳，和 items 一一对齐。部分旧数据只保存 item id，
    # 因此这里允许 timestamps 为空。
    timestamps: List[int] | None = None


@dataclass(frozen=True)
class NextItemSample:
    """验证或测试阶段使用的 next-item prediction 样本。"""

    user: int
    history: List[int]
    target: int


@dataclass(frozen=True)
class LeaveOneOutSplit:
    """对完整用户序列做 leave-one-out 划分。

    train_sequences 包含 valid/test target 之前的所有 item。
    valid_samples 使用倒数第二个 item 作为 target。
    test_samples 使用最后一个 item 作为 target。
    """

    train_sequences: List[UserSequence]
    valid_samples: List[NextItemSample]
    test_samples: List[NextItemSample]
    seen_items_by_user: Dict[int, List[int]]


def _parse_interaction(value: Any) -> tuple[int, int | None]:
    """把一条原始交互记录统一解析成 (item, timestamp)。"""

    if isinstance(value, dict):
        if "item" not in value:
            raise KeyError(f"Interaction record is missing 'item': {value}")
        timestamp = value.get("timestamp")
        return int(value["item"]), int(timestamp) if timestamp is not None else None

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Interaction record list cannot be empty.")
        timestamp = value[1] if len(value) > 1 else None
        return int(value[0]), int(timestamp) if timestamp is not None else None

    return int(value), None


def load_interactions(inter_path: Path | str) -> Dict[int, UserSequence]:
    """读取 <dataset>.inter.json，并转成整数 user/item 序列。

    兼容旧格式的 item-id list，也兼容带时间戳的新格式：
    {"user": [{"item": item_id, "timestamp": ts}, ...]}.
    """
    path = Path(inter_path)
    with path.open("r", encoding="utf-8") as fp:
        raw_data = json.load(fp)

    user_to_sequence: Dict[int, UserSequence] = {}
    for raw_user, raw_interactions in raw_data.items():
        user = int(raw_user)
        items: List[int] = []
        timestamps: List[int] = []
        has_timestamps = True

        # 如果同一个用户序列里混有带时间戳和不带时间戳的记录，
        # 整个序列都按“不含时间戳”处理，避免调用方误用不完整的对齐信息。
        for raw_interaction in raw_interactions:
            item, timestamp = _parse_interaction(raw_interaction)
            items.append(item)
            if timestamp is None:
                has_timestamps = False
            else:
                timestamps.append(timestamp)

        user_to_sequence[user] = UserSequence(
            user=user,
            items=items,
            timestamps=timestamps if has_timestamps else None,
        )
    return user_to_sequence


def ordered_user_sequences(
    user_to_items: Mapping[int, Sequence[int] | UserSequence],
    min_sequence_len: int = 1,
) -> List[UserSequence]:
    """按重映射后的 user id 排序，返回稳定的用户序列。

    即使 JSON 文件重写后对象顺序变化，按 user id 排序也能保证生成样本稳定。
    """
    sequences = []
    for user in sorted(user_to_items):
        value = user_to_items[user]
        if isinstance(value, UserSequence):
            items = [int(item) for item in value.items]
            timestamps = value.timestamps
        else:
            items = [int(item) for item in value]
            timestamps = None
        if len(items) < min_sequence_len:
            continue
        sequences.append(UserSequence(user=int(user), items=items, timestamps=timestamps))
    return sequences


def load_user_sequences(
    inter_path: Path | str,
    min_sequence_len: int = 1,
) -> List[UserSequence]:
    return ordered_user_sequences(
        load_interactions(inter_path),
        min_sequence_len=min_sequence_len,
    )


def truncate_history(history: Sequence[int], max_history_len: int | None = None) -> List[int]:
    """只保留最近的 max_history_len 个历史 item。"""

    values = [int(item) for item in history]
    if max_history_len is None:
        return values
    if max_history_len < 0:
        raise ValueError(f"max_history_len must be >= 0 or None, got {max_history_len}")
    return values[-max_history_len:] if max_history_len else []


def leave_one_out_split(
    user_sequences: Iterable[UserSequence],
    min_sequence_len: int = 3,
    max_history_len: int | None = None,
) -> LeaveOneOutSplit:
    """按序列顺序把每个用户划分为 train、valid 和 test。

    最后一个 item 作为 test target，倒数第二个 item 作为 valid target，
    更早的 item 作为训练序列。输入序列应当已经在预处理阶段按时间排好序。
    """
    train_sequences: List[UserSequence] = []
    valid_samples: List[NextItemSample] = []
    test_samples: List[NextItemSample] = []
    seen_items_by_user: Dict[int, List[int]] = {}

    for user_sequence in user_sequences:
        user = int(user_sequence.user)
        items = [int(item) for item in user_sequence.items]
        if len(items) < min_sequence_len:
            continue

        train_items = items[:-2]
        valid_item = items[-2]
        test_item = items[-1]

        # 当前下游 next-item 样本只消费 item history 和 target，
        # 因此这里只把 train 部分的时间戳继续带下去。
        # 如果后续划分策略需要 target timestamp，再扩展 NextItemSample。
        train_timestamps = (
            [int(timestamp) for timestamp in user_sequence.timestamps[:-2]]
            if user_sequence.timestamps is not None
            else None
        )
        train_sequences.append(
            UserSequence(user=user, items=train_items, timestamps=train_timestamps)
        )
        valid_samples.append(
            NextItemSample(
                user=user,
                history=truncate_history(train_items, max_history_len),
                target=valid_item,
            )
        )
        test_samples.append(
            NextItemSample(
                user=user,
                history=truncate_history([*train_items, valid_item], max_history_len),
                target=test_item,
            )
        )
        seen_items_by_user[user] = sorted(set(train_items))

    return LeaveOneOutSplit(
        train_sequences=train_sequences,
        valid_samples=valid_samples,
        test_samples=test_samples,
        seen_items_by_user=seen_items_by_user,
    )


def split_from_interactions(
    inter_path: Path | str,
    min_sequence_len: int = 3,
    max_history_len: int | None = None,
) -> LeaveOneOutSplit:
    """便捷入口：读取 <dataset>.inter.json 并完成划分。"""

    return leave_one_out_split(
        load_user_sequences(inter_path, min_sequence_len=min_sequence_len),
        min_sequence_len=min_sequence_len,
        max_history_len=max_history_len,
    )


def build_prefix_train_samples(
    train_sequences: Iterable[UserSequence],
    max_history_len: int | None = None,
    use_sliding_window: bool = True,
) -> List[Dict[str, Any]]:
    """构造 TIGER/GPT 类模型训练用的 prefix-target 样本。

    例如 items=[a, b, c] 时生成：
      history=[a], target=b
      history=[a, b], target=c
    """
    samples: List[Dict[str, Any]] = []

    for user_sequence in train_sequences:
        items = [int(item) for item in user_sequence.items]
        if len(items) < 2:
            continue

        if use_sliding_window:
            target_indices = range(1, len(items))
        else:
            target_indices = [len(items) - 1]

        for target_idx in target_indices:
            history = truncate_history(items[:target_idx], max_history_len)
            samples.append(
                {
                    "user": str(user_sequence.user),
                    "history": [str(item) for item in history],
                    "target": str(items[target_idx]),
                }
            )

    return samples


def build_eval_samples(samples: Iterable[NextItemSample]) -> List[Dict[str, Any]]:
    """把 valid/test 样本转成和旧 JSONL 一致的 dict 结构。"""

    return [
        {
            "user": str(sample.user),
            "history": [str(item) for item in sample.history],
            "target": str(sample.target),
        }
        for sample in samples
    ]


def expand_sequence_windows(item_seq: Sequence[int], max_len: int) -> List[Dict[str, Any]]:
    """从单个训练序列构造 RPG/SASRec 使用的滑窗样本。

    第一个窗口会标注每个 next-item 位置；后续窗口只标注最后一个位置，
    这和当前 RPG/SASRec 的训练约定保持一致。
    """
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    items = [int(item) for item in item_seq]
    if len(items) < 2:
        return []

    n_return_examples = max(len(items) - max_len, 1)
    first_window = items[: min(len(items), max_len + 1)]
    examples = [{"item_seq": first_window, "label_all_positions": True}]

    for start in range(1, n_return_examples):
        examples.append(
            {
                "item_seq": items[start : start + max_len + 1],
                "label_all_positions": False,
            }
        )
    return examples


def build_sequence_window_train_examples(
    train_sequences: Iterable[UserSequence],
    max_len: int,
) -> List[Dict[str, Any]]:
    """为所有用户构造 RPG/SASRec 的滑窗训练样本。"""

    examples: List[Dict[str, Any]] = []
    for user_sequence in train_sequences:
        examples.extend(expand_sequence_windows(user_sequence.items, max_len=max_len))
    return examples
