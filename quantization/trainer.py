import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split
import utils
from collections import defaultdict

class Trainer:
    """
    通用量化器 Trainer (复用 TensorDataset)
    """

    FULL_TRAIN_VAL_SPLIT = 0.05
    PROGRESS_NCOLS = 120

    def __init__(self, config: dict, model: torch.nn.Module, device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.model_name = config["model_name"]
        self.model_cfg = config[self.model_name.lower()]
        self.train_cfg = self.model_cfg["training_params"]
        self.common_cfg = config["common"]

        self.validation_split = self.common_cfg["validation_split"]
        self.seed = self.common_cfg["seed"]
        self.num_workers = self.common_cfg["num_workers"]
        self.predict_batch_size = self.common_cfg["predict_batch_size"]

        self.batch_size = self.train_cfg["batch_size"]
        self.epochs = self.train_cfg["epochs"]
        self.optimizer_name = self.train_cfg["optimizer"]
        self.lr = float(self.train_cfg["lr"])
        self.weight_decay = float(self.train_cfg["weight_decay"])
        self.max_grad_norm = self.train_cfg.get("max_grad_norm", None)
        if self.max_grad_norm is not None:
            self.max_grad_norm = float(self.max_grad_norm)
        self.checkpoint_selection = str(
            self.train_cfg.get("checkpoint_selection", "final")
        ).lower()
        selection_aliases = {
            "last": "final",
            "final": "final",
            "best": "best_val",
            "best_val": "best_val",
            "val_best": "best_val",
        }
        if self.checkpoint_selection not in selection_aliases:
            raise ValueError(
                "checkpoint_selection must be one of: final, best_val."
            )
        self.checkpoint_selection = selection_aliases[self.checkpoint_selection]
        self.early_stop_patience = int(
            self.train_cfg.get("early_stop_patience", 30)
        )
        self.validation_metric = str(
            self.train_cfg.get("validation_metric", "mse")
        ).lower()
        if self.validation_metric not in {"mse", "loss_total"}:
            raise ValueError("validation_metric must be one of: mse, loss_total.")
        self.eval_metrics_interval = int(self.train_cfg.get("eval_metrics_interval", 100) or 0)
        self.metrics_batch_size = int(
            self.train_cfg.get("metrics_batch_size", self.predict_batch_size)
            or self.predict_batch_size
        )
        self.prefix_cosine_max_items = int(
            self.train_cfg.get("prefix_cosine_max_items", 1000) or 0
        )

        self.model_params = self.model_cfg["model_params"]
        self.has_dup_layer = bool(self.model_params.get("has_dup_layer", False))
        self.codebook_size = self.model_params["codebook_size"]

        self.logger = logging.getLogger(f"Trainer[{self.model_name}]")

    def fit(self, embeddings_data, ckpt_dir):
        """
        通用的 fit 方法，接收文本 embedding 数据。
        """
        if getattr(self.model, "is_iterative", True):
            return self._fit_iterative(embeddings_data, ckpt_dir)
        else:
            return self._fit_one_shot(embeddings_data, ckpt_dir)

    def _fit_iterative(self, embeddings_data, ckpt_dir):
        """处理需要迭代训练的模型 (接收 numpy 数据)。"""
        self.logger.info(f"开始迭代式训练 ({self.model_name})...")

        if isinstance(embeddings_data, tuple):
            raise ValueError("当前 Trainer 仅支持单文本模态 embedding。")
        tensor_data = torch.from_numpy(embeddings_data).float()
        dataset = TensorDataset(tensor_data)

        test_size = self.validation_split
        if test_size > 0:
            train_idx, val_idx = train_test_split(
                list(range(len(dataset))),
                test_size=test_size,
                random_state=self.seed
            )
            self.logger.info(f"数据集已划分为 {1-test_size:.0%} 训练 / {test_size:.0%} 验证")
        else:
            train_idx = list(range(len(dataset)))
            _, val_idx = train_test_split(
                train_idx,
                test_size=self.FULL_TRAIN_VAL_SPLIT,
                random_state=self.seed
            )
            self.logger.info("validation_split <= 0，训练集使用全集，验证集从训练集中抽取 5%")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True) if val_idx else None
        self.logger.info(f"DataLoader: batch_size={self.batch_size}, num_workers={self.num_workers}")

        params_to_optimize = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = None
        if params_to_optimize:
            optimizer_class = getattr(torch.optim, self.optimizer_name)
            optimizer = optimizer_class(params_to_optimize, lr=self.lr, weight_decay=self.weight_decay)
            self.logger.info(f"优化器: {self.optimizer_name}, LR: {self.lr}, WeightDecay: {self.weight_decay}")
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                self.logger.info(f"梯度裁剪: max_grad_norm={self.max_grad_norm}")
        else:
            self.logger.info("模型没有可训练参数，不创建优化器。")
        if self.eval_metrics_interval > 0:
            self.logger.info(
                "训练中指标评估: interval=%d epochs, metrics_batch_size=%d, prefix_cosine_max_items=%d",
                self.eval_metrics_interval,
                self.metrics_batch_size,
                self.prefix_cosine_max_items,
            )

        final_path = os.path.join(ckpt_dir, f"{self.model_name}_final.pth")
        best_path = os.path.join(ckpt_dir, f"{self.model_name}_best_val.pth")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        use_best_val = self.checkpoint_selection == "best_val"
        if use_best_val:
            if val_loader is None:
                self.logger.warning(
                    "checkpoint_selection=best_val 需要验证集；将回退到 final checkpoint。"
                )
                use_best_val = False
            else:
                self.logger.info(
                    "checkpoint_selection=best_val，使用验证集 %s 选择 checkpoint，"
                    "early_stop_patience=%d epochs。",
                    self.validation_metric,
                    self.early_stop_patience,
                )
        else:
            self.logger.info("checkpoint_selection=final，使用最后一次训练模型生成 code。")

        best_val_metric = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        last_epoch_num = 0
        avg_losses = {"loss_total": float("inf")}
        avg_val_loss = float("inf")
        avg_val_metric = float("inf")
        pbar = tqdm(range(self.epochs), desc=f"Training {self.model_name}", ncols=self.PROGRESS_NCOLS)
        for epoch in pbar:
            self.model.train()
            epoch_loss_sum = defaultdict(float)
            for batch in train_loader:
                loss_dict = {}
                batch_xs = batch[0].to(self.device)
                outputs = self.model(xs=batch_xs)
                loss_dict = self.model.compute_loss(outputs, batch_data=batch_xs)

                loss_total = loss_dict["loss_total"]

                if optimizer and hasattr(loss_total, 'requires_grad') and loss_total.requires_grad:
                    optimizer.zero_grad()
                    loss_total.backward()
                    if self.max_grad_norm is not None and self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params_to_optimize, self.max_grad_norm)
                    optimizer.step()

                for key, val in loss_dict.items():
                    item_val = val.item() if isinstance(val, torch.Tensor) else float(val)
                    epoch_loss_sum[key] += item_val

            num_batches = len(train_loader)
            avg_losses = {k: v / num_batches for k, v in epoch_loss_sum.items()}

            avg_val_loss = float('inf')
            avg_val_losses = {}
            avg_val_metric = float("inf")
            if val_loader:
                self.model.eval()
                val_loss_sums = defaultdict(float)
                val_weight_sum = 0
                with torch.no_grad():
                    for batch in val_loader:
                        val_loss_dict = {}
                        batch_xs = batch[0].to(self.device)
                        outputs = self.model(xs=batch_xs)
                        val_loss_dict = self.model.compute_loss(outputs, batch_data=batch_xs)

                        batch_weight = int(batch_xs.shape[0])
                        val_weight_sum += batch_weight
                        for key, val in val_loss_dict.items():
                            item_val = val.item() if isinstance(val, torch.Tensor) else float(val)
                            val_loss_sums[key] += item_val * batch_weight

                if val_weight_sum > 0:
                    avg_val_losses = {
                        k: v / val_weight_sum
                        for k, v in val_loss_sums.items()
                    }
                    avg_val_loss = avg_val_losses.get("loss_total", float("inf"))
                    avg_val_metric = self._select_validation_metric(avg_val_losses)

            postfix_str = f"TrL={avg_losses['loss_total']:.4f}"
            if val_loader:
                postfix_str += f"|VL={avg_val_loss:.4f}"
                if self.validation_metric == "mse":
                    postfix_str += f"|VMSE={avg_val_metric:.4f}"
                if 'loss_recon' in avg_losses: postfix_str += f"|Rec={avg_losses['loss_recon']:.4f}"
                if 'loss_latent' in avg_losses: postfix_str += f"|Lat={avg_losses['loss_latent']:.4f}"
            pbar.set_postfix_str(postfix_str)

            epoch_num = epoch + 1
            last_epoch_num = epoch_num
            if use_best_val:
                if avg_val_metric < best_val_metric:
                    best_val_metric = avg_val_metric
                    best_epoch = epoch_num
                    epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), best_path)
                    self.logger.info(
                        "💾 New best validation %s: %.6f at epoch %d，已保存至: %s",
                        self.validation_metric,
                        best_val_metric,
                        best_epoch,
                        best_path,
                    )
                else:
                    epochs_without_improvement += 1
                    self.logger.info(
                        "验证集 %s 未提升: 当前 %.6f，最佳 %.6f at epoch %d，"
                        "early stop %d/%d",
                        self.validation_metric,
                        avg_val_metric,
                        best_val_metric,
                        best_epoch,
                        epochs_without_improvement,
                        self.early_stop_patience,
                    )
                    if (
                        self.early_stop_patience > 0
                        and epochs_without_improvement >= self.early_stop_patience
                    ):
                        self.logger.info(
                            "⏹️ 验证集 %s 连续 %d epochs 未提升，提前停止训练。",
                            self.validation_metric,
                            self.early_stop_patience,
                        )
                        break

            if self._should_evaluate_training_metrics(epoch_num):
                self._evaluate_training_metrics(embeddings_data, ckpt_dir, epoch_num)

        pbar.close()
        self.logger.info("=" * 100)
        self.logger.info(f"🏁 迭代式训练完成 [{self.model_name}]")
        final_train_loss = avg_losses['loss_total']
        self.logger.info(f"📉 Final Epoch({last_epoch_num}) 训练集 Loss: {final_train_loss:.6f}")
        if val_loader:
            self.logger.info(f"📉 Final Epoch 验证集 Loss: {avg_val_loss:.6f}")

        if optimizer:
            torch.save(self.model.state_dict(), final_path)
            self.logger.info(f"💾 Final Epoch 模型已保存至: {final_path}")
        selected_path = final_path
        if use_best_val and os.path.exists(best_path):
            selected_path = best_path
            self.logger.info(
                "🏆 将使用验证集最优 checkpoint 生成 code: %s "
                "(best_epoch=%d, best_%s=%.6f)",
                selected_path,
                best_epoch,
                self.validation_metric,
                best_val_metric,
            )
        else:
            self.logger.info("将使用 final checkpoint 生成 code: %s", selected_path)
        self.logger.info("=" * 100)

        return selected_path if os.path.exists(selected_path) else None

    def _select_validation_metric(self, avg_val_losses):
        if self.validation_metric == "mse":
            if "loss_recon" not in avg_val_losses:
                raise ValueError(
                    "validation_metric=mse requires model.compute_loss to return loss_recon."
                )
            return avg_val_losses["loss_recon"]
        return avg_val_losses["loss_total"]

    def _fit_one_shot(self, embeddings_data, ckpt_dir: str) -> str:
        """处理一次性拟合的模型 (接收 numpy 数据)。"""
        self.logger.info(f"开始 one-shot 拟合 ({self.model_name})...")
        self.model.train()

        if isinstance(embeddings_data, tuple):
            raise ValueError("当前 Trainer 仅支持单文本模态 embedding。")
        fit_device = torch.device("cpu") if getattr(self.model, "fit_on_cpu", False) else self.device
        full_data_tensor = torch.from_numpy(embeddings_data).float().to(fit_device)

        if hasattr(self.model, 'fit') and callable(getattr(self.model, 'fit')):
             self.logger.info("调用 model.fit()...")
             self.model.fit(full_data_tensor)
        else:
             self.logger.info("调用 model forward()...")
             self.model(full_data_tensor)

        # 使用空文件作为信号，因为模型状态可能在内部或不可保存
        fitted_signal_path = os.path.join(ckpt_dir, f"{self.model_name}_fitted.signal")
        os.makedirs(os.path.dirname(fitted_signal_path), exist_ok=True)
        with open(fitted_signal_path, 'w') as f:
            f.write('fitted')

        self.logger.info("=" * 100)
        self.logger.info(f"🏁 One-shot 拟合完成 [{self.model_name}]")
        self.logger.info(f"💾 拟合完成信号已创建: {fitted_signal_path}")
        self.logger.info("=" * 100)
        return fitted_signal_path

    def _should_evaluate_training_metrics(self, epoch_num: int) -> bool:
        if self.eval_metrics_interval <= 0:
            return False
        return epoch_num % self.eval_metrics_interval == 0 or epoch_num == self.epochs

    def _evaluate_training_metrics(self, embeddings_data, ckpt_dir: str, epoch_num: int):
        self.logger.info(
            "开始训练中指标评估 [%s] epoch=%d...",
            self.model_name,
            epoch_num,
        )
        base_codes = self._collect_base_codes(
            embeddings_data,
            batch_size=self.metrics_batch_size,
            desc=f"评估编码 epoch {epoch_num}",
            leave=False,
        )
        codebook_metrics = utils.calculate_codebook_metrics(base_codes, self.model_params)
        prefix_cosine_metrics = utils.calculate_prefix_cosine_metrics(
            base_codes,
            embeddings_data,
            max_items_per_group=self.prefix_cosine_max_items,
            random_seed=self.seed,
        )
        utils.merge_prefix_cosine_into_codebook_metrics(
            codebook_metrics,
            prefix_cosine_metrics,
        )

        utils.log_codebook_metrics(
            codebook_metrics,
            prefix=f"{self.model_name} Epoch {epoch_num} SID 指标",
        )

        metrics = {
            "epoch": int(epoch_num),
            "codebook": codebook_metrics,
        }
        metrics_path = os.path.join(ckpt_dir, f"{self.model_name}_metrics_epoch_{epoch_num}.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info("训练中指标已保存至: %s", metrics_path)

    @torch.no_grad()
    def _collect_base_codes(self, embeddings_data, batch_size: int, desc: str, leave: bool = True):
        if isinstance(embeddings_data, tuple):
            raise ValueError("当前 Trainer 仅支持单文本模态 embedding。")

        was_training = self.model.training
        self.model.eval()
        tensor_data = torch.from_numpy(embeddings_data).float()
        dataset = TensorDataset(tensor_data)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)

        all_codes = []

        try:
            for batch in tqdm(loader, desc=desc, ncols=self.PROGRESS_NCOLS, leave=leave):
                codes = None
                batch_xs = batch[0].to(self.device)
                if hasattr(self.model, "get_codes"):
                    codes = self.model.get_codes(xs=batch_xs)
                elif hasattr(self.model, "encode"):
                    output = self.model.encode(xs=batch_xs)
                    # 兼容不同模型的 encode 输出
                    if isinstance(output, torch.Tensor) and output.dtype in [torch.int, torch.long]:
                        codes = output
                    elif hasattr(self.model, 'quantizer'):
                        z_e = output
                        _, _, codes = self.model.quantizer(z_e)
                    else:
                        codes = output
                else:
                    raise ValueError(f"{self.model_name} 缺少 get_codes/encode 方法")

                if codes is not None and isinstance(codes, torch.Tensor):
                    all_codes.append(codes.detach().cpu().numpy().astype(np.int64))
                else:
                    self.logger.warning("模型未返回有效的 codes tensor，跳过此批次。")
        finally:
            if was_training:
                self.model.train()

        if not all_codes:
            raise RuntimeError("未能生成任何 codes。无法保存码本。")

        return np.vstack(all_codes)

    @torch.no_grad()
    def predict(self, embeddings_data, output_path):
        """生成码本 (接收 numpy 数据)"""
        self.logger.info(f"开始生成码本 ({self.model_name}) -> {output_path}")

        base_codes = self._collect_base_codes(
            embeddings_data,
            batch_size=self.predict_batch_size,
            desc="编码中",
            leave=True,
        )
        self.logger.info(f"基础码本生成完毕，形状: {base_codes.shape}")

        metrics = utils.calculate_codebook_metrics(base_codes, self.model_params)
        utils.log_codebook_metrics(metrics, prefix=f"{self.model_name} 基础 SID 指标")

        metrics_path = os.path.splitext(output_path)[0] + ".metrics.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"码本指标已保存至: {metrics_path}")

        final_codes = base_codes
        if self.has_dup_layer:
            self.logger.info("将构建去重层。")
            if self.codebook_size is None or self.codebook_size <= 0:
                raise ValueError("无法获取有效的 'codebook_size'，无法构建去重层。")
            dedup = utils.build_dedup_layer(base_codes, self.codebook_size)
            final_codes = np.concatenate([base_codes, dedup], axis=1)
            self.logger.info(f"添加去重层后维度: {final_codes.shape}")
        else:
            self.logger.info("配置中 'has_dup_layer' 设为 False，不构建去重层。")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, final_codes)

        json_path = output_path.replace(".npy", ".codebook.json")
        json_dict = {str(i): " ".join([f"<L{l}_{v}>" for l, v in enumerate(row)])
                     for i, row in enumerate(final_codes)}
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)

        self.logger.info(f"✅ 码本保存完成，最终形状: {final_codes.shape}，已保存至: {output_path} (及 .json)")
        return final_codes
