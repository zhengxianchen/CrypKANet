import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, matthews_corrcoef, confusion_matrix, \
    accuracy_score, precision_score, recall_score


def evaluate_and_report_metrics(y_true_tensor, y_scores_tensor, optimize_for='f1', plot_curves=False):
    """
    评估模型性能并报告多种指标，可选择寻找最佳阈值和绘制曲线。

    Args:
        y_true_tensor (torch.Tensor): 真实标签的 PyTorch Tensor (例如 test_label_fc)。
        y_scores_tensor (torch.Tensor): 模型预测分数的 PyTorch Tensor (例如 test_score)。
        optimize_for (str): 寻找最佳阈值的优化目标，可以是 'f1' (F1-score) 或 'mcc' (MCC)。
                            默认为 'f1'。
        plot_curves (bool): 是否绘制 ROC 和 Precision-Recall 曲线。默认为 False。

    Returns:
        dict: 包含所有计算指标的字典，以及最佳阈值。
    """

    # 确保输入是 NumPy 数组，并且 y_true 是整数类型 (0或1)
    # y_true = np.array(y_true_tensor.detach().cpu()).astype(int)
    # y_scores = np.array(y_scores_tensor.detach().cpu())
    y_true = y_true_tensor
    y_scores = y_scores_tensor

    # --- 1. 计算与阈值无关的指标 (AUROC, AUPR) ---
    # 计算 ROC 曲线
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc_ = auc(fpr, tpr)

    # 计算 Precision-Recall 曲线
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
    aupr = auc(recall_curve, precision_curve)

    # --- 2. 寻找最佳阈值及在该阈值下的指标 ---
    best_metric_value = -1.0  # 用于比较 F1 或 MCC
    best_threshold = 0.5  # 默认一个初始阈值
    best_metrics_at_threshold = {}

    # 使用 roc_curve 提供的阈值进行搜索，这些阈值覆盖了所有可能的分类点
    # 添加 0.0 和 1.0 以确保覆盖所有极端情况
    thresholds_for_search = np.unique(np.concatenate(([0.0], roc_thresholds, [1.0])))

    for threshold in thresholds_for_search:
        # 将预测分数二值化
        binary_pred = (y_scores >= threshold).astype(int)

        # 检查二值化预测是否只有一类，这会导致某些指标计算失败
        if len(np.unique(binary_pred)) < 2:
            # 如果所有预测都是0或1，则MCC和F1可能为0或NaN，此处给一个低分
            current_metric_value = -2.0  # 确保不会被选为最佳
        else:
            # 计算当前阈值下的 F1-score 和 MCC
            current_f1 = f1_score(y_true, binary_pred, zero_division=0)
            current_mcc = matthews_corrcoef(y_true, binary_pred)

            if optimize_for == 'f1':
                current_metric_value = current_f1
            elif optimize_for == 'mcc':
                current_metric_value = current_mcc
            else:
                raise ValueError("`optimize_for` must be 'f1' or 'mcc'.")

        # 更新最佳阈值和最佳指标
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_threshold = threshold

            # 在找到更好的阈值时，计算并保存所有相关指标
            tn, fp, fn, tp = confusion_matrix(y_true, binary_pred).ravel()

            # 避免除以零
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            best_metrics_at_threshold = {
                "accuracy": accuracy_score(y_true, binary_pred),
                "precision": precision_score(y_true, binary_pred, zero_division=0),
                "recall": recall_score(y_true, binary_pred, zero_division=0),
                "f1": current_f1,
                "mcc": current_mcc,
                "spe": specificity,  # 使用 'spe' 保持与你原始输出一致
                "threshold": best_threshold
            }

    # 如果 best_metrics_at_threshold 仍然为空 (例如，所有预测都一样)，则使用默认值
    if not best_metrics_at_threshold:
        binary_pred_default = (y_scores >= 0.5).astype(int)  # 使用默认阈值0.5
        tn, fp, fn, tp = confusion_matrix(y_true, binary_pred_default).ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        best_metrics_at_threshold = {
            "accuracy": accuracy_score(y_true, binary_pred_default),
            "precision": precision_score(y_true, binary_pred_default, zero_division=0),
            "recall": recall_score(y_true, binary_pred_default, zero_division=0),
            "f1": f1_score(y_true, binary_pred_default, zero_division=0),
            "mcc": matthews_corrcoef(y_true, binary_pred_default),
            "spe": specificity,
            "threshold": 0.5  # 默认阈值
        }

        # 显示性能指标在图上 (可选)
        metrics_str = (f"Best Threshold: {best_metrics_at_threshold['threshold']:.4f}\n"
                       f"Accuracy: {best_metrics_at_threshold['accuracy']:.4f}\n"
                       f"Precision: {best_metrics_at_threshold['precision']:.4f}\n"
                       f"Recall: {best_metrics_at_threshold['recall']:.4f}\n"
                       f"Specificity: {best_metrics_at_threshold['spe']:.4f}\n"
                       f"MCC: {best_metrics_at_threshold['mcc']:.4f}\n"
                       f"F1 Score: {best_metrics_at_threshold['f1']:.4f}")
        # 尝试将文本放置在图的合适位置，可能需要根据实际图的范围调整
        plt.figure(2)  # 在PR曲线上添加文本
        plt.text(0.6, 0.2, metrics_str, bbox=dict(facecolor='white', alpha=0.7), fontsize=9)

        plt.show()  # 显示所有图

    # --- 4. 打印并返回结果 ---
    print(f"\n--- Model Evaluation Results (Optimized for {optimize_for}) ---")
    print(f"AUROC: {roc_auc_:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"Best Threshold (for {optimize_for}): {best_metrics_at_threshold['threshold']:.4f}")
    print(f"Accuracy: {best_metrics_at_threshold['accuracy']:.4f}")
    print(f"Precision: {best_metrics_at_threshold['precision']:.4f}")
    print(f"Recall: {best_metrics_at_threshold['recall']:.4f}")
    print(f"Specificity (SPE): {best_metrics_at_threshold['spe']:.4f}")
    print(f"MCC: {best_metrics_at_threshold['mcc']:.4f}")
    print(f"F1 Score: {best_metrics_at_threshold['f1']:.4f}")
    print("-------------------------------------------------")

    final_results = {
        'accuracy': best_metrics_at_threshold['accuracy'],
        'spe': best_metrics_at_threshold['spe'],
        'precision': best_metrics_at_threshold['precision'],
        'recall': best_metrics_at_threshold['recall'],
        'f1': best_metrics_at_threshold['f1'],
        'mcc': best_metrics_at_threshold['mcc'],
        'auc': roc_auc_,
        'pr_auc': aupr,
        'best_threshold': best_metrics_at_threshold['threshold']
    }
    return final_results