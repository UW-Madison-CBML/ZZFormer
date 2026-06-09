import os
import yaml
import argparse
import random
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader

from hierarchicalsoftmax import SoftmaxNode
from hierarchicalsoftmax.inference import node_probabilities, greedy_predictions

from model.model_transformer_hierarchical import (
    HierarchicalTransformerClassifier,
    build_classification_tree,
    build_label_to_node_id,
    node_lineage_string,
)

from utils import _get_depth_labels, _compute_sklearn_metrics, _save_confusion_outputs
from data.datalaoder_infer2 import HierarchicalFASTADataset
from data.dataloader_inference import HierarchicalSequenceDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ORDER_TO_SUPERFAMILIES={'DIRS': [],
 'Helitron': [],
 'LINE': ['CR1', 'I', 'Jockey', 'L1', 'R2', 'RTE', 'Rex1'],
 'LTR': ['Bel-Pao', 'Copia', 'Gypsy', 'ERV'],
 'PLE': [],
 'SINE': ['ID', 'SINE1/7SL', 'SINE2/tRNA', 'SINE3/5S'],
 'TIR': ['CACTA', 'MULE', 'PIF', 'TcMar', 'hAT']}
# =============================================================================
# INFERENCE LOOP (unchanged)
# =============================================================================
def run_inference(model, dataloader, classification_tree, thresholds=[0.0], accessions=None, 
                  save_dir=None,
                  run_name="run",     
                  ):
    all_logits = []
    all_target_ids = []

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for tokens, src_key_padding_mask, target_node_ids in dataloader:
            tokens = tokens.to(DEVICE, non_blocking=True)
            src_key_padding_mask = src_key_padding_mask.to(DEVICE, non_blocking=True)
            target_node_ids = target_node_ids.to(DEVICE, non_blocking=True)

            outputs = model(tokens, src_key_padding_mask, target_node_ids)
            loss = outputs["total_loss"]
            logits = outputs["logits"]

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_target_ids.append(target_node_ids.cpu())

    avg_loss = total_loss / len(dataloader)
    all_logits = torch.cat(all_logits, dim=0)
    all_target_ids = torch.cat(all_target_ids, dim=0)

    probs = node_probabilities(all_logits, root=classification_tree)

    results = {"loss": avg_loss}
    total_samples = len(all_target_ids)

    for threshold in thresholds:
        t_str = str(threshold).replace(".", "")

        print(f"\n{'='*60}")
        print(f"  THRESHOLD = {threshold}")
        print(f"{'='*60}")

        pred_nodes = greedy_predictions(
            probs, root=classification_tree, threshold=threshold
        )

        # ---- Order-level metrics (depth 1) ----
        # true_order_raw, pred_order_raw = _get_depth_labels(
        #     pred_nodes, all_target_ids, classification_tree, depth=1
        # )

        true_order_raw, pred_order_raw = _get_depth_labels(
            pred_nodes, all_target_ids, classification_tree, depth=1,
            order_to_superfamilies=ORDER_TO_SUPERFAMILIES
        )



        true_order = []
        pred_order = []
        for t, p in zip(true_order_raw, pred_order_raw):
            if t is not None:
                true_order.append(t)
                pred_order.append(p)

        n_order_skipped = total_samples - len(true_order)
        if n_order_skipped > 0:
            print(f"\n  {n_order_skipped} samples not classified at order level (stopped at root)")

        order_metrics = _compute_sklearn_metrics(
            true_order, pred_order, total_samples,
            level_name=f"Order (threshold={threshold})"
        )

        results[f"order_acc_t{t_str}"]            = order_metrics["accuracy"]
        results[f"order_prec_t{t_str}"]           = order_metrics["precision"]
        results[f"order_rec_t{t_str}"]            = order_metrics["recall"]
        results[f"order_f1_t{t_str}"]             = order_metrics["f1"]
        results[f"order_classified_t{t_str}"]     = order_metrics["n_samples"]
        # NEW: number of ground-truth classes the macro avg is taken over
        results[f"order_gt_classes_t{t_str}"]     = order_metrics["n_gt_classes"]
        # NEW: phantom predictions (predicted classes not in ground truth)
        results[f"order_phantom_classes_t{t_str}"] = order_metrics["phantom_predicted_classes"]
        results[f"order_never_predicted_t{t_str}"] = order_metrics["never_predicted_classes"]

        if save_dir is not None:
            _save_confusion_outputs(
                true_labels=true_order,
                pred_labels=pred_order,
                total_samples=total_samples,
                out_dir=os.path.join(save_dir, "order_confusion_matrices"),
                run_name=run_name.replace(".fasta", ""),
                level_name="order",
                threshold=threshold,
            )

        # ---- Superfamily-level metrics (depth 2) ----
        # true_sf_raw, pred_sf_raw = _get_depth_labels(
        #     pred_nodes, all_target_ids, classification_tree, depth=2
        # )


        true_sf_raw, pred_sf_raw = _get_depth_labels(
            pred_nodes, all_target_ids, classification_tree, depth=2,
            order_to_superfamilies=ORDER_TO_SUPERFAMILIES
        )

        true_sf = []
        pred_sf = []
        for t, p in zip(true_sf_raw, pred_sf_raw):
            if t is not None:
                true_sf.append(t)
                pred_sf.append(p)

        n_sf_skipped = total_samples - len(true_sf)
        print(f"\n  {n_sf_skipped} samples excluded from SF metrics:")
        print(f"    - No superfamily in ground truth (DIRS, Helitron, PLE, etc.)")
        print(f"    - Or prediction stopped at order level due to threshold")

        if len(true_sf) > 0:
            sf_metrics = _compute_sklearn_metrics(
                true_sf, pred_sf, total_samples,
                level_name=f"Superfamily (threshold={threshold})"
            )
        else:
            print(f"\n  No samples classified at superfamily level with threshold={threshold}")
            sf_metrics = {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_samples": 0, "n_total_samples": total_samples,
                "n_gt_classes": 0,
                "never_predicted_classes": [],
                "phantom_predicted_classes": [],
            }

        results[f"sf_acc_t{t_str}"]              = sf_metrics["accuracy"]
        results[f"sf_prec_t{t_str}"]             = sf_metrics["precision"]
        results[f"sf_rec_t{t_str}"]              = sf_metrics["recall"]
        results[f"sf_f1_t{t_str}"]               = sf_metrics["f1"]
        results[f"sf_classified_t{t_str}"]       = sf_metrics.get("n_samples", 0)
        results[f"sf_gt_classes_t{t_str}"]       = sf_metrics.get("n_gt_classes", 0)
        results[f"sf_phantom_classes_t{t_str}"]  = sf_metrics.get("phantom_predicted_classes", [])
        results[f"sf_never_predicted_t{t_str}"]  = sf_metrics.get("never_predicted_classes", [])

        # Save SF confusion matrix files
        if save_dir is not None:
            _save_confusion_outputs(
                true_labels=true_sf,
                pred_labels=pred_sf,
                total_samples=total_samples,
                out_dir=os.path.join(save_dir, "sf_confusion_matrices"),
                run_name=run_name.replace(".fasta", ""),
                level_name="sf",
                threshold=threshold,
            )

    # ---- Per-sample predictions ----
    default_threshold = thresholds[0]
    default_preds = greedy_predictions(
        probs, root=classification_tree, threshold=default_threshold
    )

    per_sample = []
    for i, (pred_node, target_id) in enumerate(zip(default_preds, all_target_ids)):
        target_node = classification_tree.node_list[target_id.item()]
        sample = {
            "idx": i,
            "true_label": node_lineage_string(target_node),
            "pred_label": node_lineage_string(pred_node),
            "pred_depth": pred_node.depth,
            "true_depth": target_node.depth,
        }
        if accessions is not None and i < len(accessions):
            sample["accession"] = accessions[i]
        per_sample.append(sample)

    results["per_sample"] = per_sample

    # Print summary
    print(f"\n{'='*60}")
    print(f"  INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Total samples: {total_samples}")

    t_strs = [str(t).replace(".", "") for t in thresholds]
    header = f"  {'Metric':<30}" + "".join(f" {'t='+str(t):>10}" for t in thresholds)
    print(header)
    print(f"  {'-'*(30 + 11*len(thresholds))}")
    print(f"  {'Order Classified':<30}"   + "".join(f" {results[f'order_classified_t{ts}']:>10}"   for ts in t_strs))
    print(f"  {'Order GT Classes':<30}"   + "".join(f" {results[f'order_gt_classes_t{ts}']:>10}"   for ts in t_strs))
    print(f"  {'Order Accuracy':<30}"     + "".join(f" {results[f'order_acc_t{ts}']:>10.4f}"       for ts in t_strs))
    print(f"  {'Order F1 (macro)':<30}"   + "".join(f" {results[f'order_f1_t{ts}']:>10.4f}"        for ts in t_strs))
    print(f"  {'-'*(30 + 11*len(thresholds))}")
    print(f"  {'SF Classified':<30}"      + "".join(f" {results[f'sf_classified_t{ts}']:>10}"      for ts in t_strs))
    print(f"  {'SF GT Classes':<30}"      + "".join(f" {results[f'sf_gt_classes_t{ts}']:>10}"      for ts in t_strs))
    print(f"  {'SF Accuracy':<30}"        + "".join(f" {results[f'sf_acc_t{ts}']:>10.4f}"          for ts in t_strs))
    print(f"  {'SF F1 (macro)':<30}"      + "".join(f" {results[f'sf_f1_t{ts}']:>10.4f}"           for ts in t_strs))

    return results

# =============================================================================
# MAIN
# =============================================================================

def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    


    vocab = {
        "PAD": 0,
        "A": 1,
        "C": 2,
        "G": 3,
        "T": 4,
        "X": 5
    }
    VOCAB_SIZE = len(vocab)
    vocab_size = VOCAB_SIZE  # PAD, A, C, G, T, X
    PAD_TOKEN = vocab["PAD"]
    ignore_index = -100


    # ================================================================
    # 1. LOAD CHECKPOINT
    # ================================================================
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # ================================================================
    # 2. REBUILD THE CLASSIFICATION TREE
    # ================================================================
    if "classification_tree" in checkpoint:
        print("Loading classification tree from checkpoint")
        classification_tree = checkpoint["classification_tree"]
        classification_tree.set_indexes_if_unset()
    else:
        print("Classification tree not in checkpoint, building from config")
        order_to_superfamilies = config["hierarchy"]
        classification_tree = build_classification_tree(
            order_to_superfamilies,
            label_smoothing=config.get("label_smoothing", 0.0),
            gamma=config.get("gamma", 0.0),
        )

    classification_tree.render(print=True)
    print(f"Tree layer_size (model output dim): {classification_tree.layer_size}")
    print(f"Total nodes: {len(classification_tree.node_list)}")

    label_map = build_label_to_node_id(classification_tree)

    # ================================================================
    # 3. REBUILD MODEL AND LOAD WEIGHTS
    # ================================================================
    model = HierarchicalTransformerClassifier(
        src_vocab_size=checkpoint.get("src_vocab_size", vocab_size),
        classification_tree=classification_tree,
        d_model=checkpoint.get("d_model", config["model"]["d_model"]),
        n_heads=config["model"]["nhead"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        num_layers=config["model"]["num_layers"],
        max_position_embeddings=config["model"]["max_seq_len"],
        pad_token_id=checkpoint.get("pad_token_id", PAD_TOKEN),
        classifier_hidden_dim=config["model"]["classifier_hidden_dim"],
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    print(f"Model loaded from epoch {epoch}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ================================================================
    # 4. LOAD DATASET — FASTA or pickle depending on input
    # ================================================================
    test_path = args.test_dir
    accessions = None

    if test_path.endswith(".pkl") or test_path.endswith(".pickle"):
        # ---- Pickle input ----
        print(f"\nLoading pickle data from: {test_path}")
        with open(test_path, "rb") as f:
            test_seqs = pickle.load(f)
        print(f"Raw sequences loaded: {len(test_seqs)}")

        test_dataset = HierarchicalSequenceDataset(
            test_seqs,
            label_to_id=label_map,
            max_seq_len=config["model"]["max_seq_len"],
            pad_token_id=PAD_TOKEN,
            ignore_index=ignore_index,
        )
    else:
        # ---- FASTA input ----
        print(f"\nLoading FASTA data from: {test_path}")

        test_dataset = HierarchicalFASTADataset(
            fasta_paths=test_path,
            label_to_id=label_map,
            max_seq_len=config["model"]["max_seq_len"],
            pad_token_id=PAD_TOKEN,
            map_rules_str=args.map_rules,
            min_seq_len=args.min_seq_len,
            vocab=vocab,     
            keep_unknown=False,
        )
        
        accessions = test_dataset.accessions

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batchsize"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    # ================================================================
    # 5. RUN INFERENCE
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  RUNNING INFERENCE")
    print(f"{'='*60}")

    # results = run_inference(
    #     model, test_loader, classification_tree,
    #     thresholds=[0.0, 0.0],
    #     accessions=accessions,
    # )

    # ================================================================
    # 6. SAVE RESULTS
    # ================================================================
    save_dir = args.save_dir or os.path.dirname(args.checkpoint)
    os.makedirs(save_dir, exist_ok=True)


    results = run_inference(
        model,
        test_loader,
        classification_tree,
        thresholds=[0.0, 0.0],
        accessions=accessions,
        save_dir=save_dir,          # NEW
        run_name=args.run_name,     # NEW
    )

    # Save metrics to text file
    metrics_path = os.path.join(save_dir, f"inference_metrics_{args.run_name}_nothresh.txt")
    with open(metrics_path, "a") as f:
        f.write(f"{'='*100}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test data:  {args.test_dir}\n")
        f.write(f"Map rules:  {args.map_rules}\n")
        f.write(f"Epoch:      {epoch}\n")
        f.write(f"Samples:    {len(test_dataset)}\n")
        f.write(f"Loss:       {results['loss']:.4f}\n")
        f.write(f"{'='*100}\n\n")

        for threshold in [0.0]:
            t_str = str(threshold).replace(".", "")
            f.write(f"Threshold {threshold}\n")
            f.write(f"{'':>10} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Accuracy':>12} {'Classified':>12} {'Active Cls':>12}\n")
            f.write(
                f"{'Order':>10} "
                f"{results[f'order_prec_t{t_str}']:>12.4f} "
                f"{results[f'order_rec_t{t_str}']:>12.4f} "
                f"{results[f'order_f1_t{t_str}']:>12.4f} "
                f"{results[f'order_acc_t{t_str}']:>12.4f} "
                f"{results[f'order_classified_t{t_str}']:>12}\n"
                # f"{results[f'order_active_classes_t{t_str}']:>12}\n"
            )
            f.write(
                f"{'SF':>10} "
                f"{results[f'sf_prec_t{t_str}']:>12.4f} "
                f"{results[f'sf_rec_t{t_str}']:>12.4f} "
                f"{results[f'sf_f1_t{t_str}']:>12.4f} "
                f"{results[f'sf_acc_t{t_str}']:>12.4f} "
                f"{results[f'sf_classified_t{t_str}']:>12}\n"
                # f"{results[f'sf_active_classes_t{t_str}']:>12}\n"
            )
            f.write(f"\n")

    print(f"\nMetrics saved to: {metrics_path}")

    # Save per-sample predictions to pickle
    predictions_path = os.path.join(save_dir, f"inference_predictions_{args.run_name}.pkl")
    with open(predictions_path, "wb") as f:
        pickle.dump(results["per_sample"], f)

    print(f"Per-sample predictions saved to: {predictions_path}")

    # Print example predictions
    print(f"\n  Example predictions (first 20):")
    header = f"  {'Accession':<30} {'True Label':<30} {'Predicted Label':<30} {'Depth':>6}"
    print(header)
    print(f"  {'-'*96}")
    
    for sample in results["per_sample"][:20]:
        acc = sample.get("accession", f"sample_{sample['idx']}")
        print(
            f"  {acc:<30} "
            f"{sample['true_label']:<30} "
            f"{sample['pred_label']:<30} "
            f"{sample['pred_depth']:>6}"
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained hierarchical model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint .pt file")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to test data: .pkl pickle file OR .fa/.fasta FASTA file or directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save inference results")
    parser.add_argument("--run_name", type=str, default="test", help="Name for output files")
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument("--map_rules", type=str, default="",
                        help="Comma-separated label remapping rules, e.g. "
                             "'/I-Jockey=/I,TcMar-Pogo=TcMar,Helitron=RC'")
    parser.add_argument("--min_seq_len", type=int, default=0,
                        help="Minimum sequence length for FASTA input (skip shorter)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)