# utils_coco.py

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

###############################################################################
# 1) Load COCO categories and build a cat_id -> (class_id, class_name) mapping
###############################################################################

def load_coco_categories(json_file):
    """
    Load the 'categories' section from a COCO-style annotation file
    (e.g., 'instances_val2017.json') and build two dicts:
      - cat_id_to_name: {cat_id -> category_name}
      - cat_id_to_class_id: {cat_id -> 0-based class index}

    We will:
      1) Collect all category IDs from 'categories'.
      2) Sort them in ascending order.
      3) Assign a consecutive class_id starting from 0.

    This ensures we don't rely on 'cat_id-1', which can be incorrect if COCO
    category IDs are not strictly consecutive or skip certain values.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract the list of categories
    categories = data["categories"]

    # Build {cat_id -> cat_name}
    cat_id_to_name = {}
    for cat in categories:
        cid = cat["id"]
        cname = cat["name"]
        cat_id_to_name[cid] = cname

    # Sort all cat_ids and build a stable mapping {cat_id -> class_id}
    sorted_cat_ids = sorted(cat_id_to_name.keys())
    cat_id_to_class_id = {}
    for idx, cid in enumerate(sorted_cat_ids):
        cat_id_to_class_id[cid] = idx

    return cat_id_to_name, cat_id_to_class_id


###############################################################################
# 2) Load COCO annotations into a unified [image_id, class_id, bbox] format
###############################################################################

def load_coco_annotations(json_file, cat_id_to_class_id):
    """
    Parse COCO-style annotations (e.g., 'instances_val2017.json') into a list of dict:
      {
        "image_id": int,
        "class_id": int,      # mapped using cat_id_to_class_id
        "bbox": [x1, y1, x2, y2]
      }

    COCO bounding boxes are [x, y, w, h]. We convert them to [x1, y1, x2, y2].
    We skip any annotation with iscrowd=1.

    Args:
      json_file (str): Path to the COCO annotation JSON.
      cat_id_to_class_id (dict): mapping from cat_id -> class_id (0-based).

    Returns:
      A list of ground truth dicts.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]
    ground_truths = []

    for ann in annotations:
        if ann.get("iscrowd", 0) == 1:
            continue

        cat_id = ann["category_id"]
        if cat_id not in cat_id_to_class_id:
            # skip categories we don't have in our mapping
            continue

        class_id = cat_id_to_class_id[cat_id]

        x, y, w, h = ann["bbox"]
        x2 = x + w
        y2 = y + h

        gt_dict = {
            "image_id": ann["image_id"],
            "class_id": class_id,
            "bbox": [x, y, x2, y2]
        }
        ground_truths.append(gt_dict)

    return ground_truths


###############################################################################
# 3) Visualize GT boxes (green) and predicted boxes (red) on a single image
###############################################################################

def visualize_detections(
    image_path,
    gt_list,
    pred_list,
    class_names,
    score_thr=0.3
):
    """
    Draw ground-truth boxes (in green) and predicted boxes (in red) on the image.

    Args:
      image_path (str): path to the image, e.g. "val2017/000000123456.jpg"
      gt_list (list): ground truth boxes for this image, each item:
         {
           "class_id": int,
           "bbox": [x1, y1, x2, y2]
         }
      pred_list (list): predicted boxes for this image, each item:
         {
           "class_id": int,
           "bbox": [x1, y1, x2, y2],
           "score": float
         }
      class_names (list or dict): index -> class name, so class_names[class_id]
      score_thr (float): only draw predictions above this confidence.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw ground-truth boxes in green
    for gt in gt_list:
        cid = gt["class_id"]
        x1, y1, x2, y2 = gt["bbox"]
        cname = class_names[cid] if cid < len(class_names) else f"cls_{cid}"
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 10), f"GT: {cname}", fill="green")

    # Draw predicted boxes in red
    for pd in pred_list:
        if pd["score"] < score_thr:
            continue
        cid = pd["class_id"]
        x1, y1, x2, y2 = pd["bbox"]
        cname = class_names[cid] if cid < len(class_names) else f"cls_{cid}"
        conf = pd["score"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y2 + 10), f"{cname} {conf:.2f}", fill="red")

    img.show()


###############################################################################
# 4) Compute mAP (single IoU=0.5, 11-point VOC-style)
###############################################################################

def compute_mAP(preds, gts, num_classes=80, iou_thresh=0.5):
    """
    Compute mean Average Precision (mAP) at IoU=0.5 using 11-point interpolation
    (similar to Pascal VOC 2007).

    Args:
      preds (list): each item is a dict:
        {
          "image_id": int,
          "class_id": int,
          "bbox": [x1, y1, x2, y2],
          "score": float
        }
      gts (list): each item is a dict:
        {
          "image_id": int,
          "class_id": int,
          "bbox": [x1, y1, x2, y2]
        }
      num_classes (int): total classes, default=80 for COCO.
      iou_thresh (float): IoU threshold to consider a detection True Positive.

    Returns:
      float: the mean AP over all classes that have at least one ground-truth box.
    """
    # Group ground truths by (image_id, class_id)
    gt_dict = {}
    for gt in gts:
        key = (gt["image_id"], gt["class_id"])
        if key not in gt_dict:
            gt_dict[key] = []
        gt_dict[key].append(gt["bbox"])

    AP_per_class = []

    for c in range(num_classes):
        # Gather predictions of class c
        class_preds = [p for p in preds if p["class_id"] == c]
        # Sort by confidence descending
        class_preds.sort(key=lambda x: x["score"], reverse=True)

        # Count total GT boxes for class c
        total_gt_c = 0
        for (img_id, cls_id), bboxes in gt_dict.items():
            if cls_id == c:
                total_gt_c += len(bboxes)
        if total_gt_c == 0:
            # no ground-truth for this class
            continue

        matched = {}  # (image_id, gt_idx) -> bool
        TP = np.zeros(len(class_preds), dtype=np.float32)
        FP = np.zeros(len(class_preds), dtype=np.float32)

        for i, pred in enumerate(class_preds):
            pred_box = pred["bbox"]
            image_id = pred["image_id"]
            gt_bboxes = gt_dict.get((image_id, c), [])

            # find best iou among this image's GT boxes
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_bboxes):
                iou_val = iou(pred_box, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh:
                match_key = (image_id, best_gt_idx)
                if match_key not in matched:
                    TP[i] = 1.0
                    matched[match_key] = True
                else:
                    FP[i] = 1.0
            else:
                FP[i] = 1.0

        # compute precision/recall
        cum_TP = np.cumsum(TP)
        cum_FP = np.cumsum(FP)

        recalls = cum_TP / float(total_gt_c)
        precisions = cum_TP / np.maximum(cum_TP + cum_FP, np.finfo(np.float32).eps)

        # 11-point interpolation
        recall_levels = np.linspace(0.0, 1.0, 11)
        precisions_at_recall = []
        for rl in recall_levels:
            idxs = np.where(recalls >= rl)[0]
            if idxs.size > 0:
                precisions_at_recall.append(np.max(precisions[idxs]))
            else:
                precisions_at_recall.append(0.0)
        AP_c = np.mean(precisions_at_recall)
        AP_per_class.append(AP_c)

    if len(AP_per_class) == 0:
        return 0.0
    return float(np.mean(AP_per_class))


###############################################################################
# Helper: IoU function used inside compute_mAP
###############################################################################

def iou(box1, box2):
    """
    Intersection over Union for two boxes [x1, y1, x2, y2].
    Returns a float in [0,1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


import os
import json
import torch
import torchvision
from math import ceil
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

###############################################################################
# Pre-trained Faster R-CNN model from torchvision
###############################################################################
###############################################################################
# Chunk-based inference: process images in chunks, parse results, store JSON
###############################################################################

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

def parse_frcnn_result(img_path, output, image_id, score_thr=0.0):
    """
    Parse one FRCNN output dict -> a list of detection dicts in:
      {
        "image_id": int,
        "class_id": int,     # same as label in [1..90]
        "bbox": [x1, y1, x2, y2],
        "score": float
      }
    'score_thr' can filter out very low confidences if desired.
    """
    boxes = output["boxes"].cpu().detach().numpy()  # shape (N,4)
    labels = output["labels"].cpu().detach().numpy()  # shape (N,)
    scores = output["scores"].cpu().detach().numpy()  # shape (N,)

    predictions = []
    for i in range(len(boxes)):
        sc = float(scores[i])
        if sc < score_thr:
            continue
        lab = int(labels[i])  # in [1..90], 0=background
        x1, y1, x2, y2 = boxes[i].tolist()

        predictions.append({
            "image_id": image_id,
            "class_id": lab,     # keep the label as is
            "bbox": [x1, y1, x2, y2],
            "score": sc
        })
    return predictions

def gather_frcnn_predictions_in_chunks(model, image_paths, chunk_size=500, batch_size=1, score_thr=0.0, output_dir="frcnn_pred_chunks"):
    """
    1) Process images in chunks.
    2) For each chunk, do FRCNN inference (batch size set by 'batch_size').
    3) Parse outputs -> store in JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_files = []

    # We'll do a naive "batch=1" approach here for clarity. You can adapt to larger if needed.
    for chunk_idx, sub_paths in enumerate(chunk_list(image_paths, chunk_size)):
        print(f"\n=== Processing chunk {chunk_idx+1} with {len(sub_paths)} images ===")
        chunk_predictions = []

        for img_path in sub_paths:
            # Derive image_id from filename if your val2017 is "000000123456.jpg" => 123456
            filename = os.path.basename(img_path)
            image_id = int(os.path.splitext(filename)[0])

            # Load the image as a PIL or tensor
            img_pil = Image.open(img_path).convert("RGB")
            # Typically for FRCNN you do the transform manually if you want batch>1,
            # but let's do the simple approach:
            img_tensor = torchvision.transforms.functional.to_tensor(img_pil).unsqueeze(0).to(next(model.parameters()).device)

            with torch.no_grad():
                output = model(img_tensor)[0]  # one image -> one dict
            preds_for_image = parse_frcnn_result(img_path, output, image_id, score_thr=score_thr)
            chunk_predictions.extend(preds_for_image)

        out_file = os.path.join(output_dir, f"predictions_chunk_{chunk_idx}.json")
        with open(out_file, "w") as f:
            json.dump(chunk_predictions, f)
        chunk_files.append(out_file)
        print(f"Chunk {chunk_idx+1} saved to {out_file}.")

    return chunk_files

def load_chunk_predictions(chunk_files):
    """
    Combine multiple chunk JSON files into one list of predictions.
    """
    all_preds = []
    for cf in chunk_files:
        with open(cf, "r") as f:
            cdata = json.load(f)
        all_preds.extend(cdata)
    return all_preds


def load_coco_annotations_withoutmapping(json_file, max_cat=80):
    """
    Parse 'instances_val2017.json' but do NOT map category_id -> cat_id-1.
    We keep category_id as 'class_id'. We also skip iscrowd=1 and cat_id>max_cat.

    Each annotation -> {
      "image_id": int,
      "class_id": int,   # same as category_id
      "bbox": [x1, y1, x2, y2]
    }
    """
    import json
    with open(json_file, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]
    ground_truths = []
    for ann in annotations:
        if ann.get("iscrowd", 0) == 1:
            continue
        cat_id = ann["category_id"]
        # skip if cat_id > max_cat or cat_id < 1
        if cat_id < 1 or cat_id > max_cat:
            continue

        x, y, w, h = ann["bbox"]
        x2 = x + w
        y2 = y + h

        ground_truths.append({
            "image_id": ann["image_id"],
            "class_id": cat_id,
            "bbox": [x, y, x2, y2]
        })

    return ground_truths
