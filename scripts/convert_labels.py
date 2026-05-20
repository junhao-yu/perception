"""
Convert LabelMe JSON annotations to YOLO format for fine-tuning.

Usage:
  python scripts/convert_labels.py \
      --json_dir dataset/cups_bottles/ \
      --classes "cup,bottle"
"""

import argparse, json, os, sys
import numpy as np
from pathlib import Path


def polygon_to_bbox(points):
    """Compute axis-aligned bounding box from polygon/rectangle points."""
    pts = np.array(points)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return x1, y1, x2, y2


def bbox_iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def deduplicate_boxes(boxes, iou_thresh=0.85):
    """Remove duplicate boxes (keep the one with more area)."""
    if len(boxes) < 2:
        return boxes
    kept = []
    used = set()
    for i, bi in enumerate(boxes):
        if i in used:
            continue
        dup_found = False
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            if bbox_iou(bi, boxes[j]) > iou_thresh:
                # Keep larger box
                area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                if area_i >= area_j:
                    used.add(j)
                else:
                    used.add(i)
                    dup_found = True
                    break
        if not dup_found:
            kept.append(bi)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, required=True,
                        help='Directory with .jpg and .json files')
    parser.add_argument('--classes', type=str, required=True,
                        help='Comma-separated class names, e.g. "cup,bottle"')
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"Target classes: {classes}")

    json_dir = Path(args.json_dir)
    json_files = sorted(json_dir.glob('*.json'))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        sys.exit(1)

    # Create labels dir
    labels_dir = json_dir / 'labels'
    os.makedirs(labels_dir, exist_ok=True)

    total_objects = 0
    skipped = 0

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        W, H = data['imageWidth'], data['imageHeight']
        objects = []

        for shape in data['shapes']:
            label = shape['label']
            if label not in class_to_id:
                skipped += 1
                continue

            x1, y1, x2, y2 = polygon_to_bbox(shape['points'])
            cls_id = class_to_id[label]
            objects.append((cls_id, x1, y1, x2, y2))

        # Deduplicate per class
        deduped = []
        for cls_id in class_to_id.values():
            cls_boxes = [o[1:] for o in objects if o[0] == cls_id]
            cls_boxes = deduplicate_boxes(cls_boxes)
            for bbox in cls_boxes:
                deduped.append((cls_id,) + bbox)

        # Write YOLO format: cls_id cx cy w h (normalized)
        stem = jf.stem
        lbl_path = labels_dir / f'{stem}.txt'
        with open(lbl_path, 'w') as f:
            for cls_id, x1, y1, x2, y2 in deduped:
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                f.write(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
                total_objects += 1

    print(f"Processed {len(json_files)} images")
    print(f"Total objects: {total_objects}  (skipped {skipped} non-target)")
    print(f"Labels written to: {labels_dir}/")


if __name__ == '__main__':
    main()
