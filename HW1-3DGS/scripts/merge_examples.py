#!/usr/bin/env python3
"""
Merge 10 images in report/images/3.1/examples into one image with 5 rows.
Each row: left = compare image, right = heatmap image.

Heuristics:
- If filenames contain 'heat' or 'heatmap' they are treated as heatmaps.
- If that classification yields 5/5, pair by sorted order within each group.
- Otherwise pair by sorted file order: (0,1), (2,3), ...

Output: report/images/3.1/examples/merged_examples.png
"""
import os
from PIL import Image

INPUT_DIR = 'report/images/3.1/examples'
OUT_PATH = os.path.join(INPUT_DIR, 'merged_examples.png')
PAD = 6

def list_images(folder):
    exts = ('.png', '.jpg', '.jpeg')
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    return files

def classify_pairs(files):
    # classify by heuristic first
    heat = [f for f in files if 'heat' in f.lower() or 'heatmap' in f.lower()]
    comp = [f for f in files if f not in heat]

    def stem_name(fn):
        name = os.path.splitext(fn)[0]
        # remove common suffixes
        for suf in ('_heatmap', '-heatmap', '_heat', '-heat', '_hm', '-hm', '_compare', '-compare', '_cmp'):
            if name.lower().endswith(suf):
                return name[: -len(suf)]
        # if starts with numeric id like 00000_xxx, take the numeric prefix
        parts = name.split('_')
        if parts[0].isdigit():
            return parts[0]
        # fallback: take text before first non-alnum char
        return name

    # attempt pairing by stem (best effort)
    heat_map = {stem_name(f): f for f in heat}
    comp_map = {stem_name(f): f for f in comp}
    common = sorted(k for k in comp_map.keys() if k in heat_map)
    if len(common) == 5:
        return [(comp_map[k], heat_map[k]) for k in common]

    # if classification by 'heat' keyword yields equal groups, pair by sorted order
    if len(heat) == 5 and len(comp) == 5:
        return list(zip(sorted(comp), sorted(heat)))

    # fallback: split into first/second halves (ensures 5 rows)
    if len(files) >= 10:
        lefts = files[:5]
        rights = files[5:10]
        return list(zip(lefts, rights))

    raise SystemExit(f"Need at least 10 images in {INPUT_DIR} (found {len(files)})")

def open_and_resize(path, target_h):
    im = Image.open(path).convert('RGBA')
    w, h = im.size
    if h != target_h:
        new_w = int(w * (target_h / h))
        im = im.resize((new_w, target_h), Image.LANCZOS)
    return im

def main():
    if not os.path.isdir(INPUT_DIR):
        raise SystemExit(f"Missing folder: {INPUT_DIR}")
    files = list_images(INPUT_DIR)
    if len(files) < 10:
        raise SystemExit(f"Expected 10 images in {INPUT_DIR}, found {len(files)}")
    pairs = classify_pairs(files)

    # compute a common target height: use the minimum height among all images to avoid upscaling
    heights = []
    for a, b in pairs:
        pa = os.path.join(INPUT_DIR, a)
        pb = os.path.join(INPUT_DIR, b)
        with Image.open(pa) as ia:
            heights.append(ia.size[1])
        with Image.open(pb) as ib:
            heights.append(ib.size[1])
    target_h = min(heights)

    left_widths = []
    right_widths = []
    rows = []
    for left, right in pairs:
        la = open_and_resize(os.path.join(INPUT_DIR, left), target_h)
        rb = open_and_resize(os.path.join(INPUT_DIR, right), target_h)
        left_widths.append(la.size[0])
        right_widths.append(rb.size[0])
        rows.append((la, rb))

    col_left_w = max(left_widths)
    col_right_w = max(right_widths)
    row_h = target_h

    total_w = col_left_w + PAD + col_right_w
    total_h = len(rows) * row_h + (len(rows) - 1) * PAD

    out = Image.new('RGBA', (total_w, total_h), (255,255,255,255))

    y = 0
    for la, rb in rows:
        # pad left and right images to column widths and paste
        left_canvas = Image.new('RGBA', (col_left_w, row_h), (255,255,255,255))
        right_canvas = Image.new('RGBA', (col_right_w, row_h), (255,255,255,255))
        left_canvas.paste(la, (0, (row_h - la.size[1])//2), la)
        right_canvas.paste(rb, (0, (row_h - rb.size[1])//2), rb)

        out.paste(left_canvas, (0, y), left_canvas)
        out.paste(right_canvas, (col_left_w + PAD, y), right_canvas)
        y += row_h + PAD

    # save as PNG (flatten to RGB)
    out.convert('RGB').save(OUT_PATH, pnginfo=None)
    print('WROTE', OUT_PATH)

if __name__ == '__main__':
    main()
