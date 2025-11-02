#!/usr/bin/env python3
import os
import argparse
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import csv
import math

def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.asarray(img).astype(np.float32) / 255.0

def compare(render_dir, gt_dir):
    files = sorted([f for f in os.listdir(gt_dir)])
    rows = []
    for fname in files:
        rpath = os.path.join(render_dir, fname)
        gpath = os.path.join(gt_dir, fname)
        r = load_image(rpath)
        g = load_image(gpath)
        psnr_val = 20.0 * math.log10(1.0) - 10.0 * math.log10(mean_squared_error(g, r))
        ssim_val = ssim(r, g, data_range=1.0, channel_axis=2)
        rows.append((fname, psnr_val, ssim_val))
    return rows

def main(render_dir, gt_dir, out_dir):
    rows = compare(render_dir, gt_dir)
    if len(rows) == 0:
        print('No matching images')
        return
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, 'w') as cf:
        w = csv.writer(cf)
        for r in rows:
            w.writerow(r)
            print(f"{r[0]}: PSNR={r[1]:.4f}, SSIM={r[2]:.4f}")
    psnrs = np.array([r[1] for r in rows], dtype=np.float64)
    ss = np.array([r[2] for r in rows], dtype=np.float64)
    print('PSNR mean/std:', float(np.nanmean(psnrs)), float(np.nanstd(psnrs)))
    print('SSIM mean/std:', float(np.nanmean(ss)), float(np.nanstd(ss)))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--render', required=True)
    p.add_argument('--gt', default='assets/gt')
    p.add_argument('--out', default='report/images/3.1/compare.csv')
    args = p.parse_args()
    main(args.render, args.gt, args.out)
