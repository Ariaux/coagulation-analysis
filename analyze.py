#!/usr/bin/env python3
"""
Coagulation Analysis — 直接分析已裁切好的方格图片
ImageJ workflow: 8-bit → Invert → Measure Mean (整张图)

用法:
  python3 analyze.py 图片.jpg
  python3 analyze.py 文件夹/ --watch   # 拖图自动分析
  python3 analyze.py 文件夹/ --batch   # 批量分析
"""
import sys, os, json, glob, time, argparse
import numpy as np
import cv2

def analyze_image(image_path):
    """ImageJ: 8-bit → invert → measure whole image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    # ImageJ-exact grayscale: 0.299*R + 0.587*G + 0.114*B
    b, g, r = img[:,:,0].astype(np.float32), img[:,:,1].astype(np.float32), img[:,:,2].astype(np.float32)
    gray = np.clip(0.114*b + 0.587*g + 0.299*r, 0, 255).astype(np.uint8)
    inverted = cv2.bitwise_not(gray)

    mean_val = float(np.mean(inverted))
    median_val = float(np.median(inverted))
    std_val = float(np.std(inverted))
    return {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "min": int(np.min(inverted)),
        "max": int(np.max(inverted)),
    }

def main():
    parser = argparse.ArgumentParser(description="Coagulation: direct square analysis (no ROI)")
    parser.add_argument("path", help="Image file or folder")
    parser.add_argument("--batch", action="store_true", help="Process all images in folder")
    parser.add_argument("--watch", action="store_true", help="Watch folder, auto-process new images")
    args = parser.parse_args()

    # Collect images
    if os.path.isdir(args.path):
        folder = os.path.abspath(args.path)
        if args.watch:
            print(f"Watching: {folder}/")
            print("Drop cropped square images → auto-analyze. Ctrl+C to stop.\n")
            processed = set()
            while True:
                current = set(glob.glob(os.path.join(folder, "*.jpg")) +
                             glob.glob(os.path.join(folder, "*.JPG")) +
                             glob.glob(os.path.join(folder, "*.png")) +
                             glob.glob(os.path.join(folder, "*.PNG")))
                for p in sorted(current - processed):
                    r = analyze_image(p)
                    if r:
                        print(f"[{os.path.basename(p)}] mean={r['mean']:.1f} median={r['median']:.1f} std={r['std']:.1f}")
                    processed.add(p)
                time.sleep(1)
        elif args.batch:
            images = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                           glob.glob(os.path.join(folder, "*.JPG")) +
                           glob.glob(os.path.join(folder, "*.png")) +
                           glob.glob(os.path.join(folder, "*.PNG")))
            if not images:
                print("No images found")
                return
            all_results = []
            print(f"{'File':<40s} {'Mean':>8s} {'Median':>8s} {'Std':>8s}")
            print("-" * 66)
            for p in images:
                r = analyze_image(p)
                if r:
                    name = os.path.basename(p)[:38]
                    print(f"{name:<40s} {r['mean']:>8.1f} {r['median']:>8.1f} {r['std']:>8.1f}")
                    all_results.append({"file": name, **r})
            # Save CSV
            csv_path = os.path.join(folder, "results.csv")
            with open(csv_path, "w") as f:
                f.write("file,mean,median,std,min,max\n")
                for r in all_results:
                    f.write(f"{r['file']},{r['mean']},{r['median']},{r['std']},{r['min']},{r['max']}\n")
            print(f"\nSaved: {csv_path}")
        else:
            print("Use --batch or --watch with a folder")
    else:
        r = analyze_image(args.path)
        if r:
            print(f"Mean={r['mean']:.1f} Median={r['median']:.1f} Std={r['std']:.1f}")

if __name__ == "__main__":
    main()
