"""
Safe stratified redistributor for chest X-ray datasets.

Features:
- Computes overall class ratios and redistributes files to train/val/test with target ratios (default 75/15/10).
- Performs per-class splits to maintain class balance (stratified by class naturally since we split per-class).
- Copies files to a new output directory by default to avoid data loss. Optionally move files inplace.
- Supports dry-run mode and logging of actions and errors.
- Handles filename collisions by appending a suffix.

Usage examples:
python redistribute_safe.py --source "C:\\Users\\PujaSaha\\Documents\\chest_xray_pneumonia" --dry-run
python redistribute_safe.py --source "..." --output "..._redistributed" --move

"""
import argparse
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_image_files(dirpath: Path) -> List[Path]:
    if not dirpath.exists():
        return []
    return [p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def unique_dest(dest: Path) -> Path:
    """If dest exists, append a numeric suffix before the extension to avoid collision."""
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def stratified_split_per_class(all_images: List[Path], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split a list of file paths into train/val/test preserving counts as closely as possible."""
    n = len(all_images)
    if n == 0:
        return [], [], []
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    # Compute counts
    train_count = int(n * train_ratio)
    val_count = int(n * val_ratio)
    test_count = n - train_count - val_count
    train_idx = indices[:train_count]
    val_idx = indices[train_count:train_count + val_count]
    test_idx = indices[train_count + val_count:]
    train_files = [all_images[i] for i in train_idx]
    val_files = [all_images[i] for i in val_idx]
    test_files = [all_images[i] for i in test_idx]
    return train_files, val_files, test_files


def collect_all_images(source: Path, classes: List[str]) -> dict:
    data = {}
    for cls in classes:
        images = []
        for split in ['train', 'val', 'test']:
            p = source / split / cls
            images.extend(get_image_files(p))
        data[cls] = images
    return data


def copy_or_move_files(files: List[Path], dest_dir: Path, dry_run: bool, move: bool, logger: List[str]):
    ensure_dir(dest_dir)
    for f in files:
        try:
            dest = dest_dir / f.name
            dest = unique_dest(dest)
            if dry_run:
                logger.append(f"DRY-RUN: would {'move' if move else 'copy'} {f} -> {dest}")
            else:
                if move:
                    shutil.move(str(f), str(dest))
                else:
                    shutil.copy2(str(f), str(dest))
                logger.append(f"OK: {'moved' if move else 'copied'} {f} -> {dest}")
        except Exception as e:
            logger.append(f"ERROR: {f} -> {dest_dir} : {e}")


def analyze_counts(data: dict) -> Tuple[int, dict]:
    totals = {cls: len(imgs) for cls, imgs in data.items()}
    grand_total = sum(totals.values())
    return grand_total, totals


def main():
    parser = argparse.ArgumentParser(description='Safe stratified redistributor (copy by default)')
    parser.add_argument('--source', required=True, help='Source dataset path (contains train/val/test subfolders)')
    parser.add_argument('--output', required=False, help='Output base path (defaults to <source>_redistributed)')
    parser.add_argument('--train', type=float, default=0.75, help='Train ratio')
    parser.add_argument('--val', type=float, default=0.15, help='Val ratio')
    parser.add_argument('--test', type=float, default=0.10, help='Test ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Do not copy/move, only simulate')
    parser.add_argument('--move', action='store_true', help='Move files instead of copying (destructive)')
    parser.add_argument('--classes', nargs='+', default=['NORMAL', 'PNEUMONIA'], help='List of class folder names')
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        print(f"Source path does not exist: {source}")
        return

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        output = source.parent / (source.name + '_redistributed')

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print('Train+Val+Test ratios must sum to 1.0')
        return

    # Collect images
    classes = args.classes
    data = collect_all_images(source, classes)
    grand_total, totals = analyze_counts(data)

    print(f"Found total {grand_total} images across {len(classes)} classes")
    for cls in classes:
        print(f"  {cls}: {totals.get(cls,0)} images")

    # Compute per-class splits
    redistributed = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}

    for cls in classes:
        imgs = data.get(cls, [])
        train_files, val_files, test_files = stratified_split_per_class(
            imgs, args.train, args.val, args.test, args.seed
        )
        redistributed['train'][cls] = train_files
        redistributed['val'][cls] = val_files
        redistributed['test'][cls] = test_files
        print(f"{cls}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Log and perform copy/move
    actions_log = []
    print(f"\nOutput base directory: {output}")
    print(f"Dry-run: {args.dry_run}; Move: {args.move}")

    for split in ['train', 'val', 'test']:
        for cls in classes:
            dest_dir = output / split / cls
            files = redistributed[split][cls]
            copy_or_move_files(files, dest_dir, args.dry_run, args.move, actions_log)

    # Write log
    log_path = output / 'redistribution_log.txt'
    ensure_dir(output)
    with open(log_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(actions_log))

    print('\nDone. Log written to:', log_path)
    if args.dry_run:
        print('No files were copied/moved. Rerun without --dry-run to apply changes.')


if __name__ == '__main__':
    main()