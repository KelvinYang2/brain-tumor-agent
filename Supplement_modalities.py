"""
supplement_modalities.py

补全 cases_dir 下每个 case 缺失的 MRI modality.

现状:
  D:\\M3D\\cases_dir\\BraTS2021_XXXXX\\
    ├── flair.nii.gz     ✓ 已有
    ├── gt.nii.gz        ✓ 已有
    └── pred.nii.gz      ✓ 已有

缺的 (为 (B) 和 (D2) 实验准备):
    ├── t1ce.nii.gz   ← 从 demo_site 复制
    ├── t1.nii.gz     ← 从 demo_site 复制
    └── t2.nii.gz     ← 从 demo_site 复制

源目录:
  C:\\Users\\kelvin\\Downloads\\brats_web\\demo_site\\data\\BraTS2021_XXXXX\\
    ├── t1ce.nii.gz
    ├── t1.nii.gz
    ├── t2.nii.gz
    ├── flair.nii.gz
    └── gt.nii.gz

按 cases_dir 里的 case 名称, 从 demo_site 拷贝对应 modality.

Usage:
  python supplement_modalities.py
  python supplement_modalities.py --dry_run    # 只打印不复制
  python supplement_modalities.py --modalities t1ce  # 只补 t1ce
  python supplement_modalities.py --overwrite   # 已有文件也覆盖
"""

import argparse
import shutil
import sys
from pathlib import Path


# 默认路径 - 按实际情况可以改
DEFAULT_SRC = r"C:\Users\kelvin\Downloads\brats_web\demo_site\data"
DEFAULT_DST = r"D:\M3D\cases_dir"
DEFAULT_MODALITIES = ["t1ce", "t1", "t2"]


def human_size(bytes_: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"

def find_modality_file(case_src_dir: Path, modality: str) -> Path | None:
    """
    在 case 源目录下查找指定 modality 的文件.
    支持多种常见命名:
      - t1ce.nii.gz
      - BraTS2021_XXXXX_t1ce.nii.gz
      - T1ce.nii.gz (大小写不敏感)
    """
    candidates = [
        case_src_dir / f"{modality}.nii.gz",
        case_src_dir / f"{modality.upper()}.nii.gz",
    ]
    # 按 glob 模式搜索
    candidates.extend(case_src_dir.glob(f"*_{modality}.nii.gz"))
    candidates.extend(case_src_dir.glob(f"*_{modality.upper()}.nii.gz"))
    candidates.extend(case_src_dir.glob(f"*{modality}.nii.gz"))

    # 去重
    seen = set()
    for c in candidates:
        if c.exists() and c not in seen:
            seen.add(c)
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description="补全 cases_dir 的 modality")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="源数据目录 (default: demo_site/data)")
    ap.add_argument("--dst", default=DEFAULT_DST,
                    help="目标 cases_dir (default: D:/M3D/cases_dir)")
    ap.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES,
                    choices=["t1ce", "t1", "t2", "flair"],
                    help="要补的 modality (default: t1ce t1 t2)")
    ap.add_argument("--dry_run", action="store_true",
                    help="只打印将要复制的文件, 不实际复制")
    ap.add_argument("--overwrite", action="store_true",
                    help="已有文件也覆盖 (默认跳过)")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"源目录不存在: {src_root}")
        sys.exit(1)
    if not dst_root.exists():
        print(f"目标目录不存在: {dst_root}")
        sys.exit(1)

    # 扫描目标下的所有 case
    dst_cases = sorted([
        d for d in dst_root.iterdir()
        if d.is_dir() and d.name.startswith("BraTS2021_")
    ])
    if not dst_cases:
        print(f"{dst_root} 下没找到 BraTS2021_* 子文件夹")
        sys.exit(1)

    print(f"目标 cases_dir: {dst_root}")
    print(f"源数据目录:     {src_root}")
    print(f"要补的 modality: {args.modalities}")
    print(f"Case 数量:       {len(dst_cases)}")
    print(f"Dry run:        {args.dry_run}")
    print(f"Overwrite:      {args.overwrite}")
    print()

    # 统计
    stats = {
        "copied":     0,
        "skipped_existed": 0,
        "missing_src":     0,
        "missing_case_dir":0,
        "total_size":      0,
    }
    missing_cases = []

    for i, case_dst_dir in enumerate(dst_cases, 1):
        case_id = case_dst_dir.name
        case_src_dir = src_root / case_id

        if not case_src_dir.exists():
            print(f"[{i:3d}/{len(dst_cases)}] {case_id} - 源目录不存在, 跳过")
            missing_cases.append(case_id)
            stats["missing_case_dir"] += 1
            continue

        case_actions = []  # 这个 case 要做什么
        for mod in args.modalities:
            dst_file = case_dst_dir / f"{mod}.nii.gz"
            src_file = find_modality_file(case_src_dir, mod)

            if dst_file.exists() and not args.overwrite:
                case_actions.append((mod, "skip", "已存在"))
                stats["skipped_existed"] += 1
            elif src_file is None:
                case_actions.append((mod, "miss", "源文件找不到"))
                stats["missing_src"] += 1
            else:
                size = src_file.stat().st_size
                if args.dry_run:
                    case_actions.append((mod, "dry", f"{human_size(size)}"))
                else:
                    try:
                        shutil.copy2(src_file, dst_file)
                        case_actions.append((mod, "ok", f"{human_size(size)}"))
                        stats["copied"] += 1
                        stats["total_size"] += size
                    except Exception as e:
                        case_actions.append((mod, "err", str(e)))

        # 打印这个 case 的状态
        status_str = "  ".join(
            f"{mod}:{'✓' if a == 'ok' else '=' if a == 'skip' else '?' if a == 'miss' else '~' if a == 'dry' else '✗'}"
            for mod, a, _ in case_actions
        )
        detail_str = "  ".join(f"{mod}={note}" for mod, _, note in case_actions)
        print(f"[{i:3d}/{len(dst_cases)}] {case_id}  {status_str}  ({detail_str})")

    # 汇总
    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("DRY RUN — 没有实际复制任何文件")
    print(f"复制成功:       {stats['copied']}")
    print(f"已存在, 跳过:   {stats['skipped_existed']}")
    print(f"源文件找不到:   {stats['missing_src']}")
    print(f"源 case 目录缺: {stats['missing_case_dir']}")
    print(f"总传输大小:     {human_size(stats['total_size'])}")

    if missing_cases:
        print(f"\n{len(missing_cases)} 个 case 在源目录里找不到:")
        for c in missing_cases[:10]:
            print(f"   - {c}")
        if len(missing_cases) > 10:
            print(f"   ... 还有 {len(missing_cases) - 10} 个")

    # 最终验证: 每个 case 现在有哪些 modality
    print(f"\n最终文件状态 (前 5 个 case):")
    expected = ["flair.nii.gz", "t1ce.nii.gz", "t1.nii.gz", "t2.nii.gz",
                "gt.nii.gz", "pred.nii.gz"]
    for case_dst_dir in dst_cases[:5]:
        case_id = case_dst_dir.name
        status = {f: (case_dst_dir / f).exists() for f in expected}
        status_str = "  ".join(f"{f.replace('.nii.gz', '')}:{'✓' if v else '✗'}"
                                for f, v in status.items())
        print(f"  {case_id}  {status_str}")

    # 给 (B) 和 (D2) 的准备状态
    print(f"\n(B) 和 (D2) 实验就绪度检查:")
    ready_d2 = sum(
        1 for c in dst_cases
        if (c / "t1ce.nii.gz").exists() and (c / "flair.nii.gz").exists()
    )
    ready_b = sum(
        1 for c in dst_cases
        if (c / "flair.nii.gz").exists() and (c / "gt.nii.gz").exists()
    )
    print(f"  (D2) Claude + features + T1ce slices: {ready_d2}/{len(dst_cases)} case 就绪")
    print(f"  (B)  MedGemma + KAIST features:       {ready_b}/{len(dst_cases)} case 就绪")


if __name__ == "__main__":
    main()