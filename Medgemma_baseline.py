"""
批量 MedGemma 1.5 4B zero-shot inference on BraTS cases.

Phase 0 baseline (A') — parameter-matched alternative to M3D-LaMed.

配置:
  - 输入: axial 2D slices (默认 t1ce, 也支持 flair/t1/t2)
  - Slices: 12 axial slices, 神经学方位 (patient-left = screen-left)
  - Generation: do_sample=False, no_repeat_ngram_size=4, repetition_penalty=1.2
  - 多 EOS token (Gemma 3 的 <end_of_turn>)

Prompt 根据 modality 自动调整, 避免误导模型.

Input:
  cases_dir/
    BraTS2021_00000/
      t1ce.nii.gz / flair.nii.gz / t1.nii.gz / t2.nii.gz
    ...

Output:
  results_medgemma.json (或 --output 指定的文件)
    { case_id: { diagnosis, laterality, lobe, grade, enhancement,
                 raw_output, inference_time_s } }

Usage:
  # T1ce (默认)
  python medgemma_baseline.py --cases_dir D:/M3D/cases_dir \
      --output results_medgemma_t1ce.json

  # FLAIR
  python medgemma_baseline.py --cases_dir D:/M3D/cases_dir \
      --modality flair --output results_medgemma_flair.json

  # 调试
  python medgemma_baseline.py --cases_dir D:/M3D/cases_dir \
      --modality flair --output test.json --limit 3
"""

import torch
import numpy as np
import nibabel as nib
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from pathlib import Path
from collections import Counter
import argparse
import json
import time
import sys


MODEL_ID_DEFAULT = r"models\medgemma-1.5-4b-it"
N_SLICES = 12


def build_prompt(modality: str) -> str:
    if modality == "t1ce":
        seq_desc = "post-contrast T1-weighted (T1ce)"
        focus = (
            "Pay attention to enhancement patterns (ring, solid, none) and "
            "central necrosis."
        )
    elif modality == "flair":
        seq_desc = "FLAIR"
        focus = (
            "Pay attention to hyperintense regions (edema, gliosis), "
            "lesion extent, and margin characteristics (well-defined vs infiltrative)."
        )
    elif modality == "t1":
        seq_desc = "T1-weighted (non-contrast)"
        focus = (
            "Pay attention to signal intensity patterns, lesion boundaries, "
            "and any hypointense (dark) or hyperintense (bright) regions."
        )
    elif modality == "t2":
        seq_desc = "T2-weighted"
        focus = (
            "Pay attention to hyperintense regions (fluid, edema, cysts) "
            "and fluid-sensitive abnormalities."
        )
    else:
        seq_desc = f"{modality.upper()}"
        focus = "Describe any abnormalities."

    return (
        f"These are {N_SLICES} axial slices from a brain {seq_desc} MRI scan, "
        "sampled evenly from inferior to superior. "
        "The images follow the neurological display convention: "
        "the patient's LEFT side is on the LEFT of the image, "
        "and anterior is at the TOP. "
        f"{focus} "
        "Describe any abnormalities and provide the most likely diagnosis. Be concise."
    )


def nifti_to_slices(nifti_path: str, n_slices: int = 12):
    """
    采样 axial slices, 正确方位 (patient-left = screen-left).
    """
    img_nib = nib.as_closest_canonical(nib.load(nifti_path))
    img = img_nib.get_fdata().astype(np.float32)
    # Shape 现在是 (X, Y, Z): X=L→R, Y=P→A, Z=I→S

    n_total = img.shape[2]
    start, end = int(n_total * 0.1), int(n_total * 0.9)
    indices = np.linspace(start, end, n_slices, dtype=int)

    slices = []
    for idx in indices:
        s = img[:, :, idx]
        s = s.T            # (X, Y) → (Y, X): row=P→A, col=L→R
        s = np.flipud(s)   # 让 A 在顶部

        s = s - s.min()
        if s.max() > 0:
            s = s / s.max() * 255
        slices.append(Image.fromarray(s.astype(np.uint8)).convert("RGB"))

    return slices


def get_stop_tokens(processor):
    tokens = [processor.tokenizer.eos_token_id]
    for tok in ("<end_of_turn>", "<eos>"):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != processor.tokenizer.unk_token_id:
            tokens.append(tid)
    return list(set(t for t in tokens if t is not None))


def run_inference(model, processor, images, prompt_text: str, stop_tokens) -> str:
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
        eos_token_id=stop_tokens,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
    )
    decoded = processor.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("model\n")[-1].strip() if "model\n" in decoded else decoded
    return answer


def parse_free_text(text: str) -> dict:
    """Keyword-based parsing, 和 M3D-LaMed baseline 对齐."""
    text_lower = text.lower()
    result = {
        "diagnosis":   "unknown",
        "laterality":  "unknown",
        "lobe":        "unknown",
        "grade":       "unknown",
        "enhancement": "unknown",
    }

    # --- Diagnosis ---
    diag_patterns = [
        ("glioblastoma",     ["glioblastoma", "gbm", "grade iv glioma", "grade 4 glioma",
                              "glioblastoma multiforme"]),
        ("low_grade_glioma", ["low-grade glioma", "low grade glioma", "grade ii glioma",
                              "grade 2 glioma", "grade i glioma", "grade 1 glioma",
                              "diffuse astrocytoma", "oligodendroglioma",
                              "pilocytic astrocytoma", "lgg"]),
        ("meningioma",       ["meningioma"]),
        ("metastasis",       ["metastasis", "metastatic", "metastases", "secondary tumor"]),
        ("lymphoma",         ["lymphoma", "cns lymphoma"]),
        ("abscess",          ["abscess", "brain abscess", "cerebral abscess"]),
    ]
    for diag, keywords in diag_patterns:
        if any(kw in text_lower for kw in keywords):
            result["diagnosis"] = diag
            break
    if result["diagnosis"] == "unknown":
        if "glioma" in text_lower or "glial" in text_lower:
            result["diagnosis"] = "glioblastoma"
        elif any(w in text_lower for w in ["tumor", "tumour", "mass", "lesion", "neoplasm"]):
            result["diagnosis"] = "other"

    # --- Laterality ---
    has_left = "left" in text_lower
    has_right = "right" in text_lower
    if has_left and not has_right:
        result["laterality"] = "LEFT"
    elif has_right and not has_left:
        result["laterality"] = "RIGHT"
    elif "bilateral" in text_lower or (has_left and has_right):
        if "bilateral" in text_lower:
            result["laterality"] = "BILATERAL"
        else:
            left_idx = text_lower.find("left")
            right_idx = text_lower.find("right")
            result["laterality"] = "LEFT" if left_idx < right_idx else "RIGHT"
    elif "midline" in text_lower:
        result["laterality"] = "MIDLINE"

    # --- Lobe ---
    for lobe in ["temporal", "frontal", "parietal", "occipital",
                 "cerebellar", "cerebellum", "brainstem", "thalamic", "thalamus"]:
        if lobe in text_lower:
            if lobe in ("cerebellum", "cerebellar"):
                result["lobe"] = "cerebellar"
            elif lobe in ("thalamic", "thalamus"):
                result["lobe"] = "thalamic"
            else:
                result["lobe"] = lobe
            break

    # --- Grade ---
    if any(w in text_lower for w in ["high-grade", "high grade", "grade iv", "grade 4",
                                      "grade iii", "grade 3", "anaplastic",
                                      "glioblastoma", "gbm"]):
        result["grade"] = "high"
    elif any(w in text_lower for w in ["low-grade", "low grade", "grade ii", "grade 2",
                                        "grade i", "grade 1", "benign",
                                        "pilocytic", "well-differentiated"]):
        result["grade"] = "low"

    # --- Enhancement ---
    if ("ring" in text_lower or "rim-enhancing" in text_lower or
        "rim enhancement" in text_lower or "peripheral enhancement" in text_lower):
        result["enhancement"] = "ring"
    elif "homogeneous" in text_lower and "enhanc" in text_lower:
        result["enhancement"] = "homogeneous"
    elif "heterogeneous" in text_lower and "enhanc" in text_lower:
        result["enhancement"] = "heterogeneous"
    elif any(w in text_lower for w in ["no enhancement", "non-enhancing",
                                        "without enhancement", "does not enhance"]):
        result["enhancement"] = "none"
    elif "enhanc" in text_lower:
        result["enhancement"] = "present"

    return result


def main():
    ap = argparse.ArgumentParser(description="Batch MedGemma 1.5 zero-shot baseline")
    ap.add_argument("--cases_dir", required=True,
                    help="包含 BraTS2021_XXXXX 子文件夹的目录")
    ap.add_argument("--model_dir", default=MODEL_ID_DEFAULT,
                    help="MedGemma 模型路径或 HuggingFace ID")
    ap.add_argument("--output", default="results_medgemma.json", help="输出 JSON")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个 (调试)")
    ap.add_argument("--modality", default="t1ce",
                    choices=["t1ce", "flair", "t1", "t2"],
                    help="用哪个 modality (默认 t1ce)")
    args = ap.parse_args()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.exists():
        print(f"cases_dir 不存在: {cases_dir}")
        sys.exit(1)

    # 扫描 case
    case_dirs = sorted([
        d for d in cases_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS2021_")
    ])
    if not case_dirs:
        print(f"{cases_dir} 下没找到 BraTS2021_* 子目录")
        sys.exit(1)

    if args.limit:
        case_dirs = case_dirs[:args.limit]

    # 根据 modality 生成对应 prompt
    prompt = build_prompt(args.modality)
    print(f"找到 {len(case_dirs)} 个 case, modality={args.modality}")
    print(f"Prompt: {prompt[:140]}...\n")

    # 加载模型
    print("Loading MedGemma...")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_dir)
    stop_tokens = get_stop_tokens(processor)
    print(f"  Loaded in {time.time() - t0:.1f}s. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Stop tokens: {stop_tokens}\n")

    # 断点续跑
    existing = {}
    out_path = Path(args.output)
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"检测到已有结果 ({len(existing)} cases), 将跳过这些 case\n")
        except json.JSONDecodeError:
            existing = {}

    results = dict(existing)
    t_total = time.time()
    n_new = 0

    for i, case_dir in enumerate(case_dirs):
        case_id = case_dir.name

        if case_id in results:
            print(f"[{i + 1}/{len(case_dirs)}] {case_id} - 已有, 跳过")
            continue

        nifti_path = case_dir / f"{args.modality}.nii.gz"
        if not nifti_path.exists():
            print(f"[{i + 1}/{len(case_dirs)}] {case_id} - 缺 {args.modality}.nii.gz")
            continue

        print(f"[{i + 1}/{len(case_dirs)}] {case_id}")

        try:
            slices = nifti_to_slices(str(nifti_path), N_SLICES)

            t_case = time.time()
            raw = run_inference(model, processor, slices, prompt, stop_tokens)
            dt = time.time() - t_case

            parsed = parse_free_text(raw)

            results[case_id] = {
                **parsed,
                "raw_output": raw,
                "inference_time_s": round(dt, 2),
                "modality": args.modality,
            }

            print(f"  ({dt:.1f}s) diagnosis={parsed['diagnosis']}, "
                  f"laterality={parsed['laterality']}, lobe={parsed['lobe']}")
            n_new += 1

            if n_new % 10 == 0:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"  [checkpoint] 已存 {len(results)} cases")

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            results[case_id] = {
                "diagnosis":   "ERROR",
                "laterality":  "unknown",
                "lobe":        "unknown",
                "grade":       "unknown",
                "enhancement": "unknown",
                "raw_output":  f"ERROR: {e}",
                "inference_time_s": 0,
                "modality":    args.modality,
            }

    elapsed = time.time() - t_total

    # 最终保存
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"完成: {len(results)} cases ({n_new} 新增), 总耗时 {elapsed:.0f}s")
    print(f"Modality: {args.modality}")
    print(f"结果: {args.output}")

    diagnoses = [r["diagnosis"] for r in results.values()]
    laterals = [r["laterality"] for r in results.values()]
    lobes = [r["lobe"] for r in results.values()]

    print(f"\n诊断分布:")
    for diag, count in Counter(diagnoses).most_common():
        print(f"  {diag}: {count} ({100 * count / len(diagnoses):.0f}%)")

    print(f"\n侧别分布:")
    for lat, count in Counter(laterals).most_common():
        print(f"  {lat}: {count} ({100 * count / len(laterals):.0f}%)")

    print(f"\n脑叶分布:")
    for lobe, count in Counter(lobes).most_common():
        print(f"  {lobe}: {count} ({100 * count / len(lobes):.0f}%)")

    unknown_rate = sum(1 for d in diagnoses if d in ("unknown", "other", "ERROR")) / len(diagnoses)
    lat_known_rate = sum(1 for l in laterals if l != "unknown") / len(laterals)
    lobe_known_rate = sum(1 for l in lobes if l != "unknown") / len(lobes)

    print(f"\nunknown/other/ERROR: {100 * unknown_rate:.0f}%")
    print(f"侧别识别率: {100 * lat_known_rate:.0f}%")
    print(f"脑叶识别率: {100 * lobe_known_rate:.0f}%")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()