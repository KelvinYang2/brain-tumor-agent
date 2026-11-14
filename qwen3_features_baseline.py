"""
qwen3_features_baseline.py

Qwen3-14B + KAIST features 配置 (B' — 比 MedGemma 更大尺度的对照).

设计意图:
  Phase 0 已经发现 4B LLM 不能用结构化特征做诊断推理 (MedGemma+KAIST 真实 26%).
  Qwen3-14B 大致 ≈ Qwen2.5-32B 能力, 是 4-bit 量化下 24GB GPU 能跑的最大模型.
  目标: 测出 14B 级别能不能给 specialist+LLM 架构带来真实增益.

关键技术:
  1. enable_thinking=False — 避免 Qwen3 默认 thinking 把 token 用光 (MedGemma 同样问题)
  2. 4-bit 量化 (bitsandbytes nf4) — 14B 模型 ~9GB VRAM
  3. 复用 feature_extractor.format_prompt() — 和 MedGemma+KAIST / Claude+KAIST 完全一致

Input:
  cases_dir/BraTS2021_XXXXX/{flair, gt or pred}.nii.gz

Output:
  results_qwen3_kaist_gt.json   (GT mask)
  results_qwen3_kaist_pred.json (Pred mask)

Usage:
  # Smoke test 3 个
  python qwen3_features_baseline.py --cases_dir D:/M3D/cases_dir \
      --mask_name gt.nii.gz --output test_qwen3_gt.json --limit 3

  # 全量
  python qwen3_features_baseline.py --cases_dir D:/M3D/cases_dir \
      --mask_name gt.nii.gz --output results_qwen3_kaist_gt.json
  python qwen3_features_baseline.py --cases_dir D:/M3D/cases_dir \
      --mask_name pred.nii.gz --output results_qwen3_kaist_pred.json
"""

import sys
import json
import re
import time
import argparse
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# feature_extractor.py 在 D:\M3D 下
sys.path.insert(0, r"D:\M3D")
from feature_extractor import extract_features, format_prompt


MODEL_ID_DEFAULT = r"D:\models\Qwen3-14B-4bit"


def find_cases(cases_dir: Path, mask_name: str):
    cases = []
    for d in sorted(cases_dir.iterdir()):
        if not d.is_dir():
            continue
        flair_candidates = [d / "flair.nii.gz"] + list(d.glob("*_flair.nii.gz"))
        flair = next((f for f in flair_candidates if f.exists()), None)
        if flair is None:
            continue

        mask = None
        direct = d / mask_name
        if direct.exists():
            mask = direct
        else:
            stem = mask_name.replace(".nii.gz", "")
            candidates = list(d.glob(f"*_{stem}.nii.gz")) + list(d.glob(f"*{stem}.nii.gz"))
            if candidates:
                mask = candidates[0]
        if mask is None:
            continue

        cases.append({"case_id": d.name, "flair": str(flair), "mask": str(mask)})
    return cases

def load_qwen3(model_dir: str):
    print(f"Loading Qwen3 from {model_dir} (pre-quantized 4-bit)...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",   # ← 加这一行
    )
    model.eval()

    print(f"  Loaded in {time.time() - t0:.1f}s. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def run_qwen3(model, tokenizer, prompt: str, max_new_tokens: int = 800) -> str:
    """
    Qwen3 推理 — 关键: enable_thinking=False 避免 thinking 占满 token.

    用 non-thinking 模式的官方推荐参数:
      Temperature=0.7, TopP=0.8, TopK=20, MinP=0
    """
    messages = [{"role": "user", "content": prompt}]

    # 关键: enable_thinking=False
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


# ================================================================
# Parsing — 4 级 fallback (和 medgemma_features_baseline.py 一致)
# ================================================================
def parse_output(raw_text: str) -> dict:
    result = {
        "diagnosis":     "unknown",
        "confidence":    0.0,
        "laterality":    "unknown",
        "lobe":          "unknown",
        "grade":         "unknown",
        "enhancement":   "unknown",
        "key_evidence":  [],
        "differentials": [],
        "parse_success": False,
        "parse_method":  "none",
    }

    try:
        data = json.loads(raw_text.strip())
        for k in ["diagnosis", "confidence", "laterality", "lobe",
                  "grade", "enhancement", "key_evidence", "differentials"]:
            if k in data:
                result[k] = data[k]
        result["parse_success"] = True
        result["parse_method"] = "direct_json"
        return _normalize(result)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw_text.strip(),
                     flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
        for k in ["diagnosis", "confidence", "laterality", "lobe",
                  "grade", "enhancement", "key_evidence", "differentials"]:
            if k in data:
                result[k] = data[k]
        result["parse_success"] = True
        result["parse_method"] = "stripped_fences"
        return _normalize(result)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        try:
            data = json.loads(match.group(0))
            for k in ["diagnosis", "confidence", "laterality", "lobe",
                      "grade", "enhancement", "key_evidence", "differentials"]:
                if k in data:
                    result[k] = data[k]
            result["parse_success"] = True
            result["parse_method"] = "regex_extract"
            return _normalize(result)
        except json.JSONDecodeError:
            pass

    # Field-level regex fallback
    field_extracted = {}
    for field in ["diagnosis", "laterality", "lobe", "grade", "enhancement"]:
        m = re.search(rf'"{field}"\s*:\s*"([^"]+)"', raw_text, re.IGNORECASE)
        if m:
            field_extracted[field] = m.group(1)
    m_conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw_text)
    if m_conf:
        try:
            v = m_conf.group(1).rstrip(".")
            field_extracted["confidence"] = float(v) if v else 0.0
        except ValueError:
            pass
    if field_extracted:
        for k, v in field_extracted.items():
            result[k] = v
        result["parse_success"] = True
        result["parse_method"] = "field_regex"
        return _normalize(result)

    # Keyword fallback (最宽松)
    kw = _keyword_parse(raw_text)
    result.update(kw)
    result["parse_method"] = "keyword_fallback"
    return result


def _normalize(result: dict) -> dict:
    for k in ["diagnosis", "lobe", "grade", "enhancement"]:
        v = result.get(k)
        if isinstance(v, str):
            result[k] = v.lower().strip().strip(".").strip()
    if isinstance(result.get("laterality"), str):
        result["laterality"] = result["laterality"].upper().strip().strip(".").strip()
    return result


def _keyword_parse(text: str) -> dict:
    text_lower = text.lower()
    out = {}

    diag_patterns = [
        ("glioblastoma",     ["glioblastoma", "gbm", "grade iv glioma", "grade 4 glioma"]),
        ("low_grade_glioma", ["low-grade glioma", "low grade glioma", "grade ii glioma",
                              "grade 2 glioma", "diffuse astrocytoma", "oligodendroglioma",
                              "pilocytic astrocytoma", "lgg"]),
        ("meningioma",       ["meningioma"]),
        ("metastasis",       ["metastasis", "metastatic", "metastases"]),
        ("lymphoma",         ["lymphoma"]),
        ("abscess",          ["abscess"]),
    ]
    for diag, keywords in diag_patterns:
        if any(kw in text_lower for kw in keywords):
            out["diagnosis"] = diag
            break
    if "diagnosis" not in out:
        if "glioma" in text_lower:
            out["diagnosis"] = "glioblastoma"
        elif any(w in text_lower for w in ["tumor", "mass", "lesion", "neoplasm"]):
            out["diagnosis"] = "other"

    has_left = "left" in text_lower
    has_right = "right" in text_lower
    if has_left and not has_right:
        out["laterality"] = "LEFT"
    elif has_right and not has_left:
        out["laterality"] = "RIGHT"
    elif has_left and has_right:
        if "bilateral" in text_lower:
            out["laterality"] = "BILATERAL"
        else:
            out["laterality"] = "LEFT" if text_lower.find("left") < text_lower.find("right") else "RIGHT"
    elif "midline" in text_lower:
        out["laterality"] = "MIDLINE"

    for lobe in ["temporal", "frontal", "parietal", "occipital",
                 "cerebellar", "brainstem", "thalamic"]:
        if lobe in text_lower:
            out["lobe"] = lobe
            break

    if any(w in text_lower for w in ["high-grade", "high grade", "grade iv", "grade 4",
                                      "grade iii", "grade 3", "anaplastic",
                                      "glioblastoma", "gbm"]):
        out["grade"] = "high"
    elif any(w in text_lower for w in ["low-grade", "low grade", "grade ii", "grade 2",
                                        "grade i", "grade 1", "benign"]):
        out["grade"] = "low"

    if "ring" in text_lower or "rim-enhancing" in text_lower or "peripheral enhancement" in text_lower:
        out["enhancement"] = "ring"
    elif "homogeneous" in text_lower and "enhanc" in text_lower:
        out["enhancement"] = "homogeneous"
    elif "heterogeneous" in text_lower and "enhanc" in text_lower:
        out["enhancement"] = "heterogeneous"
    elif any(w in text_lower for w in ["no enhancement", "non-enhancing"]):
        out["enhancement"] = "none"

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases_dir", required=True, type=Path)
    ap.add_argument("--mask_name", default="gt.nii.gz")
    ap.add_argument("--output", default="results_qwen3_kaist.json", type=Path)
    ap.add_argument("--model_dir", default=MODEL_ID_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cases = find_cases(args.cases_dir, args.mask_name)
    if args.limit:
        cases = cases[:args.limit]

    print(f"[setup] cases_dir = {args.cases_dir}")
    print(f"[setup] mask      = {args.mask_name}")
    print(f"[setup] model     = {args.model_dir}")
    print(f"[setup] output    = {args.output}")
    print(f"[setup] n cases   = {len(cases)}\n")

    model, tokenizer = load_qwen3(args.model_dir)

    results = {}
    if args.resume and args.output.exists():
        try:
            results = json.loads(args.output.read_text(encoding="utf-8"))
            print(f"[resume] 已有 {len(results)} 个结果\n")
        except json.JSONDecodeError:
            pass

    n_parse_ok, n_parse_fallback = 0, 0
    t_start = time.time()

    for i, case in enumerate(cases, 1):
        cid = case["case_id"]
        if args.resume and cid in results and results[cid].get("parsed", {}).get("parse_success"):
            continue
        print(f"[{i:3d}/{len(cases)}] {cid}")

        try:
            feats = extract_features(case["flair"], case["mask"])
        except Exception as e:
            print(f"  feature_extract 失败: {e}")
            results[cid] = {"error": f"feature_extract: {e}"}
            continue
        if feats is None:
            print("  空 mask")
            results[cid] = {"error": "empty_mask"}
            continue

        print(f"  side={feats['laterality']:>20s}  V_WT={feats['volume_whole_tumor_cm3']:>5.1f}cm³  "
              f"enh={feats['enhancement_pattern']}")

        prompt = format_prompt(feats)

        try:
            t_c = time.time()
            raw = run_qwen3(model, tokenizer, prompt)
            dt = time.time() - t_c
        except Exception as e:
            print(f"  inference 失败: {e}")
            results[cid] = {"features": feats, "prompt": prompt, "error": str(e)}
            continue

        parsed = parse_output(raw)
        if parsed["parse_success"] and parsed["parse_method"] != "keyword_fallback":
            n_parse_ok += 1
        else:
            n_parse_fallback += 1

        print(f"  ({dt:.1f}s, {parsed['parse_method']}) "
              f"diag={parsed['diagnosis']}, lat={parsed['laterality']}, "
              f"lobe={parsed['lobe']}, grade={parsed['grade']}")

        results[cid] = {
            "features": feats,
            "prompt":   prompt,
            "raw_output": raw,
            "parsed":   parsed,
            "inference_time_s": round(dt, 2),
            "model":    "Qwen3-14B-4bit",
            "mask_source": args.mask_name,
        }

        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"完成: {len(results)} cases, 耗时 {elapsed:.0f}s "
          f"({elapsed / max(len(results), 1):.1f}s/case)")
    print(f"  结构化 parse: {n_parse_ok}")
    print(f"  fallback:    {n_parse_fallback}")

    diagnoses = [str(r["parsed"]["diagnosis"]) for r in results.values()
                 if r.get("parsed")]
    laterals = [str(r["parsed"]["laterality"]) for r in results.values()
                if r.get("parsed")]

    if diagnoses:
        print(f"\n诊断分布:")
        for d, c in Counter(diagnoses).most_common():
            print(f"  {d}: {c} ({100 * c / len(diagnoses):.0f}%)")

    if laterals:
        print(f"\nLaterality 分布:")
        for l, c in Counter(laterals).most_common():
            print(f"  {l}: {c} ({100 * c / len(laterals):.0f}%)")

    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\n结果保存到: {args.output}")


if __name__ == "__main__":
    main()