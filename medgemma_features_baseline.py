"""
MedGemma 4B + KAIST features (纯文本 prompt, 不传图像)

严格 4B-scale ablation vs (A) M3D-LaMed.
(B) 和 (D1) Claude+KAIST 用完全一样的 prompt, 只换 LLM.

关键技术:
  - Assistant prefix injection: 预填 '{"diagnosis": "' 强制 JSON 输出
  - bad_words_ids: 屏蔽 <unused94>thought 等 thinking 触发 token
  - field_regex fallback: 从带垃圾尾巴的 JSON 里抠字段 (MedGemma 常态)

Input:
  cases_dir/
    BraTS2021_00000/
      flair.nii.gz
      gt.nii.gz   或   pred.nii.gz

Output:
  results_medgemma_kaist_gt.json   (用 GT mask 提取特征)
  results_medgemma_kaist_pred.json (用 KAIST 预测 mask 提取特征)

Usage:
  # Smoke test
  python medgemma_features_baseline.py --cases_dir D:/M3D/cases_dir \
      --mask_name gt.nii.gz --output test_b_gt.json --limit 3

  # 全量
  python medgemma_features_baseline.py --cases_dir D:/M3D/cases_dir \
      --mask_name gt.nii.gz --output results_medgemma_kaist_gt.json
  python medgemma_features_baseline.py --cases_dir D:/M3D/cases_dir \
      --mask_name pred.nii.gz --output results_medgemma_kaist_pred.json
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


MODEL_ID_DEFAULT = r"models\medgemma-1.5-4b-it"


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


def load_medgemma(model_dir: str):
    print(f"Loading MedGemma from {model_dir}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def get_stop_tokens(tokenizer):
    tokens = [tokenizer.eos_token_id]
    for tok in ("<end_of_turn>", "<eos>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            tokens.append(tid)
    return list(set(t for t in tokens if t is not None))


def run_medgemma_text(model, tokenizer, prompt: str, stop_tokens,
                      max_new_tokens: int = 600) -> str:
    """
    真正压制 thinking mode 的推理.

    策略:
      1. Prompt 直接以 "Output the JSON now:" 结尾, 不留 thinking 空间
      2. Assistant prefix injection: 手动预填 {"diagnosis": "
      3. Bad words: 屏蔽 <unused94>, thought 等多 token 序列
    """
    prompt_final = prompt + (
        "\n\nOutput the JSON object now. No thinking, no analysis, "
        "no explanation. Only the JSON:"
    )

    messages = [{"role": "user", "content": prompt_final}]

    # 预填 assistant 前缀, 强制进入 JSON
    chat_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    chat_text = chat_text + '{"diagnosis": "'

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prefix_len = inputs["input_ids"].shape[1]

    # Thinking 触发 token 黑名单
    bad_sequences = [
        "<unused94>thought", "<unused94>", "thought\n",
        "analysis", "Analysis", "Let me", "let me",
        "I need to", "I will", "The user wants",
        "Thinking:", "thinking:",
    ]
    bad_words_ids = []
    for seq in bad_sequences:
        try:
            ids = tokenizer(seq, add_special_tokens=False)["input_ids"]
            if ids and all(i != tokenizer.unk_token_id for i in ids):
                bad_words_ids.append(ids)
        except Exception:
            pass

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stop_tokens,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_ids if bad_words_ids else None,
        )

    generated = tokenizer.decode(output[0][prefix_len:], skip_special_tokens=True)
    return '{"diagnosis": "' + generated.strip()


# ================================================================
# Parsing — 4 级 fallback:
#   direct_json → stripped_fences → regex_extract → field_regex → keyword_fallback
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

    # Attempt 1: direct JSON
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

    # Attempt 2: strip markdown fences
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

    # Attempt 3: outermost {...}
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

    # Attempt 3.5: field-level regex
    # (MedGemma 常态 — JSON 有拼写错误/非法数字/尾部垃圾, 整体不可 parse 但字段值可抠)
    field_extracted = {}
    for field in ["diagnosis", "laterality", "lobe", "grade", "enhancement"]:
        m = re.search(rf'"{field}"\s*:\s*"([^"]+)"', raw_text, re.IGNORECASE)
        if m:
            field_extracted[field] = m.group(1)
    m_conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw_text)
    if m_conf:
        try:
            # MedGemma 会写 "0." 这种非法 float, 清一下
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

    # Attempt 4: keyword fallback (最宽松)
    kw = _keyword_parse(raw_text)
    result.update(kw)
    result["parse_method"] = "keyword_fallback"
    return result


def _normalize(result: dict) -> dict:
    """统一大小写 + 清理 MedGemma 常见脏字符."""
    for k in ["diagnosis", "lobe", "grade", "enhancement"]:
        v = result.get(k)
        if isinstance(v, str):
            # MedGemma 会写 ".low grade glioma." 这种带点的脏字符
            result[k] = v.lower().strip().strip(".").strip()
    if isinstance(result.get("laterality"), str):
        result["laterality"] = result["laterality"].upper().strip().strip(".").strip()
    return result


def _keyword_parse(text: str) -> dict:
    """Fallback — 从自由文本抽字段."""
    text_lower = text.lower()
    out = {}

    diag_patterns = [
        ("glioblastoma",     ["glioblastoma", "gbm", "grade iv glioma", "grade 4 glioma",
                              "glioblastoma multiforme"]),
        ("low_grade_glioma", ["low-grade glioma", "low grade glioma", "grade ii glioma",
                              "grade 2 glioma", "grade i glioma", "grade 1 glioma",
                              "diffuse astrocytoma", "oligodendroglioma",
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
    ap.add_argument("--mask_name", default="gt.nii.gz",
                    help="gt.nii.gz 或 pred.nii.gz")
    ap.add_argument("--output", default="results_medgemma_kaist.json", type=Path)
    ap.add_argument("--model_dir", default=MODEL_ID_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cases = find_cases(args.cases_dir, args.mask_name)
    if args.limit:
        cases = cases[:args.limit]

    print(f"[setup] cases_dir = {args.cases_dir}")
    print(f"[setup] mask      = {args.mask_name}")
    print(f"[setup] output    = {args.output}")
    print(f"[setup] n cases   = {len(cases)}\n")

    model, tokenizer = load_medgemma(args.model_dir)
    stop_tokens = get_stop_tokens(tokenizer)

    results = {}
    if args.resume and args.output.exists():
        try:
            results = json.loads(args.output.read_text(encoding="utf-8"))
            print(f"[resume] 已有 {len(results)} 个结果\n")
        except json.JSONDecodeError:
            pass

    n_parse_ok, n_parse_fallback, n_parse_fail = 0, 0, 0
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
            print("  空 mask, 跳过")
            results[cid] = {"error": "empty_mask"}
            continue

        print(f"  side={feats['laterality']:>20s}  V_WT={feats['volume_whole_tumor_cm3']:>5.1f}cm³  "
              f"enh={feats['enhancement_pattern']}")

        prompt = format_prompt(feats)

        try:
            t_c = time.time()
            raw = run_medgemma_text(model, tokenizer, prompt, stop_tokens)
            dt = time.time() - t_c
        except Exception as e:
            print(f"  inference 失败: {e}")
            results[cid] = {"features": feats, "prompt": prompt, "error": str(e)}
            continue

        parsed = parse_output(raw)
        # 结构化 parse 成功 = 走了 direct/stripped/regex/field_regex 任一个
        if parsed["parse_success"] and parsed["parse_method"] != "keyword_fallback":
            n_parse_ok += 1
        elif parsed["parse_method"] == "keyword_fallback":
            n_parse_fallback += 1
        else:
            n_parse_fail += 1

        print(f"  ({dt:.1f}s, {parsed['parse_method']}) "
              f"diag={parsed['diagnosis']}, lat={parsed['laterality']}, "
              f"lobe={parsed['lobe']}, grade={parsed['grade']}")

        results[cid] = {
            "features": feats,
            "prompt":   prompt,
            "raw_output": raw,
            "parsed":   parsed,
            "inference_time_s": round(dt, 2),
            "model":    "medgemma-1.5-4b-it",
            "mask_source": args.mask_name,
        }

        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"完成: {len(results)} cases, 耗时 {elapsed:.0f}s "
          f"({elapsed / max(len(results), 1):.1f}s/case)")
    print(f"  结构化 parse (direct/stripped/regex/field_regex): {n_parse_ok}")
    print(f"  keyword fallback: {n_parse_fallback}")
    print(f"  全失败:           {n_parse_fail}")

    # 分布 — 强制转 str 防 list/dict 崩 Counter
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