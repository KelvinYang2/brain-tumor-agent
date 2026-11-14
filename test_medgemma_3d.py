"""
对比测试 v3: 修复关键 bug — 方位!

核心 bug (v1/v2):
  nib.load().get_fdata() 返回原始 array, shape (H, W, D).
  第一轴是 Left-Right (对于 RAS 数据), 第二轴是 Posterior-Anterior, 第三轴是 Inferior-Superior.

  但 PIL Image.fromarray(s) 默认把:
    - 第一轴当作纵向 (row/y)
    - 第二轴当作横向 (col/x)

  所以原本的 Left-Right 方向被变成了"上下", 图片整体转了 90°.
  模型看到一个"立起来的大脑", 说 left/right 全是瞎猜.

修复:
  1. nib.as_closest_canonical() 强制 RAS+ 方位
  2. slice 取出后 .T 转置: 让第一轴 (L-R) 变成横向, 第二轴 (P-A) 变成纵向
  3. 上下翻转: axial view 的头应该朝上 (anterior 在上)

验证: 修复后显示的 axial slice 应该是神经学约定 —
  屏幕左 = 病人左, 屏幕上 = 前方 (anterior)
"""

import torch
import numpy as np
import nibabel as nib
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import time

MODEL_ID = r"models\medgemma-1.5-4b-it"

CASE_DIR   = r"C:\Users\kelvin\Downloads\brats_web\demo_site\data\BraTS2021_00000"
FLAIR_PATH = rf"{CASE_DIR}\flair.nii.gz"
T1CE_PATH  = rf"{CASE_DIR}\t1ce.nii.gz"

N_SLICES = 12

RUN_FLAIR_ONLY = True
RUN_T1CE_ONLY  = True
RUN_MULTIMODAL = True

# 保存第一张 slice 用于人工验证方位是否正确
SAVE_DEBUG_SLICE = True


def nifti_to_slices(nifti_path: str, n_slices: int = 12, save_debug: bool = False):
    """
    从 NIfTI 采样 axial slice, 正确处理方位.

    返回的 PIL Image 遵循神经学约定:
      - 屏幕左 = 病人左 (anatomical left)
      - 屏幕上 = 前方 (anterior)
    """
    # 1. 强制转到 RAS+ canonical 方位
    img_nib = nib.as_closest_canonical(nib.load(nifti_path))
    img = img_nib.get_fdata().astype(np.float32)
    # 现在 img shape = (X, Y, Z), 其中
    #   X 轴: Left → Right (从病人左到病人右)
    #   Y 轴: Posterior → Anterior
    #   Z 轴: Inferior → Superior

    n_total = img.shape[2]
    start, end = int(n_total * 0.1), int(n_total * 0.9)
    indices = np.linspace(start, end, n_slices, dtype=int)

    slices = []
    for idx in indices:
        # 2. 取 axial slice. img[:, :, idx] shape = (X, Y),
        #    第一轴是 L→R (横向), 第二轴是 P→A (纵向)
        s = img[:, :, idx]

        # 3. 让它符合神经学显示约定:
        #    - 屏幕横向 (col) = L→R: 第一轴已经是 L→R, 但 PIL 把第一轴当纵向
        #      所以需要 transpose, 让第一轴变成横向
        #    - 屏幕纵向 (row) = Anterior 在上: transpose 后第二轴变纵向 (P→A),
        #      但显示时 row 0 在顶部, 所以需要 flipud 让 A 在上
        s = s.T           # (X, Y) → (Y, X): 现在 row=P→A, col=L→R
        s = np.flipud(s)  # 翻转纵向: 现在 row 顶部=A, col 左=L

        # Normalize
        s = s - s.min()
        if s.max() > 0:
            s = s / s.max() * 255
        pil_img = Image.fromarray(s.astype(np.uint8)).convert("RGB")
        slices.append(pil_img)

    # Debug: 保存中间一张 slice 看方位对不对
    if save_debug:
        debug_path = "debug_slice_orientation.png"
        slices[len(slices) // 2].save(debug_path)
        print(f"  [debug] 保存中间 slice 到 {debug_path} "
              f"(应该看到: 左=病人左, 上=前方)")

    return slices, indices, n_total


def get_stop_tokens(processor):
    tokens = [processor.tokenizer.eos_token_id]
    for tok in ("<end_of_turn>", "<eos>"):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != processor.tokenizer.unk_token_id:
            tokens.append(tid)
    return list(set(t for t in tokens if t is not None))


def run_config(model, processor, images, prompt_text: str, label: str, stop_tokens):
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)

    print(f"\n{'=' * 60}")
    print(f"[{label}] {len(images)} images, input tokens: {inputs['input_ids'].shape[1]}")
    print(f"{'=' * 60}")

    t0 = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
        eos_token_id=stop_tokens,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
    )
    dt = time.time() - t0

    decoded = processor.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("model\n")[-1].strip() if "model\n" in decoded else decoded
    print(f"({dt:.1f}s)")
    print(answer)
    return answer, dt


print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
stop_tokens = get_stop_tokens(processor)
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Stop tokens: {stop_tokens}")

print(f"\nExtracting slices from case: BraTS2021_00000")
print("  (using nib.as_closest_canonical + transpose + flipud for correct orientation)")
flair_slices, indices, n_total = nifti_to_slices(
    FLAIR_PATH, N_SLICES, save_debug=SAVE_DEBUG_SLICE
)
t1ce_slices, _, _ = nifti_to_slices(T1CE_PATH, N_SLICES)
print(f"  Volume depth: {n_total} slices, sampled z: {indices.tolist()}")
print(f"  ⚠️ 请打开 debug_slice_orientation.png 确认: "
      f"看到肿瘤在屏幕左侧, 前方在屏幕上方")

results = {}

if RUN_FLAIR_ONLY:
    prompt = (
        f"These are {N_SLICES} axial slices from a brain FLAIR MRI scan, "
        "sampled evenly from inferior to superior. "
        "The images follow the neurological display convention: "
        "the patient's LEFT side is on the LEFT of the image, "
        "and anterior is at the TOP. "
        "Describe any abnormalities and provide the most likely diagnosis. Be concise."
    )
    ans, dt = run_config(model, processor, flair_slices, prompt, "FLAIR only", stop_tokens)
    results["flair_only"] = {"answer": ans, "time_s": dt}
    torch.cuda.empty_cache()

if RUN_T1CE_ONLY:
    prompt = (
        f"These are {N_SLICES} axial slices from a brain post-contrast "
        "T1-weighted (T1ce) MRI scan, from inferior to superior. "
        "The images follow the neurological convention: "
        "the patient's LEFT side is on the LEFT of the image. "
        "Pay attention to enhancement patterns (ring, solid, none) and central necrosis. "
        "Describe abnormalities and provide the most likely diagnosis. Be concise."
    )
    ans, dt = run_config(model, processor, t1ce_slices, prompt, "T1ce only", stop_tokens)
    results["t1ce_only"] = {"answer": ans, "time_s": dt}
    torch.cuda.empty_cache()

if RUN_MULTIMODAL:
    interleaved = []
    for t1ce_img, flair_img in zip(t1ce_slices, flair_slices):
        interleaved.append(t1ce_img)
        interleaved.append(flair_img)

    prompt = (
        f"These are {N_SLICES * 2} axial slices from a brain MRI. "
        f"Odd-numbered images (1, 3, 5, ...) are post-contrast T1-weighted (T1ce), "
        f"even-numbered images are FLAIR, covering {N_SLICES} spatial locations "
        "from inferior to superior. "
        "Neurological convention: patient's LEFT is on the LEFT of the image. "
        "Use T1ce for enhancement and necrosis, FLAIR for edema. "
        "Describe abnormalities and provide the most likely diagnosis. Be concise."
    )
    ans, dt = run_config(model, processor, interleaved, prompt, "T1ce + FLAIR", stop_tokens)
    results["multimodal"] = {"answer": ans, "time_s": dt}
    torch.cuda.empty_cache()

print(f"\n\n{'=' * 60}")
print("SUMMARY — Ground truth: LEFT temporal GBM, ring-enhancing, ~100 cm³")
print(f"{'=' * 60}")

gt_keywords = {
    "laterality_left":    ["left"],
    "location_temporal":  ["temporal"],
    "mentions_gbm":       ["glioblastoma", "gbm", "high-grade glioma", "high grade glioma"],
    "mentions_ring":      ["ring", "rim-enhancing", "rim enhancement", "peripheral enhancement"],
    "mentions_necrosis":  ["necrosis", "necrotic"],
    "wrong_stroke":       ["stroke", "infarct", "ischemi"],
    "wrong_right_only":   [],
}

for config, data in results.items():
    ans_lower = data["answer"].lower()
    print(f"\n[{config}] ({data['time_s']:.1f}s)")

    scores = []
    for label, keywords in gt_keywords.items():
        if label == "wrong_right_only":
            hit = ("right" in ans_lower) and ("left" not in ans_lower)
        else:
            hit = any(kw in ans_lower for kw in keywords)
        mark = (
            "✅" if (hit and not label.startswith("wrong")) else
            "❌" if (not hit and not label.startswith("wrong")) else
            "⚠️" if hit else "  "
        )
        print(f"  {mark} {label}: {hit}")
        if not label.startswith("wrong"):
            scores.append(hit)

    print(f"  → {sum(scores)}/{len(scores)} correct")

print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")