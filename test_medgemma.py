import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "models\medgemma-1.5-4b-it"


print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the typical MRI features of glioblastoma?"}
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
decoded = processor.decode(output[0], skip_special_tokens=True)
answer = decoded.split("model\n")[-1].strip() if "model\n" in decoded else decoded

print(f"\nQ: What are the typical MRI features of glioblastoma?")
print(f"A: {answer[:500]}")
print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")