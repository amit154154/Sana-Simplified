from pathlib import Path
import os, gc, json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


# Root of your zoom dataset (folder that contains per-object folders)
ZOOM_ROOT = Path("/Users/mac/PycharmProjects/rotation_image_generation/data_zoom/dataset_zoom_10")

PROMPT = """
You are labeling 3D game assets rendered on a plain white background.

Your answer will be inserted directly after this text:
"a 3d game asset of "
so that the full sentence becomes:
"a 3d game asset of {YOUR ANSWER} rendered in a white background".

Your task:
- Look at the image and write a short noun phrase describing the main object.

STRICT RULES:
- Your answer MUST start with "a" or "an".
- Your answer MUST be at least 2 words (e.g. "a deer statue", not just "deer").
- Do NOT mention the background, lighting, camera, or render.
- Do NOT write a full sentence.
- Do NOT add quotes or any extra text.

Output ONLY the phrase that goes in the blank, nothing else.
"""


OUT_CSV   = "zoom0_smolvlm2_labels.csv"
OUT_JSONL = "zoom0_smolvlm2_labels.jsonl"
# ====================


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

# ---- Load SmolVLM2 (image+text -> text) ----
MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    low_cpu_mem_usage=True,
)
model.to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)


def run_smolvlm2(img_path: Path,
                 prompt: str,
                 max_new_tokens: int = 1024,
                 temperature: float = 0.9,
                 top_p: float = 0.8):
    """
    Single-image + text inference with SmolVLM2-2.2B-Instruct
    """
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "path": str(img_path.resolve())},
            {"type": "text",  "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=DTYPE)

    with torch.inference_mode():
        output_ids = model.generate(
            max_new_tokens=max_new_tokens,
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    text_out = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    if DEVICE == "mps":
        torch.mps.empty_cache()
    gc.collect()
    return text_out

def main():
    # Find all zoom_0 images: <ZOOM_ROOT>/<obj_name>/zoom_0.png
    zoom0_paths = sorted(ZOOM_ROOT.glob("*/zoom_0.png"))

    if not zoom0_paths:
        raise RuntimeError(f"No zoom_0.png files found under {ZOOM_ROOT}")

    rows = []
    for img_path in tqdm(zoom0_paths, desc="Labeling zoom_0 images"):
        obj_name = img_path.parent.name
        ans = run_smolvlm2(img_path, PROMPT)
        model_output = ans.split("Assistant:")[-1]
        rows.append({
            "object": obj_name,
            "image": str(img_path),
            "full_chat": ans,
            "model_output":model_output,
            "procssed_text_condition": f"a 3d game asset of {model_output} rendered in a white background"

        })
        print(ans)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} labels → {OUT_CSV} / {OUT_JSONL}")


if __name__ == "__main__":
    main()