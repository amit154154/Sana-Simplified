#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===========================
# CONFIG
# ===========================

# Root of your zoom dataset: <ZOOM_ROOT>/<object_name>/zoom_0.png
ZOOM_ROOT = Path("/home/ubuntu/AMIT/data/dataset_zoom_30mb_notfiltered_10")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUT_CSV  = "/home/ubuntu/AMIT/data/zoom_30mb_notfiltered_labels.csv"

QUESTION = """
You are looking at a single 3D game asset rendered on a plain white background.

Your job is to write a high-quality text prompt that can be given directly to a text-to-image
model to generate an image that closely matches what you see.

The prompt MUST:
- Be a single descriptive sentence, not an instruction.
- Start directly with a noun phrase such as
  "3D game asset render of ..." or "A 3D game asset render of ...".
- Clearly describe the main object: type, pose, style, material, colors, and any
  distinctive features (e.g., "low-poly", "stylized", "realistic", "cartoonish").
- Explicitly mention that it is a 3D game asset render on a plain white background.

STRICT RULES:
- Do NOT use imperative verbs like "Create", "Generate", "Imagine", "Design", etc.
- Do NOT talk about the viewer or the image (NO "in this image", "you see",
  "the picture shows", etc.).
- Do NOT add meta-instructions or commentary.
- Do NOT include quotes around the prompt or any extra text before/after it.

Output ONLY the final descriptive prompt sentence, nothing else.
""".strip()

# How often to flush new rows to disk (so you can safely quit)
SAVE_EVERY = 1000

# Batch size for generation (tune this based on VRAM; with L40S, 4–8 is safe)
BATCH_SIZE = 128

# You only need a single sentence; no need for 128 tokens
MAX_NEW_TOKENS = 64


def main():
    print(">> Scanning dataset...")
    zoom0_paths = sorted(ZOOM_ROOT.glob("*/zoom_0.png"))
    if not zoom0_paths:
        raise RuntimeError(f"No zoom_0.png files found under {ZOOM_ROOT}")

    print(f">> Found {len(zoom0_paths)} images.")

    # ===========================
    # Resume logic: load existing CSV if present
    # ===========================
    labeled_objects = set()
    existing_df = None

    out_path = Path(OUT_CSV)
    if out_path.exists():
        print(f">> Found existing CSV at {OUT_CSV}, loading to resume...")
        existing_df = pd.read_csv(out_path)
        if not {"object", "label"}.issubset(existing_df.columns):
            raise RuntimeError(
                f"Existing CSV {OUT_CSV} does not have required columns ['object', 'label']"
            )
        labeled_objects = set(existing_df["object"].astype(str))
        print(f">> Already have {len(labeled_objects)} labeled objects, will skip them.")
    else:
        print(">> No existing CSV found, starting fresh.")
        existing_df = pd.DataFrame(columns=["object", "label"])

    # ===========================
    # Init processor & model
    # ===========================
    print(">> Initializing processor & model...")

    # Your images are 1024x1024, but we cap effective pixels to ~512x512 area
    min_pixels = 256 * 256
    max_pixels = 512 * 512

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # If you later want a smaller model, just change MODEL_ID to the 3B variant.
    # Or you can add quantization here, but keeping it simple for now.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    device = model.device if hasattr(model, "device") else "cuda"
    print(f">> Model loaded on: {device}")

    # ===========================
    # Helper to run a batch
    # ===========================
    def run_batch(batch_messages, batch_objects, new_rows, existing_df):
        if not batch_messages:
            return new_rows, existing_df

        # 1) Apply chat template for each conversation in the batch
        texts = [
            processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conv in batch_messages
        ]

        # 2) Vision inputs for all messages in this batch
        image_inputs, video_inputs = process_vision_info(batch_messages)

        # 3) Tokenize & move to device
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 4) Generate for the whole batch
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # 5) Strip prompt tokens, keep only generated continuation
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # 6) Collect rows
        for obj_name, label in zip(batch_objects, output_texts):
            label = label.strip()
            new_rows.append({"object": obj_name, "label": label})

        # 7) Periodically flush to disk
        if len(new_rows) >= SAVE_EVERY:
            print(">> Saving intermediate results to CSV...")
            df_new = pd.DataFrame(new_rows)
            existing_df = pd.concat([existing_df, df_new], ignore_index=True)
            existing_df.to_csv(OUT_CSV, index=False)
            new_rows = []

        return new_rows, existing_df

    # ===========================
    # Iterate over images (batched)
    # ===========================
    total_to_label = sum(
        1 for p in zoom0_paths if p.parent.name not in labeled_objects
    )
    print(f">> Need to label {total_to_label} more objects.")

    new_rows = []
    batch_messages = []
    batch_objects = []

    for img_path in tqdm(zoom0_paths, desc="Labeling zoom_0 images"):
        obj_name = img_path.parent.name

        # Skip if already labeled in previous runs
        if obj_name in labeled_objects:
            continue

        # (Optional) sanity check that image is readable
        # Image.open(img_path).convert("RGB")

        # Each conversation is a list of messages (here: single user turn)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(img_path),  # local path
                    },
                    {
                        "type": "text",
                        "text": QUESTION,
                    },
                ],
            }
        ]

        batch_messages.append(messages)
        batch_objects.append(obj_name)
        labeled_objects.add(obj_name)

        # If we've filled a batch, run it
        if len(batch_messages) >= BATCH_SIZE:
            new_rows, existing_df = run_batch(
                batch_messages, batch_objects, new_rows, existing_df
            )
            batch_messages, batch_objects = [], []

    # Run the final partial batch, if any
    new_rows, existing_df = run_batch(
        batch_messages, batch_objects, new_rows, existing_df
    )

    # ===========================
    # Final save
    # ===========================
    if new_rows:
        print(">> Saving final batch of results to CSV...")
        df_new = pd.DataFrame(new_rows)
        existing_df = pd.concat([existing_df, df_new], ignore_index=True)

    existing_df.to_csv(OUT_CSV, index=False)
    print(f">> Done. Total labels in {OUT_CSV}: {len(existing_df)}")


if __name__ == "__main__":
    main()