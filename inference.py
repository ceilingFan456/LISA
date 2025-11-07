import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from tqdm import tqdm

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    # -------------------------
    # BATCH MODE (minimal edits)
    # -------------------------
    from pathlib import Path
    PROMPT_TO_CODE = {"robot": "000", "gripper": "001", "robot arm": "002"}
    IN_ROOT  = Path("/home/t-qimhuang/disk/robot_dataset/final_test_set/run5_test105")        # input root (has image/)
    OUT_BASE = Path("/home/t-qimhuang/disk/robot_dataset/final_test_set/run5_test105_lisa")   # output base

    def _ensure_dir(d: Path):
        os.makedirs(d, exist_ok=True)

    print("Starting batch inference...")

    for image_path in tqdm(sorted((IN_ROOT / "image").rglob("*.jpg")), desc="Processing images"):
        # print(f"Processing image: {image_path}")
        # case name = first folder after "image/"
        try:
            parts = image_path.resolve().relative_to(IN_ROOT.resolve()).parts
        except Exception:
            print(f"[WARN] skip (not under IN_ROOT): {image_path}")
            continue
        if len(parts) < 2 or parts[0] != "image":
            print(f"[WARN] skip (unexpected path): {image_path}")
            continue

        case_name = parts[1]
        frame_name = image_path.name  # e.g., 00014.jpg

        # ======= image preproc (unchanged) =======
        image_np_bgr = cv2.imread(str(image_path))
        if image_np_bgr is None:
            print(f"[WARN] Cannot read image: {image_path}")
            continue
        image_np = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()
        # =========================================

        for user_text, code in PROMPT_TO_CODE.items():
            # ======= prompt build (unchanged logic) =======
            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + user_text
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()
            # ==============================================

            # ======= evaluation (unchanged) =======
            output_ids, pred_masks, iou_predictions = model.evaluate_with_iou(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=tokenizer,
            )
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

            # print("Predicted IoU:", iou_predictions)

            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            # =====================================

            # ======= combine masks & save outputs =======
            # combined = None
            # for i, pred_mask in enumerate(pred_masks):
            #     if pred_mask.shape[0] == 0:
            #         continue
            #     m = pred_mask.detach().cpu().numpy()[0] > 0
            #     combined = m if combined is None else (combined | m)

            # if combined is None:
            #     h, w = image_np.shape[:2]
            #     combined = np.zeros((h, w), dtype=bool)

            # Instead of combining all masks, just use the one with highest IoU
            if len(pred_masks) == 0:
                h, w = image_np.shape[:2]
                combined = np.zeros((h, w), dtype=bool)
            else:
                best_idx = int(torch.argmax(iou_predictions).item())
                pred_mask = pred_masks[best_idx]
                combined = pred_mask.detach().cpu().numpy()[0] > 0

            # 1) save binary mask
            mask_dir = OUT_BASE / "mask" / case_name / code
            _ensure_dir(mask_dir)
            mask_path = mask_dir / frame_name
            ok = cv2.imwrite(str(mask_path), (combined.astype(np.uint8) * 255))
            if not ok:
                print(f"[ERROR] Failed to save mask: {mask_path}")

            # 2) save ORIGINAL image (no code folder)
            img_dir = OUT_BASE / "image" / case_name
            _ensure_dir(img_dir)
            img_path_out = img_dir / frame_name
            ok = cv2.imwrite(str(img_path_out), image_np_bgr)
            if not ok:
                print(f"[ERROR] Failed to save image: {img_path_out}")

            # 3) save MASKED ORIGINAL (transparent red overlay for viz)
            masked_dir = OUT_BASE / "masked" / case_name / code
            _ensure_dir(masked_dir)
            masked_path = masked_dir / frame_name

            # combined: bool mask (H, W). image_np_bgr: original BGR
            alpha = 0.5  # 0=transparent, 1=solid
            overlay = image_np_bgr.copy()
            overlay[combined] = (0, 0, 255)  # red in BGR
            vis_bgr = cv2.addWeighted(overlay, alpha, image_np_bgr, 1 - alpha, 0)

            ok = cv2.imwrite(str(masked_path), vis_bgr)
            if not ok:
                print(f"[ERROR] Failed to save masked image: {masked_path}")



if __name__ == "__main__":
    main(sys.argv[1:])
