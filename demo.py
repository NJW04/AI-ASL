#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference on a live webcam feed using TWO trained CNNSmall models.
Draws a Region of Interest (ROI) box and only predicts the content inside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import cv2  # OpenCV
from PIL import Image

from data.transforms import get_eval_transforms
from models.cnn_small import CNNSmall # Assumes cnn_small.py is in models/

# --- Helper to load class names (no change) ---
def load_class_names(class_json_path: Path) -> List[str]:
    """Loads class names, sorting by index."""
    try:
        with open(class_json_path, "r") as f:
            class_to_idx = json.load(f)
        class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
        return class_names
    except FileNotFoundError:
        print(f"ERROR: Cannot find class indices: {class_json_path}")
        print("Please ensure 'class_indices.json' is in your 'data/' folder.")
        return None

# --- Helper to build the model from config (no change) ---
def load_model_from_checkpoint(checkpoint_path: Path) -> tuple[CNNSmall, int, List[str]]:
    """Loads a model, its config, and class names from a checkpoint path."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        return None, 0, None

    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found in: {checkpoint_path.parent}")
        return None, 0, None
        
    with open(config_path, "r") as f:
        cfg = json.load(f)

    class_names = load_class_names(Path("data") / "class_indices.json")
    if class_names is None:
        return None, 0, None

    num_classes = len(class_names)
    model_num_blocks = cfg.get("num_blocks", 3)
    model_activation = cfg.get("activation", "relu")
    model_dropout = cfg.get("dropout", 0.3)
    data_size = cfg.get("size", 96)

    model = CNNSmall(
        num_classes=num_classes,
        num_blocks=model_num_blocks,
        activation=model_activation,
        dropout=model_dropout
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path.parent.name}")
    print(f"Params: blocks={model_num_blocks}, activation={model_activation}, size={data_size}")

    return model, data_size, class_names, device

# --- Main inference function ---
def main():
    ap = argparse.ArgumentParser(description="Live webcam inference for ASL")
    ap.add_argument("--checkpoint-aug", type=Path,
                    default=Path(r"artifacts\asl_runs\2025-10-28_19-51-51__train-cnn_small_lr0.00191509_b128_bl4_act-gelu_dr0.0696056_ep20_aug_sz128\best.pt"),
                    help="Path to the 'best.pt' file for the AUGMENTED model.")
    ap.add_argument("--checkpoint-no-aug", type=Path,
                    default=Path(r"artifacts\asl_runs\2025-10-28_19-13-37__train-cnn_small_lr0.00191509_b128_bl4_act-gelu_dr0.0696056_ep20_sz128\best.pt"),
                    help="Path to the 'best.pt' file for the NON-AUGMENTED model.")
    args = ap.parse_args()

    # 1. Load Models
    model_aug, data_size_aug, class_names, device = load_model_from_checkpoint(args.checkpoint_aug)
    if model_aug is None: return
    model_no_aug, data_size_no_aug, _, _ = load_model_from_checkpoint(args.checkpoint_no_aug)
    if model_no_aug is None: return
        
    if data_size_aug != data_size_no_aug:
        print(f"ERROR: Models have different image sizes ({data_size_aug} vs {data_size_no_aug}).")
        return
    data_size = data_size_aug
    
    # 2. Get transforms
    transforms = get_eval_transforms(size=data_size, normalize="imagenet")

    # 3. Initialize Webcam
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Webcam ON ---")
    print("Place your hand inside the green box.")
    print("Press [SPACE] to predict.")
    print("Press [Q] to quit.")
    
    last_pred_aug = "..."
    last_pred_no_aug = "..."

    # --- NEW: Define the Region of Interest (ROI) Box ---
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        cap.release()
        return
    
    frame_h, frame_w = frame.shape[:2]
    
    # Define box size (e.g., 300x300 pixels)
    box_size = 300
    # Center the box
    x1 = (frame_w - box_size) // 2
    y1 = (frame_h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    # --- END NEW ---

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Flip horizontally

        # --- NEW: Get the cropped frame for inference ---
        # IMPORTANT: Crop the *original* frame *before* drawing on it
        frame_crop = frame[y1:y2, x1:x2]
        # --- END NEW ---

        # --- Draw UI elements on the display frame ---
        # Draw the ROI box (green)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw the prediction text box (white)
        cv2.rectangle(frame, (0, 0), (550, 80), (255, 255, 255), -1) 
        cv2.putText(frame, f"Aug Model: {last_pred_aug}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(frame, f"No-Aug Model: {last_pred_no_aug}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Show the single, annotated frame
        cv2.imshow("ASL Predictor - Press [SPACE] to test, [Q] to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord(' '):
            print("Predicting...")
            
            # --- MODIFICATION: Process the *cropped* frame ---
            rgb_frame = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB) # Use frame_crop
            pil_img = Image.fromarray(rgb_frame)
            img_tensor = transforms(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                # --- Model 1: Augmented ---
                logits_aug = model_aug(img_tensor)
                probs_aug = torch.softmax(logits_aug, dim=1)
                pred_idx_aug = logits_aug.argmax(dim=1).item()
                pred_prob_aug = probs_aug[0, pred_idx_aug].item()
                
                last_pred_aug = f"{class_names[pred_idx_aug]} ({pred_prob_aug*100:.1f}%)"
                print(f"  -> Aug Model: {last_pred_aug}")

                # --- Model 2: Non-Augmented ---
                logits_no_aug = model_no_aug(img_tensor)
                probs_no_aug = torch.softmax(logits_no_aug, dim=1)
                pred_idx_no_aug = logits_no_aug.argmax(dim=1).item()
                pred_prob_no_aug = probs_no_aug[0, pred_idx_no_aug].item()
                
                last_pred_no_aug = f"{class_names[pred_idx_no_aug]} ({pred_prob_no_aug*100:.1f}%)"
                print(f"  -> No-Aug Model: {last_pred_no_aug}")
            # --- END MODIFICATION ---

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam OFF. Exiting.")

if __name__ == "__main__":
    main()