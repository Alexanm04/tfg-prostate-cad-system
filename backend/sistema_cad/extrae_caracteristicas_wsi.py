import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import gc
from trident import load_wsi
from trident.segmentation_models import  segmentation_model_factory
from conch.open_clip_custom import create_model_from_pretrained

model_selected="conch"
phase="test_wsi"
wsi_dir = os.path.expanduser("~/WSIs_y_Máscaras/wsis")
mask_dir = os.path.expanduser("~/WSIs_y_Máscaras/masks")
output_dir = f"./features/{phase}"
os.makedirs(output_dir, exist_ok=True)
patch_size = 512
overlap = 0.5
scale_factor = 4
original_patch_size = patch_size * scale_factor
min_tissue = 0.2
batch_size = 32

if model_selected == "conch":
    token="tokenfalso"
    model, transformation  = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token= token)
else:
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

seg_model = segmentation_model_factory("hest")
seg_model = seg_model.to(device)
seg_model.eval()

def get_patch_label(mask_patch):
    count_128 = np.sum(mask_patch == 128)
    count_64  = np.sum(mask_patch == 64)
    count_32  = np.sum(mask_patch == 32)

    if count_128 + count_64 + count_32 > 0:
        return int(np.argmax([count_128, count_64, count_32])) + 1
    else:
        return 0

def run_batch(batch_tensors, features):
    feats = model.encode_image(
        torch.stack(batch_tensors).to(device, non_blocking=True),
        proj_contrast=False, normalize=False
    )
    features.append(feats.detach().cpu().float().numpy())

wsi_files = ["CasoHEProstata_A_0965.tiff"]
total_wsis = len(wsi_files)
Image.MAX_IMAGE_PIXELS=None

for i, wsi_filename in enumerate(wsi_files,1):
    wsi_id = wsi_filename.split('.')[0]
    print(f"[{i}/{total_wsis}] Procesando WSI: {wsi_id}...")
    output_csv = f"{output_dir}/{wsi_id}_labels_and_coords.csv"
    print(f"{i}/{total_wsis} -> Procesando {wsi_id}")
    wsi_path = os.path.join(wsi_dir, wsi_filename)
    mask_path = os.path.join(mask_dir, f"{wsi_id}.png")

    mask_image = np.array(Image.open(mask_path).convert('L'))
    mask_h, mask_w = mask_image.shape[:2]

    with load_wsi(wsi_path, lazy_init=False) as slide:
        slide.mag = 40
        slide.mpp = 0.25
        
        contours_gdf = slide.segment_tissue(
            segmentation_model = seg_model,
            target_mag = 10,
            holes_are_tissue=False,
            job_dir=None,
            batch_size=16,
            device=str(device)
        )

        patcher = slide.create_patcher(
            patch_size = patch_size,
            dst_mag = 10,
            src_mag = 40,
            overlap = int(patch_size * overlap),
            mask = contours_gdf,
            threshold = min_tissue, 
            pil = True
        )
        features_l = []
        records = []
        batch_records = []
        batch_tensors = []
        total_seen = 0
        total_valid = 0

        for pil_image, x, y in patcher:
            print(f"x={x}, y={y}, mask_w={mask_w}, mask_h={mask_h}")
            total_seen += 1

            x2_40 = min(x + original_patch_size, mask_w)
            y2_40 = min(y + original_patch_size, mask_h)
            mask_crop_40 = mask_image[y:y2_40, x:x2_40]

            target_w = mask_crop_40.shape[1] // scale_factor
            target_h = mask_crop_40.shape[0] // scale_factor
            if target_h == 0 or target_w == 0:
                continue
            mask_crop_10 = np.array(
                Image.fromarray(mask_crop_40).resize(
                    (target_w, target_h), Image.Resampling.NEAREST
                )
            )

            final_mask = np.zeros((patch_size, patch_size), dtype = mask_image.dtype)
            h_lim = min(target_h, patch_size)
            w_lim = min(target_w, patch_size)
            final_mask[:h_lim, :w_lim] = mask_crop_10[:h_lim, :w_lim]
            true_label = get_patch_label(final_mask)

            batch_tensors.append(transformation(pil_image))
            batch_records.append({
                'wsi_id': wsi_id,
                'x_ini': x,
                'y_ini': y,
                'label': true_label
            })
            total_valid +=1

            if len(batch_tensors) == batch_size:
                run_batch(batch_tensors, features_l)
                records.extend(batch_records)
                batch_tensors, batch_records = [], []
                print(f"Parches válidos extraidos: {total_valid} y Parches vistos: {total_seen}", end='\r')

        if batch_tensors:
            run_batch(batch_tensors, features_l)
            records.extend(batch_records)

    print(f"\nResumen {wsi_id}: válidos={total_valid}, vistos={total_seen}")
    if features_l:
        final_features = np.concatenate(features_l, axis=0)
        np.save(f"{output_dir}/{wsi_id}_features_{model_selected}.npy", final_features)
        pd.DataFrame(records).to_csv(f"{output_dir}/{wsi_id}_labels_and_coords.csv", index=False)
        print(f"WSI {wsi_id} completada: {total_valid} parches guardados")

    del mask_image
    gc.collect()
    torch.cuda.empty_cache()

print("¡Extracción de todas las WSI completada!")