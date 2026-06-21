import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import joblib
import tensorflow as tf

slide = "16B0024177"
val_file = "val.csv"
images = "./dataset_norm/SICAPv2"
patch_size = 512
overlap = 0.5
offset_pixels= patch_size * (1 - overlap)
original_step = 1024
factor = offset_pixels / original_step
features_extractor = "Virchow2"
model_classifier = "Red Propia"
features_val = "./features/val/features_virchow2val_norm.npy"
alpha = 160
threshold = 200
colours = {
    1: (255,255,0),
    2: (255,120,0),
    3: (255,0,0)
}

legend_labels = {
    0: "No Canceroso",
    1: "Grado de Gleason 3",
    2: "Grado de Gleason 4",
    3: "Grado de Gleason 5"
}

severity_exponents = np.array([1.0,1.0,0.85,0.7], dtype=np.float32)
epsilon=1e-5

def extract_coordinates(file):
    base_file = file.replace('.jpg','').split('_')
    x_ini = int(base_file[base_file.index('xini')+1])
    y_ini = int(base_file[base_file.index('yini')+1])
    return x_ini, y_ini

def add_legend(img, healthy_patch):
    img_w, img_h = img.size
    scale = max(img_w, img_h) / 5000.0
    font_size = int(100 * scale)
    rect_size = int(120 * scale)
    spacing = int(40 * scale)
    margin = int(40 * scale)
    outline_w = max(2, int(6*scale))
    font = "OpenSans-Regular.ttf"
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()
    
    num_items = len(legend_labels)
    
    draw_temp = ImageDraw.Draw(img)
    max_width = 0
    for label in legend_labels.values():
        box = draw_temp.textbbox((0,0), label, font=font)
        text_w = box[2] - box[0]
        max_width = max(max_width, text_w)
    box_w = int(rect_size + spacing + max_width + (margin * 2))
    box_h = int((rect_size + spacing) * num_items - spacing + (margin *2))

    gray_img = np.array(img.convert('L'))
    tissue_mask = gray_img < threshold

    corners = [
        (margin, margin),
        (img_w - box_w - margin, margin),
        (margin, img_h - box_h - margin),
        (img_w - box_w - margin, img_h - box_h - margin)
    ]

    b_x, b_y = margin, margin
    min_tissue = float('inf')
    best_position_found = False
    for x, y in corners:
        x,y = int(x), int(y)
        region = tissue_mask[max(0,y):min(img_h, y + box_h), max(0,x):min(img_w, x + box_w)]
        tissue_count = np.sum(region)

        if tissue_count < min_tissue:
            min_tissue = tissue_count
            b_x, b_y = x,y
        
        if tissue_count == 0:
            best_position_found = True
            break
        
        if not best_position_found:
            stride_x = max(50, img_w//20)
            stride_y = max(50, img_h//20)

            for y in range(margin, img_h - box_h - margin, stride_y):
                for x in range(margin, img_w - box_w - margin, stride_x):
                    region = tissue_mask[y:y + box_h, x:x + box_w]
                    tissue_count = np.sum(region)

                    if tissue_count < min_tissue:
                        min_tissue = tissue_count
                        b_x, b_y = x,y
                    
                    if tissue_count == 0:
                        best_position_found = True
                        break
                if best_position_found:
                    break
    
    box_x, box_y = b_x, b_y
    x_ini = box_x + margin
    y_ini = box_y + margin
    overlay = Image.new('RGBA', img.size, (255,255,255,0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill = (255,255,255,200))
    img = Image.alpha_composite(img.convert('RGBA'), overlay)

    draw = ImageDraw.Draw(img)

    current_y = y_ini
    for key, label in legend_labels.items():
        if key == 0:
            if healthy_patch:
               healthy_patch_resized = healthy_patch.resize((rect_size, rect_size), Image.Resampling.LANCZOS)
               img.paste(healthy_patch_resized, (x_ini, current_y))
               draw.rectangle([x_ini, current_y, x_ini + rect_size, current_y + rect_size], outline=(0,0,0), width = outline_w)
            else:
               draw.rectangle([x_ini, current_y, x_ini + rect_size, current_y + rect_size], fill = (230,230,230), outline=(0,0,0), width = outline_w)
        else:
            color = colours.get(key)
            if color:
                draw.rectangle([x_ini, current_y, x_ini + rect_size, current_y + rect_size], fill = color, outline=(0,0,0), width = outline_w)
        text_box = draw.textbbox((0,0), label, font = font)
        text_h =text_box[3] - text_box[1]
        text_y = current_y + (rect_size - text_h) // 2
        text_y -= int(20 * scale)
        draw.text((x_ini + rect_size + spacing, text_y), label, fill = (0,0,0), font=font)
        current_y += rect_size + spacing
    return img.convert('RGB')

def votes(votes_matrix, width, height, colours, alpha):
    winner = np.zeros((height, width), dtype=np.uint8)
    max_votes = np.zeros((height, width), dtype=np.uint8)

    for c in [0,1,2,3]:
        update_mask = (votes_matrix[:,:,c] >= max_votes) & (votes_matrix[:,:,c] > 0)
        winner[update_mask] = c
        max_votes[update_mask] = votes_matrix[:,:,c][update_mask]
    
    heatmap = np.zeros((height, width, 4), dtype= np.uint8)
    for c in [1,2,3]:
        color = colours[c]
        class_mask = (winner == c)
        heatmap[class_mask] = (color[0], color[1], color[2], alpha)

    return Image.fromarray(heatmap, 'RGBA') 

def cwca(score_matrix, width, height, colours, alpha):
    winner = np.argmax(score_matrix, axis=2)
    max_score = np.max(score_matrix, axis=2)
    valid_mask = max_score > 0

    heatmap = np.zeros((height, width, 4), dtype=np.uint8)
    for c in [1,2,3]:
        color = colours[c]
        class_mask = (winner == c) & valid_mask
        heatmap[class_mask] = (color[0], color[1], color[2], alpha)
    
    return Image.fromarray(heatmap, 'RGBA') 

df_val = pd.read_csv(val_file)
X_val = np.load(features_val)
slide_indexes = df_val.index[df_val['image_name'].str.startswith(slide)].to_list()
df_slide = df_val.iloc[slide_indexes].copy()

if df_slide.empty:
    print("Error")
    exit()

X_slide = X_val[slide_indexes]
scaler_path = f"./trained_models/scaler_{features_extractor}_norm.joblib"
model_path = f"./trained_models/{model_classifier}_{features_extractor}_norm.keras"
scaler = joblib.load(scaler_path)
model = tf.keras.models.load_model(model_path)
X_slide_s = scaler.transform(X_slide)
y_probability = model.predict(X_slide_s, verbose=0)
predictions = np.argmax(y_probability, axis=1)
df_slide['prediction'] = predictions

probabilities_safe = np.clip(y_probability, 1e-9, 1.0)
entropies = -np.sum(probabilities_safe * np.log2(probabilities_safe), axis=1)
cwca_weights = 1.0 / (entropies + epsilon)
adjusted_probabilities = probabilities_safe ** severity_exponents

x_list = []
y_list = []
for _, row in df_slide.iterrows():
    name = row['image_name']
    x,y = extract_coordinates(name)
    x_list.append(x)
    y_list.append(y)

x_min = min(x_list)
y_min = min(y_list)
x_max = max(x_list)
y_max = max(y_list)

width_image = int((x_max - x_min)*factor) + patch_size
height_image = int((y_max - y_min)*factor) + patch_size
background_image = Image.new('RGBA', (width_image, height_image), (255,255,255,0))
votes_ground_truth = np.zeros((height_image,width_image,4), dtype=np.uint8)
cwca_scores = np.zeros((height_image,width_image,4), dtype=np.float32)

healthy_patch = None
max_pixels = -1
for idx, (original_idx, row) in enumerate(df_slide.iterrows()):
    name = row['image_name']
    real_label = row['label']
    x, y = extract_coordinates(name)
    x_image = int((x - x_min)*factor)
    y_image = int((y - y_min)*factor)
    patch_path = os.path.join(images, name)

    if os.path.exists(patch_path):
        patch_image = Image.open(patch_path).convert('RGBA')
        patch_image = patch_image.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
        background_image.paste(patch_image, (x_image, y_image), patch_image)
        patch_rgb = np.array(patch_image.convert('RGB'))
        gray_p = np.mean(patch_rgb, axis=2)
        mask = gray_p < threshold
        pixels = np.sum(mask)
        if pixels > max_pixels:
            max_pixels = pixels
            healthy_patch = patch_image.copy()

        final_x, final_y = x_image + patch_size, y_image + patch_size

        if real_label in [0,1,2,3]:
            votes_slice_ground_truth = votes_ground_truth[y_image:final_y, x_image:final_x, real_label]
            votes_slice_ground_truth[mask] += 1
        
        patch_weight = cwca_weights[idx]
        patch_adjusted_probabilities = adjusted_probabilities[idx]
        for c in [0,1,2,3]:
            score_slice = cwca_scores[y_image:final_y, x_image:final_x, c]
            score_slice[mask] += (patch_weight * patch_adjusted_probabilities[c])


heatmap_ground_truth = votes(votes_ground_truth,width_image, height_image, colours ,alpha)
heatmap_prediction = cwca(cwca_scores,width_image, height_image, colours ,alpha)

original_img = background_image.convert('RGB')
original_img.thumbnail((5000,5000), Image.Resampling.LANCZOS)
output = f"Cwca_normalized_heatmap_{slide}_{features_extractor}_{model_classifier}_Original.jpg"
original_img.save(output, quality=95)

ground_truth_img = Image.alpha_composite(background_image, heatmap_ground_truth).convert('RGB')
ground_truth_img.thumbnail((5000,5000), Image.Resampling.LANCZOS)
ground_truth_img = add_legend(ground_truth_img, healthy_patch)
output = f"Cwca_normalized_heatmap_{slide}_{features_extractor}_{model_classifier}_GroundTruth.jpg"
ground_truth_img.save(output, quality=95)

prediction_img = Image.alpha_composite(background_image, heatmap_prediction).convert('RGB')
prediction_img.thumbnail((5000,5000), Image.Resampling.LANCZOS)
prediction_img = add_legend(prediction_img, healthy_patch)
output = f"Cwca_normalized_heatmap_{slide}_{features_extractor}_{model_classifier}_Prediction.jpg"
prediction_img.save(output, quality=95)