import numpy as np
import rasterio
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Paramètres
patch_size = (1072, 656)
#model_path = 'unet2_72_best_model.h5'  # <- Mets ici le bon chemin local
import tensorflow as tf
model = tf.keras.models.load_model("unet2_72_best_model.h5", compile=False)



# Palette de changement (publique)
change_palette = [
    '#FF6347', '#FFD700', '#32CD32', '#8B4513', '#A52A2A',
    '#4682B4', '#1E90FF', '#006400', 
    '#D3D3D3', '#808080', '#000000',
]
change_names = [
    'Urbanisation', 'Artificialisation', 'Mise en culture', 'Abandon de culture',
    'Déforestation', 'Assèchement', 'Inondation', 'Forestation', 'Pas de changement', 'Non classifié', 'Background',
]

# ---------- Fonctions principales ----------

import tempfile

# def preprocess_image(img_file, target_size=(1072, 656)):
#     # Sauvegarder le fichier temporairement
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
#         tmp.write(img_file.read())
#         tmp_path = tmp.name

#     # Lire avec rasterio depuis le fichier temporaire
#     with rasterio.open(tmp_path) as src:
#         b2 = src.read(1).astype(np.float32)
#         b3 = src.read(2).astype(np.float32)
#         b4 = src.read(3).astype(np.float32)
#         b8 = src.read(4).astype(np.float32)
#         b11 = src.read(5).astype(np.float32)

#         b2 = np.nan_to_num(b2)
#         b3 = np.nan_to_num(b3)
#         b4 = np.nan_to_num(b4)
#         b8 = np.nan_to_num(b8)
#         b11 = np.nan_to_num(b11)

#         ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
#         ndwi = (b3 - b8) / (b3 + b8 + 1e-10)
#         ndbi = (b11 - b8) / (b11 + b8 + 1e-10)

#         image = np.dstack((b4, b3, b2, b8, b11, ndvi, ndwi, ndbi))
#         resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
#     return resized
from rasterio.io import MemoryFile
import numpy as np
import cv2

import rasterio
import numpy as np
import cv2

def preprocess_image(uploaded_file, target_size=(1072, 656)):
    # Passer directement l'objet uploadé à rasterio
    with rasterio.open(uploaded_file) as src:
        b2 = src.read(1).astype(np.float32)
        b3 = src.read(2).astype(np.float32)
        b4 = src.read(3).astype(np.float32)
        b8 = src.read(4).astype(np.float32)
        b11 = src.read(5).astype(np.float32)

        b2 = np.nan_to_num(b2, nan=0)
        b3 = np.nan_to_num(b3, nan=0)
        b4 = np.nan_to_num(b4, nan=0)
        b8 = np.nan_to_num(b8, nan=0)
        b11 = np.nan_to_num(b11, nan=0)

        ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
        ndwi = (b3 - b8) / (b3 + b8 + 1e-10)
        ndbi = (b11 - b8) / (b11 + b8 + 1e-10)

        image = np.dstack((b4, b3, b2, b8, b11, ndvi, ndwi, ndbi))
        resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    return resized



def image_to_patches(image, patch_size):
    h, w, c = image.shape
    ph, pw = patch_size
    patches, positions = [], []
    for i in range(0, h, ph):
        for j in range(0, w, pw):
            patch = image[i:i+ph, j:j+pw, :]
            if patch.shape[:2] == (ph, pw):
                patches.append(patch)
                positions.append((i, j))
    return np.array(patches), positions

def reconstruct_mask(pred_patches, positions, image_shape, patch_size):
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    ph, pw = patch_size
    for pred, (i, j) in zip(pred_patches, positions):
        full_mask[i:i+ph, j:j+pw] = np.argmax(pred, axis=-1)
    return full_mask

# def predict_mask_for_image(image_file):
#     if model is None:
#         raise RuntimeError("❌ Le modèle n'a pas été chargé correctement. Vérifiez le fichier .h5.")

#     image = preprocess_image(image_file)
#     patches, positions = image_to_patches(image, patch_size)
#     predictions = []
#     for patch in patches:
#         patch_input = np.expand_dims(patch, axis=0)
#         pred = model.predict(patch_input, verbose=0)
#         predictions.append(pred[0])
#     return reconstruct_mask(predictions, positions, image.shape, patch_size)

def predict_mask_for_image(image_file):
    if model is None:
        raise RuntimeError("❌ Le modèle n'a pas été chargé correctement. Vérifiez le fichier .h5.")

    image = preprocess_image(image_file)
    patches, positions = image_to_patches(image, patch_size)

    # Prédiction en batch pour accélérer le CPU
    predictions = model.predict(patches, verbose=0)

    return reconstruct_mask(predictions, positions, image.shape, patch_size)
import re

def extract_year_from_filename(filename):
    # Cherche une année entre 2000 et 2099 dans le nom du fichier
    match = re.search(r'(20\d{2})', filename)
    if match:
        return match.group(1)
    else:
        return "Année inconnue"

def show_image_and_mask(image_file, predicted_mask, title):
    import matplotlib.patches as mpatches
    dw_palette = [
        '#419bdf', '#397d49', '#88b053', '#7a87c6', '#e49635',
        '#dfc35a', '#c4281b', '#a59b8f', '#b39fe1', '#000000'
    ]
    dw_labels = [
        "0: Eau", "1: Arbres", "2: Herbe", "3: Végétation inondée",
        "4: Cultures", "5: Arbustes", "6: Construit", "7: Sol nu",
        "8: Neige / Glace", "9: Fond (Background)"
    ]
    image = preprocess_image(image_file)
    rgb_image = image[:, :, [2, 1, 0]]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-10)
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(rgb_image)
    ax1.set_title(f"{title} - Image RGB")
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    cmap = ListedColormap(dw_palette)
    ax2.imshow(predicted_mask, cmap=cmap, vmin=0, vmax=len(dw_palette)-1)
    ax2.set_title(f"{title} - Masque Prédit")
    ax2.axis('off')
    return fig

def detect_change(mask1, mask2):
    change_map = np.zeros_like(mask1)
    change_map[np.logical_and(mask1 == 1, mask2 == 6)] = 0  # Urbanisation
    change_map[np.logical_and(mask1 == 5, mask2 == 6)] = 0
    change_map[np.logical_and(mask1 == 4, mask2 == 6)] = 1  # Artificialisation
    change_map[np.logical_and(mask1 == 7, mask2 == 4)] = 2  # Mise en culture
    change_map[np.logical_and(mask1 == 5, mask2 == 4)] = 2
    change_map[np.logical_and(mask1 == 4, mask2 == 7)] = 3  # Abandon de culture
    change_map[np.logical_and(mask1 == 4, mask2 == 5)] = 3
    change_map[np.logical_and(mask1 == 1, mask2 == 7)] = 4  # Déforestation
    change_map[np.logical_and(mask1 == 1, mask2 == 4)] = 4
    change_map[np.logical_and(mask1 == 1, mask2 == 5)] = 4
    change_map[np.logical_and(mask1 == 0, mask2 == 7)] = 5  # Assèchement
    change_map[np.logical_and(mask1 == 7, mask2 == 0)] = 6  # Inondation
    change_map[np.logical_and(mask1 == 4, mask2 == 1)] = 7  # Forestation
    change_map[np.logical_and(mask1 == 7, mask2 == 1)] = 7
    change_map[np.logical_and(mask1 == mask2, change_map == 0)] = 8  # Pas de changement
    change_map[np.logical_and(mask1 != mask2, change_map == 0)] = 9  # Non classifié
    change_map[np.logical_or(mask1 == 9, mask2 == 9)] = 10  # Background
    return change_map

def analyze_change_map(change_map, pixel_size=10):
    labels = {
        0: 'Urbanisation (Arbres → Construit)',
        1: 'Artificialisation (Cultures → Construit)',
        2: 'Mise en culture (Sol nu → Cultures)',
        3: 'Abandon de culture (Cultures → Sol nu)',
        4: 'Déforestation (Arbres → Cultures)',
        5: 'Assèchement (Eau → Sol nu)',
        6: 'Inondation (Sol nu → Eau)',
        7: 'Forestation (Cultures → Arbres)',
        8: 'Pas de changement',
        9: 'Changement non classifié',
        10: 'Background'
    }
    total_pixels = np.count_nonzero(change_map != 10)
    total_area = total_pixels * pixel_size**2
    areas = {
        labels[i]: np.count_nonzero(change_map == i) * pixel_size**2 / 1e6
        for i in labels
    }
    percentages = {k: (v * 1e6 / total_area) * 100 for k, v in areas.items() if total_area > 0}

    df = pd.DataFrame({'Superficie (km²)': areas, 'Pourcentage (%)': percentages})
    return df.round(2)

# def compute_area_table(mask1, mask2, pixel_size=10):
#     class_labels = {
#         0: "Eau", 1: "Arbres", 2: "Herbe", 3: "Végétation inondée",
#         4: "Cultures", 5: "Arbustes", 6: "Construit", 7: "Sol nu", 8: "Neige / Glace"
#     }
#     def compute_area(mask):
#         return {
#             class_labels[i]: np.count_nonzero(mask == i) * pixel_size**2 / 1e6
#             for i in class_labels
#         }
#     area_2017 = pd.Series(compute_area(mask1), name="T1 (km²)")
#     area_2019 = pd.Series(compute_area(mask2), name="T2 (km²)")
#     return pd.concat([area_2017, area_2019], axis=1).round(2)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def analyze_change_map(change_map, pixel_size=10):
    change_classes = {
        0: 'Urbanisation (Arbres → Construit)',
        1: 'Artificialisation (Cultures → Construit)',
        2: 'Mise en culture (Sol nu → Cultures)',
        3: 'Abandon de culture (Cultures → Sol nu)',
        4: 'Déforestation (Arbres → Cultures)',
        5: 'Assèchement (Eau → Sol nu)',
        6: 'Inondation (Sol nu → Eau)',
        7: 'Forestation (Cultures → Arbres)',
        8: 'Pas de changement',
        9: 'Changement non classifié',
        10: 'Background'
    }

    total_area_pixels = np.count_nonzero(change_map != 12)
    total_area_m2 = total_area_pixels * (pixel_size ** 2)

    change_area = {}
    for class_id, class_name in change_classes.items():
        if class_id != 12:
            class_pixels = np.count_nonzero(change_map == class_id)
            class_area_m2 = class_pixels * (pixel_size ** 2)
            change_area[class_name] = class_area_m2

    change_percentage = {k: (v / total_area_m2) * 100 for k, v in change_area.items()}

    # --- Affichage texte ---
    st.markdown(f"🌐 **Surface totale analysée (hors background)** : `{total_area_m2 / 1e6:.2f}` km²")
    #st.markdown("### 📈 Répartition des changements")

    #for name, area in change_area.items():
    #    st.write(f"- {name}: `{area / 1e6:.2f}` km² (`{change_percentage[name]:.2f}%`)")

    # --- Graphique barres horizontales ---
    labels = list(change_percentage.keys())
    values = list(change_percentage.values())

    fig_bar, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color='skyblue')
    ax.set_xlabel('Pourcentage de changement (%)')
    ax.set_title('Distribution des types de changement')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', va='center')

    st.pyplot(fig_bar)

    # --- Graphique camembert ---
    labels_pie = [k for k, v in change_percentage.items() if v > 0.1]
    values_pie = [change_percentage[k] for k in labels_pie]

    if values_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        ax_pie.pie(values_pie, labels=labels_pie, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
        ax_pie.set_title('Répartition des changements (camembert)')
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

def compute_area_table(mask1, mask2, change_map, pixel_area_km2=0.0001):
    classes = np.unique(np.concatenate([mask1, mask2]))
    rows = []

    for cls in classes:
        t1_area = np.sum(mask1 == cls) * pixel_area_km2
        t2_area = np.sum(mask2 == cls) * pixel_area_km2
       # change_area = np.sum(change_map == cls) * pixel_area_km2

        rows.append({
            "Classe": int(cls),
            "T1 (km²)": float(t1_area),
            "T2 (km²)": float(t2_area)
       #     "Changement (km²)": float(change_area)
        })

    df = pd.DataFrame(rows)
    return df
