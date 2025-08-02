from functions import (
    preprocess_image,
    predict_mask_for_image,
    detect_change,
    show_image_and_mask,
    extract_year_from_filename ,
    analyze_change_map,
    compute_area_table,
    change_names,
    change_palette
)
from matplotlib.colors import ListedColormap
import streamlit as st
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- Page settings ---
st.set_page_config(page_title="Détection de Changement", layout="wide")
st.title("🌍 Application de Détection de Changement")

# --- Upload des images ---
st.sidebar.header("🛰️ Sélection des images satellite (T1 et T2)")

img1 = st.sidebar.file_uploader("📂 Image satellite T1", type=["tif", "tiff"])
img2 = st.sidebar.file_uploader("📂 Image satellite T2", type=["tif", "tiff"])

if img1 and img2:
    st.success("✅ Images chargées avec succès. Prêt à lancer la détection !")

    if st.button("🔍 Lancer la détection de changement"):
        with st.spinner("Traitement en cours..."):
            # Pipeline d'analyse
            mask1 = predict_mask_for_image(img1)
            mask2 = predict_mask_for_image(img2)
            change_map = detect_change(mask1, mask2)
            table = compute_area_table(mask1, mask2, change_map)


        # --- Affichage ---
        st.subheader("🗺️ Carte d'occupation du sol - T1")
        fig1 = show_image_and_mask(img1, mask1, "Région de Ouaoula - " + extract_year_from_filename(img1.name))
        st.pyplot(fig1)

        st.subheader("🗺️ Carte d'occupation du sol - T2")
        fig2 = show_image_and_mask(img2, mask2, "Région de Ouaoula - " + extract_year_from_filename(img2.name))
        st.pyplot(fig2)


        st.subheader("🗺️ Carte de changement")
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = ListedColormap(change_palette)
        cax = ax.imshow(change_map, cmap=cmap, vmin=0, vmax=len(change_palette) - 1)
        cbar = fig.colorbar(cax, ax=ax, ticks=np.arange(len(change_names)))
        cbar.set_ticklabels(change_names)
        ax.axis('off')
        st.pyplot(fig)

        st.subheader("📊 Statistiques de changement")
        analyze_change_map(change_map)

        st.subheader("📋 Superficies comparées")
        st.dataframe(table)

else:
    st.warning("Veuillez charger les deux images satellites T1 et T2 pour démarrer.")
