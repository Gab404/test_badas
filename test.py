import cv2
import math  # Nécessaire pour détecter les 'nan'
from badas import BADASModel

# 1. Initialisation et Prédiction
model = BADASModel(device="cuda")
video_path = "UC_IHM_HLB_Full_V3_Blurred.mp4"

print("Calcul des prédictions en cours...")
predictions = model.predict(video_path, real_time=True)

# 2. Nettoyage et Tri
valid_preds = []

# On parcourt toutes les prédictions
for i, prob in enumerate(predictions):
    # On convertit en float python standard pour être sûr
    prob_val = float(prob)
    
    # On garde seulement si ce n'est PAS un 'nan'
    if not math.isnan(prob_val):
        valid_preds.append((i, prob_val))

# Si après nettoyage la liste est vide, c'est que le modèle n'a rien sorti
if not valid_preds:
    print("❌ Erreur : Le modèle n'a renvoyé que des valeurs 'nan' pour toute la vidéo.")
else:
    # Trier du plus grand au plus petit
    sorted_preds = sorted(valid_preds, key=lambda x: x[1], reverse=True)
    
    # Prendre le top 10
    top_10 = sorted_preds[:10]

    # 3. Extraction des images
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"--- Enregistrement des frames (Top 10 valides) ---")

        for i, prob in top_10:
            timestamp_seconds = i * 0.125
            
            # Calcul du temps pour le nom de fichier
            minutes = int(timestamp_seconds // 60)
            seconds = int(timestamp_seconds % 60)
            
            # Calcul de la frame
            frame_number = int(timestamp_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            
            if ret:
                # Nom de fichier propre: pred_timestamp.png
                # Ex: 0.89_0-34.png
                filename = f"{prob:.2f}_{minutes}-{seconds:02d}.png"
                cv2.imwrite(filename, frame)
                print(f"✅ {filename} sauvegardé (Temps: {timestamp_seconds:.1f}s)")
            else:
                print(f"❌ Erreur lecture frame à {timestamp_seconds:.1f}s")

        cap.release()
        print("Terminé.")