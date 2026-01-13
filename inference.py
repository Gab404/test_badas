import cv2
import math
import argparse
from badas import BADASModel

def main():
    # 1. Configuration des arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Inférence BADAS avec options flexibles")
    parser.add_argument('--video-path', type=str, required=True, 
                        help="Chemin vers le fichier vidéo (ex: test.mp4)")
    parser.add_argument('--real-time', action='store_true', 
                        help="Active l'affichage de la vidéo et de la jauge pendant l'inférence")
    
    args = parser.parse_args()

    # 2. Initialisation du modèle
    # On utilise "cuda" si disponible pour la performance
    model = BADASModel(device="cuda")
    video_path = args.video_path

    print(f"--- Démarrage de l'analyse ---")
    print(f"Vidéo : {video_path}")
    print(f"Mode temps réel : {'ACTIVER' if args.real_time else 'DÉSACTIVER'}")

    # 3. Prédiction
    # On passe l'argument real_time directement au modèle
    try:
        predictions = model.predict(video_path, real_time=args.real_time)
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction : {e}")
        return

    # 4. Nettoyage et Tri
    valid_preds = []
    for i, prob in enumerate(predictions):
        prob_val = float(prob)
        if not math.isnan(prob_val):
            valid_preds.append((i, prob_val))

    if not valid_preds:
        print("❌ Erreur : Le modèle n'a renvoyé que des valeurs 'nan'.")
    else:
        # Trier et prendre le top 10
        sorted_preds = sorted(valid_preds, key=lambda x: x[1], reverse=True)
        top_10 = sorted_preds[:10]

        # 5. Extraction des images
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Erreur : Impossible d'ouvrir la vidéo {video_path}")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"--- Enregistrement du Top 10 des risques ---")

            for i, prob in top_10:
                # Calcul basé sur l'intervalle temporel du sliding window (0.125s)
                timestamp_seconds = i * 0.125
                minutes = int(timestamp_seconds // 60)
                seconds = int(timestamp_seconds % 60)
                
                frame_number = int(timestamp_seconds * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                
                if ret:
                    filename = f"{prob:.2f}_{minutes}-{seconds:02d}.png"
                    cv2.imwrite(filename, frame)
                    print(f"✅ {filename} sauvegardé ({timestamp_seconds:.1f}s)")
                else:
                    print(f"❌ Erreur lecture frame à {timestamp_seconds:.1f}s")

            cap.release()
            print("--- Analyse terminée ---")

if __name__ == "__main__":
    main()