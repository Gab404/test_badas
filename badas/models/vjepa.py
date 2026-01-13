#!/usr/bin/env python3
"""
V-JEPA 2 Model Implementation using project's actual loading logic
"""
import sys
import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable

from core.base import BaseModel
from utils.video import (
    get_device, load_vjepa_model, preprocess_video_frames,
    get_processor_for_model, get_transform_for_model, apply_temperature_scaling
)
from utils.sliding_window import SlidingWindowPredictor


class VJEPAModel(BaseModel):
    """V-JEPA 2 model implementation using actual project logic"""
    
    def __init__(self, model_name: str, checkpoint_path: Optional[str] = None, 
                 device: Optional[str] = None, frame_count: int = 32, img_size: int = 224,
                 target_fps: Optional[float] = None, take_last_frames: bool = True,
                 use_sliding_window: bool = False, window_stride: int = 16,
                 save_preprocessed_tensors: bool = False, fill_value=None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device) if device else get_device()
        self.frame_count = frame_count
        self.img_size = img_size
        self.target_fps = target_fps
        self.take_last_frames = take_last_frames
        self.use_sliding_window = use_sliding_window
        self.window_stride = window_stride
        self.save_preprocessed_tensors = save_preprocessed_tensors
        self.fill_value = fill_value
        # Model components - loaded on demand
        self.model = None
        self.processor = None
        self.transform = None
        
        # Sliding window predictor - created on demand
        self.sliding_window_predictor = None
        
        # Storage for preprocessed tensors (for sample saving)
        self.preprocessed_tensors: Dict[str, torch.Tensor] = {}
        self.tensor_save_callback: Optional[Callable[[str, torch.Tensor], None]] = None
        
    def load(self) -> None:
        """Load V-JEPA model using project's loading logic"""
        try:
            # Load the model using shared utilities
            print(f"Loading V-JEPA model: {self.model_name}")
            if self.checkpoint_path:
                print(f"Using checkpoint: {self.checkpoint_path}")
            
            self.model = load_vjepa_model(
                model_name=self.model_name,
                checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            
            # Get processor and transform
            self.processor = get_processor_for_model(self.model_name)
            self.transform = get_transform_for_model(self.model_name, self.img_size)
            
            # Initialize sliding window predictor if needed
            if self.use_sliding_window:
                self.sliding_window_predictor = SlidingWindowPredictor(
                    window_size=self.frame_count,
                    stride=self.window_stride,
                    target_fps=self.target_fps,
                    fill_value=self.fill_value
                )
                print(f"🎬 Sliding window predictor initialized (window={self.frame_count}, stride={self.window_stride})")
            
            #print(f"✅ V-JEPA model loaded successfully")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load V-JEPA model: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading V-JEPA model: {e}")
    
    def set_tensor_save_callback(self, callback: Optional[Callable[[str, torch.Tensor], None]]) -> None:
        """Set callback function to handle saving of preprocessed tensors
        
        Args:
            callback: Function that takes (video_path, tensor) and handles saving
        """
        self.tensor_save_callback = callback
    
    def enable_tensor_saving(self, enable: bool = True) -> None:
        """Enable or disable tensor saving during prediction
        
        Args:
            enable: Whether to save preprocessed tensors
        """
        self.save_preprocessed_tensors = enable
        if not enable:
            self.preprocessed_tensors.clear()
    
    def get_saved_tensor(self, video_path: str) -> Optional[torch.Tensor]:
        """Get saved preprocessed tensor for a video path
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Saved tensor or None if not found
        """
        return self.preprocessed_tensors.get(video_path)
    
    def get_all_saved_tensors(self) -> Dict[str, torch.Tensor]:
        """Get all saved preprocessed tensors
        
        Returns:
            Dictionary mapping video paths to tensors
        """
        return self.preprocessed_tensors.copy()
    
    def clear_saved_tensors(self) -> None:
        """Clear all saved tensors from memory"""
        self.preprocessed_tensors.clear()
    
    def predict(self, video_path: str, real_time: bool = False) -> np.ndarray:
        """Predict frame-level probabilities for single video"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            if self.use_sliding_window and self.sliding_window_predictor is not None:
                # On passe le paramètre real_time ici
                return self._predict_sliding_window(video_path, real_time=real_time)
            else:
                return self._predict_regular(video_path)
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Video file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Prediction failed for {video_path}: {e}")

    # 2. Modifier _predict_sliding_window pour gérer l'affichage
    def _predict_sliding_window(self, video_path: str, real_time: bool = False) -> np.ndarray:
        """Sliding window prediction for frame-level results"""
        
        first_tensor_saved = False
        
        # Variable pour stocker la frame brute à afficher (partagée entre preprocess et predict)
        current_display_frame = None 

        def preprocess_fn(frames_array):
            """Preprocess frames for model input"""
            nonlocal first_tensor_saved, current_display_frame
            
            # --- LOGIQUE REAL TIME ---
            if real_time:
                # On prend la dernière image de la fenêtre (l'image "actuelle")
                # frames_array est en RGB (de video.py), OpenCV veut du BGR
                raw_img = frames_array[-1].copy()
                current_display_frame = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            # -------------------------

            # ... (Le reste du code preprocess existant ne change pas) ...
            if self.processor:
                try:
                    if hasattr(self.processor, '__call__'):
                        inputs = self.processor(videos=frames_array, return_tensors="pt")
                        # ... (logique d'extraction pixel_values)
                        if 'pixel_values_videos' in inputs:
                            video_tensor = inputs['pixel_values_videos'].squeeze(0)
                        elif 'pixel_values' in inputs:
                            video_tensor = inputs['pixel_values'].squeeze(0)
                        else:
                            video_tensor = list(inputs.values())[0].squeeze(0)
                    else:
                        raise ValueError("Invalid processor")
                except Exception as e:
                    # print(f"Warning: ...")
                    video_tensor = self._manual_transform_frames(frames_array)
            else:
                video_tensor = self._manual_transform_frames(frames_array)
            
            # ... (Logique de sauvegarde tensor inchangée) ...
            if self.save_preprocessed_tensors and not first_tensor_saved:
                 # ...
                 pass # (Garde ton code existant ici)

            return video_tensor
        
        def model_predict_fn(processed_frames):
            """Model prediction function for sliding window"""
            nonlocal current_display_frame

            # ... (Préparation tensor inchangée) ...
            if processed_frames.dim() == 4:
                processed_frames = processed_frames.unsqueeze(0)
            processed_frames = processed_frames.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(processed_frames)
                outputs_scaled = apply_temperature_scaling(outputs, temperature=2.0)
                probs = torch.softmax(outputs_scaled, dim=1)[:, 1].cpu().numpy()
                
                prediction_score = probs[0] # Le score final
                if real_time and current_display_frame is not None:
                    # 1. Redimensionner l'image (Agrandissement x3 ou x4)
                    scale_factor = 3
                    h, w = current_display_frame.shape[:2]
                    new_dim = (w * scale_factor, h * scale_factor)
                    big_frame = cv2.resize(current_display_frame, new_dim, interpolation=cv2.INTER_LINEAR)
                    
                    # --- DESSIN DE LA JAUGE VERTICALE ---
                    
                    # Configuration de la jauge (en pixels sur la grande image)
                    gauge_w = 30                # Largeur de la barre
                    gauge_h = int(new_dim[1] * 0.6)  # Hauteur (60% de l'écran)
                    margin_right = 50           # Marge à droite
                    margin_bottom = 50          # Marge en bas
                    
                    # Coordonnées
                    x_start = new_dim[0] - margin_right - gauge_w
                    y_bottom = new_dim[1] - margin_bottom
                    y_top = y_bottom - gauge_h
                    
                    # Dessiner le fond de la jauge (Gris foncé)
                    cv2.rectangle(big_frame, (x_start, y_top), (x_start + gauge_w, y_bottom), (40, 40, 40), -1)
                    
                    # Calcul de la couleur dynamique (Vert -> Jaune -> Rouge)
                    # Format BGR pour OpenCV
                    score = prediction_score
                    if score < 0.5:
                        # De Vert (0, 255, 0) à Jaune (0, 255, 255)
                        # Le rouge augmente, le vert reste à fond
                        ratio = score / 0.5
                        r = int(255 * ratio)
                        g = 255
                        b = 0
                    else:
                        # De Jaune (0, 255, 255) à Rouge (0, 0, 255)
                        # Le rouge reste à fond, le vert diminue
                        ratio = (score - 0.5) / 0.5
                        r = 255
                        g = int(255 * (1 - ratio))
                        b = 0
                    
                    gauge_color = (b, g, r) # BGR
                    
                    # Calcul de la hauteur de remplissage
                    fill_height = int(gauge_h * score)
                    y_fill_start = y_bottom - fill_height
                    
                    # Dessiner la partie remplie (La jauge active)
                    cv2.rectangle(big_frame, (x_start, y_fill_start), (x_start + gauge_w, y_bottom), gauge_color, -1)
                    
                    # Dessiner la bordure (Blanc)
                    cv2.rectangle(big_frame, (x_start, y_top), (x_start + gauge_w, y_bottom), (200, 200, 200), 2)
                    
                    # Ajouter des repères (Lignes horizontales à 50% et 80%)
                    # Seuil d'avertissement (50%)
                    y_50 = y_bottom - int(gauge_h * 0.5)
                    cv2.line(big_frame, (x_start - 5, y_50), (x_start + gauge_w + 5, y_50), (100, 100, 100), 1)
                    
                    # Seuil critique (80%)
                    y_80 = y_bottom - int(gauge_h * 0.8)
                    cv2.line(big_frame, (x_start - 5, y_80), (x_start + gauge_w + 5, y_80), (100, 100, 100), 1)
                    
                    # Afficher le pourcentage textuel à côté de la jauge
                    text_pct = f"{int(score * 100)}%"
                    cv2.putText(big_frame, text_pct, (x_start - 60, y_fill_start + 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gauge_color, 2)
                    
                    # Afficher l'image finale
                    cv2.imshow('BADAS Real-Time Inference', big_frame)
                    
                    # Gestion de la sortie (touche 'q')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Interruption utilisateur")
                # ----------------------------------------
                return prediction_score
        
        # Lancement du predictor
        try:
            results = self.sliding_window_predictor.predict_sliding_windows(
                video_path=video_path,
                model_predict_fn=model_predict_fn,
                preprocess_fn=preprocess_fn,
                return_per_frame=True
            )
        finally:
            # Important : Fermer la fenêtre OpenCV à la fin du traitement
            if real_time:
                cv2.destroyAllWindows()
        
        return results['per_frame']
    
    def _manual_transform_frames(self, frames_array: np.ndarray) -> torch.Tensor:
        """Manual transformation for frames array"""
        if self.transform:
            transformed_frames = [self.transform(image=f)["image"] for f in frames_array]
            return torch.stack(transformed_frames)
        else:
            # Simple normalization
            frames_tensor = torch.from_numpy(frames_array.transpose(0, 3, 1, 2)).float() / 255.0
            return frames_tensor
    
    def predict_batch(self, video_paths: List[str]) -> List[np.ndarray]:
        """Predict frame-level probabilities for multiple videos"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        results = []
        for video_path in video_paths:
            try:
                pred = self.predict(video_path)
                results.append(pred)
            except Exception as e:
                # Fail fast - don't silently continue
                raise RuntimeError(f"Batch prediction failed at {video_path}: {e}")
                
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        info = {
            "name": "V-JEPA 2",
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "frame_count": self.frame_count,
            "img_size": self.img_size,
            "target_fps": self.target_fps,
            "take_last_frames": self.take_last_frames,
            "use_sliding_window": self.use_sliding_window,
            "window_stride": self.window_stride,
            "has_model": self.model is not None,
            "has_processor": self.processor is not None,
            "has_sliding_window_predictor": self.sliding_window_predictor is not None,
            "version": "2.1"
        }
        
        if self.sliding_window_predictor:
            info["sliding_window_config"] = self.sliding_window_predictor.get_config()
        
        return info