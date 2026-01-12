from badas import BADASModel

# Initialize model
model = BADASModel(device="cuda")  # or "cpu" for CPU inference

# Predict on video
predictions = model.predict("C:/Users/gabri/Desktop/BADAS/BADAS-Open/test.mp4")

# Get collision risk for each frame window
for i, prob in enumerate(predictions):
    if prob > 0.8:
        print(f"⚠️ High collision risk at {i*0.125:.1f}s: {prob:.2%}")