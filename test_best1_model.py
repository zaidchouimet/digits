import os
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO

def test_model():
    model_path = "best1.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    img_dir = "datasets/dataset/images"
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print(f"No images found in {img_dir}")
        return

    # Deterministic sampling for reproducibility
    random.seed(42)
    sample_images = random.sample(image_files, min(10, len(image_files)))
    
    print(f"Predicting on {len(sample_images)} sample images...")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(img_dir, img_name)
        results = model(img_path, verbose=False)
        
        # Plot predictions
        res_plotted = results[0].plot()
        # Convert BGR to RGB for matplotlib
        res_plotted = res_plotted[:, :, ::-1]
        
        axes[i].imshow(res_plotted)
        axes[i].axis('off')
        axes[i].set_title(img_name)

    plt.tight_layout()
    output_path = "best1_predictions.png"
    plt.savefig(output_path)
    print(f"Saved inference results to {output_path}")

if __name__ == "__main__":
    test_model()
