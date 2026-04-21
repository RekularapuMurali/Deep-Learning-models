import numpy as np
import os
import nibabel as nib
from PIL import Image
from dae_model import build_dae

IMG_SIZE = 128
DATA_DIR = r'C:\Users\rekul\Desktop\Github\Deep-Learning-models\denoising_autoencoder\dataset\MICCAI_BraTS2020_TrainingData'

def add_noise(images, noise_factor=0.2):
    noisy = images + noise_factor * np.random.randn(*images.shape)
    return np.clip(noisy, 0.0, 1.0)

def load_nii_slices(data_dir, max_patients=40, slices_per_patient=8):
    slices = []
    folders = sorted(os.listdir(data_dir))[:max_patients]

    for folder in folders:
        nii_file = os.path.join(data_dir, folder, folder + '_t1.nii')
        if not os.path.exists(nii_file):
            continue
        print(f"Loading {folder}...")
        vol = nib.load(nii_file).get_fdata().astype(np.float32)
        if vol.max() == 0:
            continue

        vol = vol / vol.max()
        mid = vol.shape[2] // 2

        for i in range(mid - slices_per_patient, mid + slices_per_patient, 2):
            s = vol[:, :, i]
            # Only keep slices with actual brain content
            if s.mean() < 0.05:
                continue
            s_img = Image.fromarray((s * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(s_img, dtype=np.float32) / 255.0
            slices.append(arr)

    return np.array(slices)[..., np.newaxis]

print("Loading MRI slices...")
clean = load_nii_slices(DATA_DIR)
print(f"Loaded {len(clean)} slices.")
print(f"Data mean: {clean.mean():.4f}, max: {clean.max():.4f}")

noisy = add_noise(clean, noise_factor=0.2)

model = build_dae()

# Train with early stopping
import tensorflow as tf

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]
model.fit(noisy, clean, epochs=50, batch_size=16,
          validation_split=0.1, callbacks=callbacks)

os.makedirs('model', exist_ok=True)
model.save('model/dae_model.keras')
print("Done!")