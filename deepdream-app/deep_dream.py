import numpy as np
import tensorflow as tf
from PIL import Image
import os

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

layer_names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in layer_names]
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((500, 500))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def deprocess_image(img_array):
    img = img_array.copy()
    img = img / 2.0 + 0.5
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def calc_loss(img_tensor):
    img_batch = tf.expand_dims(img_tensor, axis=0)
    layer_activations = dream_model(img_batch)
    losses = [tf.reduce_mean(act) for act in layer_activations]
    return tf.reduce_sum(losses)

@tf.function
def deepdream_step(img, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img)
    gradients = tape.gradient(loss, img)
    gradients /= (tf.math.reduce_std(gradients) + 1e-8)
    img = img + gradients * step_size
    img = tf.clip_by_value(img, -1, 1)
    return img

def run_deep_dream(input_path, output_path, steps=20, step_size=0.01, num_octaves=1, octave_scale=1.3):
    print("Starting Deep Dream...", flush=True)
    img = preprocess_image(input_path)
    base_shape = img.shape[:2]

    for octave in range(num_octaves):
        print(f"Octave {octave+1}/{num_octaves}", flush=True)
        new_h = int(base_shape[0] * (octave_scale ** octave))
        new_w = int(base_shape[1] * (octave_scale ** octave))
        img_pil = Image.fromarray(np.clip((img / 2.0 + 0.5) * 255, 0, 255).astype(np.uint8))
        img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
        img = np.array(img_pil, dtype=np.float32) / 127.5 - 1.0

        img_tensor = tf.Variable(img)
        for step in range(steps):
            img_tensor = tf.Variable(deepdream_step(img_tensor, step_size))
            if step % 5 == 0:
                print(f"  Step {step}/{steps}", flush=True)

        img = img_tensor.numpy()

    print("Saving output...", flush=True)
    result = deprocess_image(img)
    result = result.resize((500, 500), Image.LANCZOS)
    result.save(output_path)
    print("Done!", flush=True)