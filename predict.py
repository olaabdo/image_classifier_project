import argparse
import tensorflow as tf
import numpy as np
import json
from PIL import Image

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    probs, classes = tf.math.top_k(predictions, k=top_k)

    return probs.numpy()[0], classes.numpy()[0]

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', type=str, help='Path to image file.')
    parser.add_argument('model_path', type=str, help='Path to saved Keras model.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K classes.')
    parser.add_argument('--category_names', type=str, default=None, help='JSON file mapping labels to names.')

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, compile=False)
    probs, classes = predict(args.image_path, model, args.top_k)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        labels = [class_names.get(str(i), str(i)) for i in classes]
    else:
        labels = [str(i) for i in classes]

    for label, prob in zip(labels, probs):
        print(f"{label}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()
