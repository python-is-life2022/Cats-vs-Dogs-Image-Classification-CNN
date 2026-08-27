# Cats vs Dogs Image Classification — Model Testing Pipeline

This module tests a trained **Cats vs Dogs CNN model** on individual test images using TensorFlow and Keras.

---

## 📌 Overview

The testing workflow performs the following steps:

1. **Load the trained model:** Loads the pretrained `.keras` CNN model.
2. **Select a test image:** Receives the image number as input.
3. **Preprocess the image:** Loads the image, resizes it to `64 × 64`, normalizes pixel values to the range `[0, 1]`, and adds a batch dimension.
4. **Make a prediction:** Uses the trained model to predict whether the image is a cat or a dog.
5. **Display the result:** Shows the selected image along with the predicted label and confidence score.

---

## 📁 Test Image Structure

After extracting the test image archive, the project structure should look like this:
```text
project-folder/
├── cat_vs_dog_images_classification_CNN_model (2).keras
└── test/
└── test_photos/
├── 1.jpg
├── 2.jpg
└── ...
```
If your test images are compressed in a ZIP file, extract them first:
```
!unzip -q test_photos.zip -d test
```
`If you are using Google Colab and the ZIP file is located in /content, use:
!unzip -q /content/test_photos.zip -d /content/test
`

## 🛠️ Code Reference

```
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

# 1. Load the trained model
model = load_model("cat_vs_dog_images_classification_CNN_model (2).keras")

# 2. Select a test image
image_number = int(input("Enter the image number: "))

# 3. Load and display the original image
image_path = f"test/test_photos/{image_number}.jpg"
image = plt.imread(image_path)

# 4. Preprocess the image
processed_image = tf.image.resize(image, (64, 64))
processed_image = tf.cast(processed_image, tf.float32) / 255.0
processed_image = tf.expand_dims(processed_image, axis=0)

# 5. Make prediction
prediction = model.predict(processed_image, verbose=0)
confidence = float(prediction[0][0])

# 6. Convert probability to class label
label = "Dog" if confidence > 0.5 else "Cat"

# 7. Show result
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.title(f"Prediction: {label} | Confidence: {confidence:.2%}")
plt.show()

```
## 🧠 Prediction Logic
The model produces a value between 0 and 1.

* A prediction greater than 0.5 is classified as Dog.
* A prediction less than or equal to 0.5 is classified as Cat.
  
```
label = "Dog" if confidence > 0.5 else "Cat"
```
