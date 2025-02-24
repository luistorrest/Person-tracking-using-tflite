import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Constants
CLASSES = ["person", "no person"]
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

def load_and_verify_image(image_path):
    """Load and verify the integrity of the image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    try:
        Image.open(image_path).verify()
    except Exception as e:
        raise IOError(f"Error verifying image file: {e}")

def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model."""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image

def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
    """Return the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold and int(classes[i]) == 0:  # Only include "person" (class_id = 0)
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def draw_detection_results(image_np, results):
    """Draw detection results on the image."""
    if results:
        for obj in results:
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * image_np.shape[1])
            xmax = int(xmax * image_np.shape[1])
            ymin = int(ymin * image_np.shape[0])
            ymax = int(ymax * image_np.shape[0])

            color = [int(c) for c in COLORS[0]]  # Use the color for "person"
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, 2)
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            label = "{}: {:.0f}%".format(CLASSES[0], obj['score'] * 100)
            cv2.putText(image_np, label, (xmin, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        text = "No person detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (image_np.shape[1] - text_size[0]) // 2
        text_y = (image_np.shape[0] + text_size[1]) // 2
        cv2.putText(image_np, text, (text_x, text_y),
                    font, font_scale, (0, 0, 255), font_thickness)

    return image_np

def run_object_detection(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and return the final image with results."""
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    preprocessed_image, original_image = preprocess_image(image_path, (input_height, input_width))
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
    original_image_np = original_image.numpy().astype(np.uint8)
    output_image = draw_detection_results(original_image_np, results)
    return output_image

def main():
    # Load the TFLite model
    model_path = r"efficientdet-tflite-lite0-int8.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Verify the image file path and integrity
    image_path = r"p2.jpg"  # Replace with the path to your image

    load_and_verify_image(image_path)

    # Run object detection
    output_image = run_object_detection(image_path, interpreter, threshold=0.5)

    # Display the output image
    Image.fromarray(output_image).show()

if __name__ == "__main__":
    main()