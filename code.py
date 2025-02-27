import tensorflow as tf  # Use TensorFlow instead of tflite_runtime
import cv2
import numpy as np
import argparse

def run(model: str, camera_id: int, width: int, height: int, num_threads: int, enable_edgetpu: bool) -> None:
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Use tf.lite.Interpreter instead of tflite.Interpreter
    interpreter = tf.lite.Interpreter(model_path=model, num_threads=num_threads)
    if enable_edgetpu:
        # Edge TPU delegation (requires libedgetpu.so)
        interpreter = tf.lite.Interpreter(
            model_path=model,
            experimental_delegates=[tf.lite.load_delegate('libedgetpu.so')],
            num_threads=num_threads
        )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check the input tensor's data type
    input_dtype = input_details[0]['dtype']
    print(f"Input tensor data type: {input_dtype}")  # Should be np.uint8

    # Class ID for "person" (usually 0 for COCO models)
    PERSON_CLASS_ID = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        rgb_image_resized = cv2.resize(rgb_image, (width, height))

        # Ensure the input data is of type UINT8
        if input_dtype == np.uint8:
            input_data = np.expand_dims(rgb_image_resized, axis=0)  # No need to normalize for UINT8
        else:
            input_data = np.expand_dims(rgb_image_resized, axis=0).astype(np.float32) / 255.0  # Normalize for FLOAT32

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        min_score = 0.3
        for i in range(len(scores)):
            # Only process detections for the "person" class
            if scores[i] > min_score and int(classes[i]) == PERSON_CLASS_ID:
                box = boxes[i]
                ymin, xmin, ymax, xmax = box
                height, width, _ = image.shape
                xmin = int(xmin * width)
                xmax = int(xmax * width)
                ymin = int(ymin * height)
                ymax = int(ymax * height)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f'Person, Score {scores[i]:.2f}'
                cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        cv2.imshow('object_detector', image)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Run TensorFlow Lite object detection on a webcam feed.')
    parser.add_argument('--model', type=str, default="efficientdet-tflite-lite0-int8.tflite", required=True, help='Path to the TensorFlow Lite model file.')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of the camera to use (default: 0).')
    parser.add_argument('--width', type=int, default=640, help='Width of the camera feed (default: 640).')
    parser.add_argument('--height', type=int, default=480, help='Height of the camera feed (default: 480).')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use for inference (default: 4).')
    parser.add_argument('--enable_edgetpu', action='store_true', help='Enable Edge TPU acceleration.')

    args = parser.parse_args()

    run(
        model=args.model,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        num_threads=args.num_threads,
        enable_edgetpu=args.enable_edgetpu
    )

if __name__ == '__main__':
    main()