# Project Planning Outline
## Layout
1. Training and Data Collection
    - Collect images of target animals
    - Annotate images with bounding boxes and labels
    - Alternatively, use pre-existing datasets if available
    - Use LabelStudio with a model backend for automated annotation
2. Model Selection and Training
    - Annotation backend: LabelStudio with a model backend for automated annotation
    - Model architecture: YOLOv5 for object detection
    - Training process:
        - Annotate a few images manually to create a small training set
        - Train the YOLOv5 model on the annotated dataset
        - Use the trained model to annotate more images, creating a larger training set
        - Pass the lowest confidence annotations back to the human annotator for fixing and then retrain the model with the updated dataset
3. Model Selection for Deployment
    - Model architecture: YOLOv5 for object detection
    - Consider using a smaller version of YOLOv5 (e.g., YOLOv5s) for deployment on edge devices
    - Optimize the model for inference on edge devices (e.g., using TensorRT or ONNX)
    - Evaluate the model's performance on a validation set and adjust as necessary
4. Draw aimpoints on the images based on the model's predictions of where the vitals of the animals are located.
    - Draw the travel line of the bullet from the aimpoint to the vitals' center
    - Current bullet data will be static
    - Distance will be a value that can be changed by the demo user
5. Deployment
    - Deploy the trained model on edge devices (e.g., NVIDIA Jetson, Raspberry Pi)
    - Set up a real-time inference pipeline to process video feed from cameras
    - Demo will use recorded video feed for testing and demonstration purposes
    - Monitor the model's performance and make adjustments as necessary
