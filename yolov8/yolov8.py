from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

if __name__ == '__main__':
    # Train the model on the cardboard box dataset
    train_results = model.train(
        data="/home/yu/perception/yolov8/dataset/data.yaml",  # Path to dataset configuration file
        epochs=50,  # Number of training epochs (adjustable)
        imgsz=640,  # Image size for training
        workers=4,
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()
