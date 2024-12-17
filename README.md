# Image Classification Project

This project implements an image classification pipeline using PyTorch. It uses a custom dataset of images, performs transformations, and trains a ResNet-50 model for classification. The project includes functionality for renaming image files, splitting data into training and validation sets, and predicting on unlabeled test data.

## Features
- Custom dataset processing and transformations.
- Training and validation of a ResNet-50 model.
- Early stopping and learning rate scheduling.
- Batch predictions for unlabeled images.
- Output predictions saved to a CSV file.

---

## Setup and Requirements

### Prerequisites
1. Python 3.8 or higher.
2. Required libraries:
   - `torch`
   - `torchvision`
   - `numpy`
   - `Pillow`
   - `google.colab` (for Colab environment)
3. GPU support is recommended for faster training.

### Installation
1. Install dependencies using pip:
   ```bash
   pip install torch torchvision pillow
   ```

2. Clone this repository or copy the code into your working directory.

3. If running on Google Colab:
   - Mount Google Drive to access the dataset:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Update the dataset paths (`path_to_file` and `unlabeled_data_dir`) to match your Google Drive folder structure.

---

## Dataset Structure

### Training Dataset
Ensure the training data is organized as follows:
```
/path/to/train
    ├── 00
    │   ├── image1.jpg
    │   ├── image2.jpg
    ├── 01
    │   ├── image1.jpg
    │   ├── image2.jpg
    ...
```

### Test Dataset
Unlabeled test data should be organized as:
```
/path/to/test
    ├── image1.jpg
    ├── image2.jpg
    ...
```

---

## Running the Code

### 1. Rename Images
Run the renaming script to standardize file names:
```python
for folder_number in range(100):
    folder_path = os.path.join(path_to_file, zero_pad(folder_number))
    ...
```

### 2. Train the Model
1. Ensure your dataset paths are correct.
2. Run the training script:
   ```python
   for epoch in range(num_epochs):
       model.train()
       ...
   ```

### 3. Predict on Unlabeled Data
1. Set the test dataset path in `unlabeled_data_dir`.
2. Run the prediction script:
   ```python
   for inputs, filename in data_loader:
       inputs = inputs.to(device)
       outputs = model(inputs)
       ...
   ```
3. Download the predictions CSV:
   ```python
   files.download('predictions.csv')
   ```

---

## Outputs
- Trained model saved as `best_model.pth`.
- Predictions saved in `predictions.csv`.

---

## Notes
- Ensure GPU is available for faster training. Check the device with:
  ```python
  print(torch.cuda.get_device_name(0))
  ```
- Adjust hyperparameters such as batch size, learning rate, and number of epochs as needed.

---

## Link to Original Notebook
You can view and run the original Google Colab notebook [here](https://colab.research.google.com/drive/1_DMevgk4hU9yjA5fx_QyuAu4EYVnAKVr?usp=sharing).
