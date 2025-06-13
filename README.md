# Light Bulb Anomaly Detection with Autoencoders

This project applies an unsupervised deep learning approach using Autoencoders to detect anomalies in light bulb images. The objective is to train a model that reconstructs normal (non-defective) images and identifies anomalies based on reconstruction error.

## Project Overview

Autoencoders are neural networks designed to learn efficient representations of input data. In this project, a convolutional autoencoder is trained on normal light bulb images. During inference, images with high reconstruction error are flagged as potentially defective.

### Dataset

- **Source**: Light bulb images were obtained from [images.cv](https://images.cv).
- **Structure**:
  - `train/`: Contains only non-defective light bulb images used for training.
  - `val/`: Contains a mix of non-defective and defective images used for model validation.
  - `test/`: Contains unseen images used to evaluate final model performance.

### Approach

1. **Data Preprocessing**: All images are resized, normalized, and prepared for model input.
2. **Model Architecture**: A convolutional autoencoder is implemented to reconstruct input images.
3. **Training**: The model is trained using only the `train/` set of non-defective images.
4. **Validation**: The `val/` set helps fine-tune the reconstruction threshold for anomaly detection.
5. **Testing**: The trained model is evaluated on the `test/` set to measure real-world anomaly detection performance.

### Results

- The model effectively distinguishes defective bulbs based on reconstruction error.
- Final performance is evaluated on the test set, and results show clear separation between normal and defective images.
- Visualization of reconstruction loss highlights regions of defect.

## Dependencies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## License

- This project is intended for academic and educational use.
