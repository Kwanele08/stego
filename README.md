# Deep Learning-Based Image Steganography with Gaussian Noise Robustness

This project investigates the use of deep learning for image steganography, with a particular focus on improving the robustness of hidden image recovery under Gaussian noise.

The system hides a grayscale secret image inside an RGB cover image using a convolutional encoder-decoder architecture. A classical 2-bit Least Significant Bit (LSB) method is also implemented as a baseline. The deep learning system is evaluated under different Gaussian noise levels, and a noise-augmented training strategy is introduced to improve recovery robustness.

---

## Project Overview

Image steganography aims to hide information inside another image while keeping the presence of the hidden information difficult to notice.

The project explores three main approaches:

1. Classical 2-bit LSB steganography
2. Baseline deep learning-based image steganography
3. Noise-robust deep learning-based image steganography

The main research question is whether Gaussian noise augmentation during training can improve secret image recovery when stego images are corrupted during transmission.

---

## Key Features

- Image-in-image steganography
- Classical 2-bit LSB baseline
- CNN-based encoder-decoder architecture
- Multi-scale convolutional feature extraction
- Gaussian noise augmentation
- Secret image reconstruction
- Multi-level Gaussian noise testing
- PSNR, SSIM, and MSE evaluation
- Fixed-sample reproducible experiments
- Evaluation across 100 image pairs
- CSV-based experimental result storage
- Statistical analysis using averages and standard deviations
- Publication-quality result graphs

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-image
- CIFAR-10 dataset

---

## System Architecture

The deep learning system follows an encoder-decoder architecture.

```text
Cover Image + Secret Image
            |
            v
         Encoder
            |
            v
       Stego Image
            |
            v
         Decoder
            |
            v
    Recovered Secret
```

The encoder receives the RGB cover image and grayscale secret image and learns to generate a stego image that visually resembles the cover.

The decoder attempts to reconstruct the original secret image from the stego image.

For the robust model, Gaussian noise is introduced between the encoder and decoder during training:

```text
Cover + Secret
      |
      v
   Encoder
      |
      v
 Stego Image
      |
      v
Gaussian Noise
      |
      v
   Decoder
      |
      v
Recovered Secret
```

This encourages the decoder to learn representations that remain recoverable under noisy conditions.

---

## Dataset

The project uses the CIFAR-10 dataset.

CIFAR-10 contains 60,000 RGB images across ten object categories.

For this project:

- Cover image size: `128 × 128 × 3`
- Secret image size: `32 × 32 × 1`

The original CIFAR-10 images are resized during preprocessing.

During training, cover and secret images are paired through the custom dataset pipeline.

During final evaluation, fixed image samples are used to improve reproducibility and ensure that both models are evaluated under identical conditions.

---

## Project Structure

```text
Steganography-research/
│
├── baselines/
│   └── lsb_steganography.py
│
├── models/
│   └── stego_model.py
│
├── utils/
│   ├── dataset_loader.py
│   ├── metrics.py
│   └── visualization.py
│
├── experiments/
│   ├── dataset_test.py
│   ├── lsb_test.py
│   ├── model_test.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── batch_evaluation.py
│   ├── noise_robustness_test.py
│   ├── multi_noise_experiment.py
│   ├── train_noise_robust_model.py
│   ├── robust_model_noise_test.py
│   ├── dataset_evaluation.py
│   ├── analyze_results.py
│   └── publication_graphs.py
│
├── stego_model.pth
├── robust_stego_model.pth
├── baseline_results.csv
├── robust_results.csv
├── summary_results.csv
├── percentage_improvement.csv
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Steganography-research
```

Create and activate a virtual environment if desired.

Install the required dependencies:

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib scikit-image tqdm
```

---

## Running the Project

### 1. Test the dataset pipeline

```bash
python experiments/dataset_test.py
```

Expected tensor dimensions:

```text
Cover:  [1, 3, 128, 128]
Secret: [1, 1, 32, 32]
```

---

### 2. Test the LSB baseline

```bash
python experiments/lsb_test.py
```

This performs:

```text
Secret Image
     |
     v
2-bit LSB Embedding
     |
     v
Stego Image
     |
     v
LSB Extraction
     |
     v
Recovered Secret
```

---

### 3. Test the neural network architecture

```bash
python experiments/model_test.py
```

Expected outputs:

```text
Stego shape:     [1, 3, 128, 128]
Recovered shape: [1, 1, 32, 32]
```

---

### 4. Train the baseline deep learning model

```bash
python experiments/train_model.py
```

The trained model is saved as:

```text
stego_model.pth
```

---

### 5. Evaluate the baseline model

```bash
python experiments/evaluate_model.py
```

Evaluation includes:

- PSNR
- SSIM
- Secret reconstruction MSE
- Cover and secret visualization

---

### 6. Train the noise-robust model

```bash
python experiments/train_noise_robust_model.py
```

The robust model is saved as:

```text
robust_stego_model.pth
```

---

### 7. Run multi-level Gaussian noise testing

```bash
python experiments/multi_noise_experiment.py
```

The following Gaussian noise standard deviations are used:

```text
0.01
0.03
0.05
0.10
0.15
```

---

### 8. Evaluate both models on fixed image samples

```bash
python experiments/dataset_evaluation.py
```

This performs controlled evaluation using the same fixed image pairs for both models.

The script generates:

```text
baseline_results.csv
robust_results.csv
```

Each model is evaluated across 100 image pairs and five Gaussian noise levels.

---

### 9. Analyze the results

```bash
python experiments/analyze_results.py
```

This generates:

```text
summary_results.csv
percentage_improvement.csv
average_psnr.png
average_ssim.png
average_secret_mse.png
percentage_improvement.png
```

---

### 10. Generate publication-quality graphs

```bash
python experiments/publication_graphs.py
```

Generated figures include:

```text
publication_psnr.png
publication_ssim.png
publication_secret_mse.png
```

The graphs include standard deviation error bars, and the secret reconstruction error is displayed using a logarithmic scale.

---

## Evaluation Metrics

### Peak Signal-to-Noise Ratio

PSNR measures the distortion between the original cover image and the generated stego image.

Higher PSNR indicates better stego image quality.

\[
PSNR = 10\log_{10}\left(\frac{MAX^2}{MSE}\right)
\]

---

### Structural Similarity Index

SSIM evaluates perceptual and structural similarity between the cover and stego images.

Values closer to `1.0` indicate greater structural similarity.

---

### Secret Mean Squared Error

Secret MSE measures the reconstruction error between the original secret image and the recovered secret image.

Lower values indicate better secret recovery.

---

## Baseline Deep Model Results

The baseline deep learning model achieved approximately:

```text
Average PSNR       : 30.58 dB
Average SSIM       : 0.9591
Average Secret MSE : 0.000678
```

These results demonstrated that the network successfully learned to hide and reconstruct secret images while maintaining strong visual similarity between the cover and stego images.

---

## Robustness Experiment

The main robustness experiment compares the original deep learning model with the Gaussian-noise-trained model.

Both models were evaluated using:

- The same fixed image pairs
- The same Gaussian noise levels
- The same evaluation metrics
- The same experimental pipeline

The robust model consistently achieved substantially lower secret reconstruction error under Gaussian noise.

Observed Secret MSE reductions were approximately:

| Gaussian Noise STD | Secret MSE Reduction |
|---:|---:|
| 0.01 | ~68% |
| 0.03 | ~86% |
| 0.05 | ~88% |
| 0.10 | ~84% |
| 0.15 | ~77% |

Overall, the robust model reduced secret reconstruction error by approximately **68–88%** across the tested Gaussian noise levels.

---

## Key Finding

The experiments demonstrate a trade-off between stego image fidelity and secret recovery robustness.

The baseline model generally maintains slightly higher PSNR and SSIM values. However, the robust model achieves significantly lower secret reconstruction error after Gaussian noise corruption.

The main finding of the project is therefore:

> Gaussian noise augmentation during training can substantially improve the robustness of deep learning-based image steganography, reducing secret reconstruction error while maintaining comparable structural image quality.

---

## Research Contribution

The main contribution of this project is the development and experimental evaluation of a noise-aware deep learning image steganography system.

The work demonstrates that robustness can be improved without changing the neural network architecture itself. Instead, the enhancement is achieved by modifying the training strategy through controlled Gaussian noise augmentation.

The research also provides a controlled comparison between:

- Classical LSB steganography
- Standard deep learning steganography
- Noise-augmented deep learning steganography

---

## Limitations

Current limitations include:

- Experiments are limited to the CIFAR-10 dataset.
- Only Gaussian noise attacks are evaluated in the robustness study.
- The current implementation focuses on relatively small images.
- Robustness against JPEG compression, resizing, cropping, and other transformations has not yet been fully investigated.
- The project does not currently include a dedicated deep-learning steganalysis detector.

---

## Future Work

Possible future improvements include:

- Testing additional image datasets
- Evaluating JPEG compression attacks
- Testing resizing, cropping, blur, and transmission artifacts
- Adding steganalysis-aware adversarial training
- Exploring attention-based architectures
- Investigating GAN-based steganography
- Supporting higher-resolution images
- Exploring adaptive noise augmentation
- Evaluating different secret payload sizes

---

## Research Status

The current implementation includes:

- Working classical LSB steganography
- Working deep learning steganography model
- Working noise-robust model
- Controlled multi-noise evaluation
- Fixed-sample reproducible experiments
- Statistical analysis
- Publication-quality figures

The technical implementation and primary experimental evaluation are complete.

---

## Author

**Kwanele Maziya**

B.Tech Computer Science and Engineering  
AI / Machine Learning and Computer Vision Research
