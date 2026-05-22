---
title: PixelNur
emoji: 🔒
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
---

<div align="center">
  
# 🔒 PixelNur
### Advanced CNN-Based Steganography System
*Hide secret messages inside images with military-grade encryption*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow)](https://huggingface.co/spaces/imrushikesh09/pixelnur)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-orange?logo=gradio)](https://gradio.app/)

[🚀 Try It Now](https://huggingface.co/spaces/imrushikesh09/pixelnur) • [🐛 Report Bug](https://github.com/rushikeshxdev/pixelNur/issues) • [📖 How It Works](#-how-it-works)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **CNN-Based Adaptive Embedding** | Intelligent texture analysis for imperceptible hiding |
| 🔐 **AES-256 Encryption** | Military-grade security with PBKDF2 key derivation |
| 🛡️ **Attack Resistant** | Survives JPEG compression, resizing & filtering |
| 📊 **High Quality Output** | PSNR >40 dB, SSIM >0.99 — visually identical |
| ⚡ **GPU Accelerated** | Fast processing with CUDA support |
| 🎯 **Blind Extraction** | No original image needed for decoding |

---

## 🚀 Quick Start

### ▶️ Embed a Message
1. Upload your cover image (PNG/JPEG)
2. Type your secret message
3. Set a strong password (16+ characters recommended)
4. Choose a robustness level
5. Download the stego image

### 🔍 Extract a Message
1. Upload the stego image
2. Enter the correct password
3. Click Extract — your hidden message is revealed

---

## 🎚️ Robustness Levels

| Level | Quality | Capacity | Attack Resistance | Best For |
|-------|---------|----------|-------------------|----------|
| **None** | ⭐⭐⭐⭐⭐ | 100% | None | Trusted local channels |
| **Low** | ⭐⭐⭐⭐ | 80% | Light JPEG | Social media platforms |
| **Medium** | ⭐⭐⭐ | 27% | JPEG QF 75+ | Messaging apps |
| **High** | ⭐⭐ | 16% | Maximum protection | Hostile environments |

---

## 🔬 How It Works

```
Cover Image → LWT Transform → CNN Mask → AES-256 Encrypt → LSB Embed → Stego Image
                                                                              ↓
                                                                       Extraction ← Password
```

1. **Lifting Wavelet Transform (LWT)** — Decomposes the image into frequency sub-bands
2. **CNN Texture Analysis** — A lightweight CNN generates a texture-aware embedding mask
3. **AES-256-CBC Encryption** — Message is encrypted with PBKDF2-HMAC-SHA256 key derivation
4. **Error Correction & Replication** — Optional redundancy for robustness against attacks
5. **LSB Matching Embedding** — Bits are embedded in the frequency domain using the CNN mask
6. **Blind Extraction** — Decoder reconstructs the message using only the password

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **PSNR** | 40–48 dB |
| **SSIM** | 0.96–0.99 |
| **Capacity** | 8–128 KB (varies by image & robustness) |
| **Processing Speed** | 2–15 seconds (GPU accelerated) |

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch — CNN-based texture analysis
- **Image Processing**: OpenCV, Lifting Wavelet Transform (LWT)
- **Cryptography**: AES-256-CBC, PBKDF2-HMAC-SHA256
- **Web Interface**: Gradio
- **Deployment**: Hugging Face Spaces

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/rushikeshxdev/pixelNur.git
cd pixelNur

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will launch at `http://localhost:7860` by default.

### Requirements

```
torch>=1.9.0
torchvision
opencv-python
gradio>=3.0
pycryptodome
numpy
Pillow
scipy
```

---

## 📁 Project Structure

```
pixelNur/
├── app.py              # Gradio web interface
├── steganography.py    # Core embedding & extraction logic
├── cnn_model.py        # CNN texture analysis model
├── crypto.py           # AES-256 encryption utilities
├── wavelet.py          # Lifting Wavelet Transform
├── requirements.txt
└── README.md
```

---

## 🎓 Research Team

**KIT's College of Engineering, Kolhapur**

| Name | Roll No. |
|------|----------|
| Aman Qureshi | 2223000503 |
| Rushikesh Randive | 2223000930 |
| Ankita Patil | 2223000302 |
| Madhura Patil | 2223000060 |

---

## 🔐 Security Notes

- Always use a **strong, unique password** (16+ characters)
- The system uses **AES-256-CBC** with a random IV per message
- Keys are derived using **PBKDF2-HMAC-SHA256** with 100,000 iterations
- For maximum security, use **None** robustness level and share stego images over trusted channels only

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Built with ❤️ for privacy and information security research
- Powered by [Hugging Face Spaces](https://huggingface.co/spaces)
- Inspired by academic research in adaptive steganography

---

<div align="center">

⭐ **Star this repo if you find it useful!**

*Built with ❤️ for privacy • Powered by Hugging Face Spaces*

</div>
