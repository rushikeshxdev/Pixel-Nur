"""
PixelNur - Gradio Interface for Hugging Face Spaces

This module provides a web interface for the PixelNur steganography system
using Gradio. It exposes embedding and extraction functionality through
an intuitive tabbed interface.

Requirements: 2.1, 2.2, 3.5, 3.6
"""

import gradio as gr
import numpy as np
import cv2
import torch
import logging
import sys
import os
import tempfile
from typing import Tuple, Optional

# Import Phase 2 modules
from src.pixelnur import PixelNur, PixelNurError, InsufficientCapacityError
from src.extraction_engine import ExtractionEngine, ExtractionError
from src.lwt_transform import LWTTransform
from src.encryption_service import EncryptionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('pixelnur-gradio')

# Constants
ROBUSTNESS_LEVELS = {
    "🌟 None (0) - Best quality, no attack resistance": "none",
    "🛡️ Low (1) - Light ECC, minor attack resistance": "low",
    "🔒 Medium (2) - Moderate ECC + replication": "medium",
    "🔐 High (3) - Maximum protection, lower capacity": "high"
}

MIN_PASSWORD_LENGTH = 16
MIN_IMAGE_SIZE = 256
MAX_IMAGE_SIZE = 2048

# Custom CSS for beautiful UI with better contrast
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* Tab styling */
.tab-nav button {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 1rem 2rem !important;
    border-radius: 0.5rem !important;
    transition: all 0.3s ease !important;
}

.tab-nav button[aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Button styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 0.5rem !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.primary-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Input field styling */
.input-field {
    border-radius: 0.5rem !important;
    border: 2px solid #e0e7ff !important;
    transition: all 0.3s ease !important;
}

.input-field:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Card styling */
.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1rem;
    border-radius: 0.75rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

/* Image upload area - IMPROVED CONTRAST */
.image-upload {
    border: 3px dashed #667eea !important;
    border-radius: 1rem !important;
    background: #ffffff !important;
    transition: all 0.3s ease !important;
}

.image-upload:hover {
    border-color: #764ba2 !important;
    background: #f8f9fa !important;
}

/* Make sure upload text is visible */
.image-upload label,
.image-upload span,
.image-upload div {
    color: #2d3436 !important;
}
"""

# Initialize PixelNur system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Initializing PixelNur on device: {device}")

try:
    pixelnur = PixelNur(device=device)
    extraction_engine = ExtractionEngine()
    lwt_transform = LWTTransform()
    encryption_service = EncryptionService()
    logger.info("PixelNur system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PixelNur system: {str(e)}", exc_info=True)
    raise


def _validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """Validate image format and dimensions."""
    if image is None or image.size == 0:
        return False, "No image provided. Please upload a valid image."
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False, "Invalid image format. Please upload a color image (RGB)."
    
    h, w = image.shape[:2]
    if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
        return False, f"Image too small: {w}×{h} pixels. Minimum: {MIN_IMAGE_SIZE}×{MIN_IMAGE_SIZE} pixels."
    
    if h > MAX_IMAGE_SIZE or w > MAX_IMAGE_SIZE:
        return False, f"Image too large: {w}×{h} pixels. Maximum: {MAX_IMAGE_SIZE}×{MAX_IMAGE_SIZE} pixels."
    
    return True, ""


def _validate_password(password: str) -> Tuple[bool, str]:
    """Validate password meets minimum requirements."""
    if not password:
        return False, "Password is required."
    
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password too short: {len(password)} characters. Minimum: {MIN_PASSWORD_LENGTH} characters."
    
    return True, ""


def _parse_robustness_level(level_str: str) -> str:
    """Convert UI string to robustness level value."""
    return ROBUSTNESS_LEVELS.get(level_str, "none")


def embed_interface(
    cover_image: np.ndarray,
    message: str,
    password: str,
    robustness_level: str
) -> Tuple[Optional[str], str, str, str]:
    """Gradio interface function for embedding."""
    try:
        valid, error = _validate_image(cover_image)
        if not valid:
            return None, f"❌ {error}", "", ""
        
        if not message or not message.strip():
            return None, "❌ Message cannot be empty.", "", ""
        
        valid, error = _validate_password(password)
        if not valid:
            return None, f"❌ {error}", "", ""
        
        cover_bgr = cv2.cvtColor(cover_image, cv2.COLOR_RGB2BGR)
        robustness_value = _parse_robustness_level(robustness_level)
        
        logger.info(
            f"Embed request: image_shape={cover_image.shape}, "
            f"message_length={len(message)}, robustness_level={robustness_value}"
        )
        
        stego_bgr, metrics = pixelnur.embed_message(
            cover_image=cover_bgr,
            message=message,
            encryption_key=password,
            robustness_level=robustness_value,
            check_capacity=True
        )
        
        output_path = os.path.join(tempfile.gettempdir(), "pixelnur_stego.png")
        cv2.imwrite(output_path, stego_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        psnr_text = f"✅ PSNR: {metrics['psnr']:.2f} dB"
        ssim_text = f"✅ SSIM: {metrics['ssim']:.4f}"
        capacity_text = (
            f"✅ Capacity: {metrics['capacity_used_bytes']} / "
            f"{metrics['capacity_available_bytes']} bytes "
            f"({metrics['capacity_utilization']:.1f}% used)\n"
            f"📁 Format: PNG (lossless, preserves hidden data)"
        )
        
        logger.info(f"Embed success: psnr={metrics['psnr']:.2f}, ssim={metrics['ssim']:.4f}")
        
        return output_path, psnr_text, ssim_text, capacity_text
        
    except InsufficientCapacityError as e:
        logger.warning(f"Capacity error: {str(e)}")
        return None, f"❌ {str(e)}", "", ""
    
    except PixelNurError as e:
        logger.error(f"PixelNur error: {str(e)}")
        return None, f"❌ {str(e)}", "", ""
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return None, f"❌ Error: {str(e)}", "", ""


def extract_interface(stego_file: str, password: str, cover_image: Optional[np.ndarray] = None) -> str:
    """Gradio interface function for extraction."""
    try:
        if not stego_file:
            return "❌ Please upload a stego image file."
        
        # Read image directly from file to avoid Gradio conversion
        stego_bgr = cv2.imread(stego_file)
        
        if stego_bgr is None:
            return "❌ Failed to read image file. Please upload a valid PNG file."
        
        valid, error = _validate_password(password)
        if not valid:
            return f"❌ {error}"
        
        # Process optional cover image (Gradio returns RGB numpy array)
        cover_bgr = None
        if cover_image is not None:
            cover_bgr = cv2.cvtColor(cover_image, cv2.COLOR_RGB2BGR)
        
        logger.info(f"Extract request: image_shape={stego_bgr.shape}, file={stego_file}, has_cover={cover_bgr is not None}")
        
        encrypted_message = extraction_engine.extract(
            stego_image=stego_bgr,
            cover_image=cover_bgr
        )
        
        message_bytes = encryption_service.decrypt(encrypted_message, password)
        message = message_bytes.decode('utf-8')
        
        logger.info(f"Extract success: message_length={len(message)}")
        
        return f"✅ Extracted message:\n\n{message}"
        
    except ExtractionError as e:
        logger.warning(f"Extraction error: {str(e)}")
        return (
            f"❌ Extraction failed: {str(e)}\n\n"
            f"Please verify:\n"
            f"• Correct password\n"
            f"• Image hasn't been modified\n"
            f"• Image contains embedded data"
        )
    except UnicodeDecodeError as e:
        logger.warning(f"UTF-8 decode error: {str(e)}")
        return (
            f"❌ Extraction failed: Unable to decode message as UTF-8.\n\n"
            f"This usually means:\n"
            f"• Wrong password (decryption produced invalid data)\n"
            f"• Image has been modified or corrupted\n"
            f"• Image doesn't contain embedded data\n\n"
            f"Please verify your password and ensure the image hasn't been altered."
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return f"❌ Error: {str(e)}"


def create_gradio_app() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="PixelNur - Advanced Steganography") as app:
        
        gr.Markdown("""
        <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem 2rem; border-radius: 1rem; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
            <h1 style="color: white; font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
                🔒 PixelNur
            </h1>
            <p style="color: rgba(255,255,255,0.95); font-size: 1.3rem; margin-bottom: 1.5rem;">
                Advanced CNN-Based Steganography System
            </p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 2rem; color: white; font-weight: 600;">
                    🎯 PSNR >40 dB
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 2rem; color: white; font-weight: 600;">
                    🛡️ Attack Resistant
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 2rem; color: white; font-weight: 600;">
                    🔐 AES-256 Encryption
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 2rem; color: white; font-weight: 600;">
                    🚀 GPU Accelerated
                </span>
            </div>
        </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("📤 Embed Message", elem_classes="tab-nav"):
                gr.Markdown("""
                <div style="background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%); padding: 1.5rem; border-radius: 0.75rem; margin-top: 1rem; color: white;">
                    <h4 style="margin: 0 0 0.5rem 0;">⚠️ Important</h4>
                    <ul style="margin: 0; padding-left: 1.5rem;">
                        <li><strong>Stego images are always saved as PNG</strong> to preserve hidden data</li>
                        <li><strong>Upload the PNG file directly</strong> (not a screenshot)</li>
                        <li>Don't re-save, edit, or compress the image</li>
                        <li>Password must match exactly</li>
                    </ul>
                </div>

                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📸 Step 1: Upload Cover Image")
                        embed_image = gr.Image(
                            label="Cover Image",
                            type="numpy",
                            height=350,
                            elem_classes="image-upload"
                        )
                        
                        gr.Markdown("### ✍️ Step 2: Enter Your Secret")
                        embed_message = gr.Textbox(
                            label="Secret Message",
                            placeholder="Type your secret message here... 🤫",
                            lines=4,
                            elem_classes="input-field"
                        )
                        
                        gr.Markdown("### 🔑 Step 3: Set Password")
                        embed_password = gr.Textbox(
                            label="Encryption Password (min 16 characters)",
                            placeholder="Enter a strong password... 🔐",
                            type="password",
                            elem_classes="input-field"
                        )
                        
                        gr.Markdown("### 🛡️ Step 4: Choose Protection Level")
                        embed_robustness = gr.Radio(
                            label="Robustness Level",
                            choices=list(ROBUSTNESS_LEVELS.keys()),
                            value=list(ROBUSTNESS_LEVELS.keys())[0],
                            elem_classes="input-field"
                        )
                        
                        embed_button = gr.Button(
                            "🔒 Embed Message Now",
                            variant="primary",
                            size="lg",
                            elem_classes="primary-button"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 Your Stego Image")
                        stego_file = gr.File(
                            label="Download Stego Image (PNG)",
                            file_count="single"
                        )
                        
                        gr.Markdown("### 📊 Quality Metrics")
                        psnr_output = gr.Textbox(
                            label="🎯 PSNR (Peak Signal-to-Noise Ratio)",
                            interactive=False,
                            elem_classes="metric-card"
                        )
                        ssim_output = gr.Textbox(
                            label="📈 SSIM (Structural Similarity Index)",
                            interactive=False,
                            elem_classes="metric-card"
                        )
                        capacity_output = gr.Textbox(
                            label="💾 Capacity Usage",
                            interactive=False,
                            elem_classes="metric-card"
                        )
                        
                        gr.Markdown("""
                        <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); padding: 1.5rem; border-radius: 0.75rem; margin-top: 1rem; color: white;">
                            <h4 style="margin: 0 0 0.5rem 0;">⚡ Quick Tips</h4>
                            <ul style="margin: 0; padding-left: 1.5rem;">
                                <li>Higher PSNR = Better image quality</li>
                                <li>SSIM closer to 1.0 = More similar to original</li>
                                <li>Download the PNG file for extraction</li>
                            </ul>
                        </div>
                        """)
                
                embed_button.click(
                    fn=embed_interface,
                    inputs=[embed_image, embed_message, embed_password, embed_robustness],
                    outputs=[stego_file, psnr_output, ssim_output, capacity_output]
                )
            
            with gr.Tab("📥 Extract Message", elem_classes="tab-nav"):
                gr.Markdown("""
                <div style="background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%); padding: 1.5rem; border-radius: 0.75rem; margin: 1rem 0; border-left: 5px solid #6c5ce7; color: white;">
                    <h3 style="margin: 0 0 0.5rem 0;">🔓 Reveal Hidden Messages</h3>
                    <p style="margin: 0;">Upload the stego image and enter the password to extract the hidden message.</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Step 1: Upload Stego Image")
                        extract_image = gr.File(
                            label="📤 Stego Image File (Upload the PNG file you downloaded)",
                            file_types=[".png"],
                            type="filepath"
                        )

                        gr.Markdown("### 📸 Step 1.5: Upload Original Cover Image (Optional)")
                        gr.Markdown("*Note: Highly recommended for Phase 2 (CNN) stego images to achieve 100% perfect extraction.*")
                        extract_cover = gr.Image(
                            label="Original Cover Image",
                            type="numpy",
                            height=250,
                            elem_classes="image-upload"
                        )

                        gr.Markdown("### 🔑 Step 2: Enter Password")
                        extract_password = gr.Textbox(
                            label="Decryption Password",
                            placeholder="Enter the password used during embedding... 🔐",
                            type="password",
                            elem_classes="input-field"
                        )
                        
                        extract_button = gr.Button(
                            "🔓 Extract Message Now",
                            variant="primary",
                            size="lg",
                            elem_classes="primary-button"
                        )
                        
                        gr.Markdown("""
                        <div style="background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%); padding: 1.5rem; border-radius: 0.75rem; margin-top: 1rem; color: white;">
                            <h4 style="margin: 0 0 0.5rem 0;">⚠️ Important</h4>
                            <ul style="margin: 0; padding-left: 1.5rem;">
                                <li>For Phase 2 (CNN) images, uploading the original cover image is recommended to guarantee perfect bit recovery.</li>
                                <li><strong>Upload the PNG file directly</strong> (not a screenshot).</li>
                                <li>Don't re-save or edit the image.</li>
                                <li>Password must match exactly.</li>
                            </ul>
                        </div>
                        """)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 💬 Extracted Message")
                        extract_output = gr.Textbox(
                            label="Hidden Message",
                            placeholder="Your secret message will appear here... 🎉",
                            lines=15,
                            interactive=False,
                            elem_classes="input-field"
                        )
                
                extract_button.click(
                    fn=extract_interface,
                    inputs=[extract_image, extract_password, extract_cover],
                    outputs=extract_output
                )
            
            with gr.Tab("📖 Help & Info", elem_classes="tab-nav"):
                gr.Markdown("""
                <div style="max-width: 900px; margin: 0 auto;">
                    
                ## 🎯 What is PixelNur?
                
                PixelNur is an advanced steganography system that hides secret messages inside images using CNN-based adaptive embedding.
                
                ---
                
                ## 🚀 Quick Start Guide
                
                ### Embedding a Message
                1. Upload a cover image (PNG, JPEG, or BMP)
                2. Type your secret message
                3. Create a strong password (minimum 16 characters)
                4. Select robustness level
                5. Click "Embed Message" and download the PNG file
                
                ### Extracting a Message
                1. Upload the stego image (the PNG file you downloaded)
                2. Enter the same password used during embedding
                3. Click "Extract Message" to reveal the hidden text
                
                ---
                
                ## 📊 Technical Specifications
                
                - **Image Quality:** PSNR >40 dB, SSIM >0.99
                - **Encryption:** SHA-256 + XOR cipher
                - **Transform:** Lifting Wavelet Transform (LWT)
                - **Embedding:** LSB matching in frequency domain
                - **Capacity:** Adaptive based on CNN estimation
                - **Processing:** GPU-accelerated (CUDA/CPU fallback)
                
                ---
                
                ## 🔗 Links & Resources
                
                <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0;">
                    <a href="https://github.com/rushikeshxdev/pixelnur" target="_blank" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 2rem; border-radius: 0.5rem; text-decoration: none; font-weight: 600;">
                        📦 GitHub Repository
                    </a>
                    <a href="https://github.com/rushikeshxdev/pixelnur/blob/main/README.md" target="_blank" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 2rem; border-radius: 0.5rem; text-decoration: none; font-weight: 600;">
                        📚 Documentation
                    </a>
                    <a href="https://github.com/rushikeshxdev/pixelnur/issues" target="_blank" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 2rem; border-radius: 0.5rem; text-decoration: none; font-weight: 600;">
                        🐛 Report Issues
                    </a>
                </div>
                
                ---
                
                <div style="background: linear-gradient(135deg, #55efc4 0%, #00b894 100%); padding: 1.5rem; border-radius: 0.75rem; margin: 1rem 0; color: white; text-align: center;">
                    <p style="margin: 0; font-size: 0.9rem;">
                        ⚡ Powered by PixelNur | Built with ❤️ for Privacy
                    </p>
                </div>
                
                </div>
                """)
    
    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860, css=CUSTOM_CSS)
