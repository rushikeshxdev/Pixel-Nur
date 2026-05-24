# PixelNur — Technical Deep Dive

Complete explanation of how the project works, every technology used, and why each decision was made.

---

## What Is This Project?

**PixelNur** is a steganography tool — it hides secret text messages inside images. The image looks completely normal to the human eye after hiding, but carries an encrypted message inside its pixel data. Only someone with the correct password can extract it.

Built as a PBL (Project-Based Learning) project at KIT's College of Engineering, Kolhapur.

---

## The Core Concept: How Hiding Works

Every digital image is made of pixels. Each pixel has 3 colour channels — **Red, Green, Blue** — each stored as a number from 0–255.

Example pixel: `R=200, G=150, B=100`

In binary: `R = 11001000`

The **Least Significant Bit (LSB)** is the rightmost bit. Changing it changes the colour value by only 1 out of 255 — completely invisible to the human eye.

```
Original:   11001000  = 200
Modified:   11001001  = 201   ← 1 LSB changed, difference = 1
```

PixelNur exploits this. It overwrites the last 1 or 2 bits of every channel in every pixel with bits from the secret message. A 1000×1000 pixel image has 3 million channels — you can hide ~375 KB of data using 1 bit per channel.

---

## Full Data Flow — Step by Step

### Embed Flow

```
User types message + password + picks image
         ↓
Browser sends multipart/form-data POST to /api/embed
         ↓
Flask receives it → PIL opens the image → convert("RGB")
         ↓
simple_steg.embed() is called
         ↓
  1. AES-256-GCM encrypts the message
  2. Packet is assembled: MAGIC + BPC + LENGTH + SALT + NONCE + CIPHERTEXT + TAG
  3. Packet is converted to a stream of bits: [0,1,1,0,1,0,...]
  4. Each bit is written into the LSB of consecutive pixel channels
  5. NumPy does this in a flat array in microseconds
         ↓
Result is a modified PIL image (visually identical)
         ↓
Saved as PNG into a BytesIO buffer → base64 encoded
         ↓
Flask returns JSON: { "image": "data:image/png;base64,...", "metrics": {...} }
         ↓
Browser renders the stego image, user downloads it
```

### Extract Flow

```
User uploads stego PNG + enters password
         ↓
Flask receives it → PIL opens image
         ↓
simple_steg.extract() is called
         ↓
  1. Read LSBs from pixel channels → reconstruct bit stream
  2. Check first 4 bytes = "SNUR" magic header (validates it's a PixelNur image)
  3. Read BPC (bits per channel used during embed)
  4. Read payload length
  5. Read exactly that many bytes of encrypted payload
  6. AES-256-GCM decrypts it using the password
  7. If wrong password → GCM authentication tag fails → raises "Wrong password"
         ↓
Flask returns JSON: { "message": "the secret text" }
         ↓
Browser shows the message
```

---

## Tech Stack — Every Library Explained

### 1. Python 3.11

**Why:** Runs everything server-side. Python 3.11 specifically because `pycryptodome` and the other heavy libraries were installed there. Python 3.9 was on the machine but lacked these packages.

---

### 2. NumPy — `numpy`

**Used in:** `src/simple_steg.py` — the heart of LSB manipulation

**What it does:**

```python
img_array = np.array(img, dtype=np.uint8)   # PIL image → 3D array [H, W, 3]
flat = img_array.flatten()                   # → 1D array of millions of bytes
flat[i] = (flat[i] & lsb_mask) | chunk      # zero out LSBs, write our bits
result_array = flat.reshape(img_array.shape) # back to 3D
```

**Why NumPy:** A 1000×1000 image has 3 million bytes. Without NumPy, a Python loop would take seconds. NumPy runs this as C-speed vectorized operations — effectively instantaneous. The `lsb_mask` trick is a bitwise AND that zeroes out the last N bits of every channel at once.

```
bpc=1 → lsb_mask = 0xFE = 11111110  (clears last 1 bit)
bpc=2 → lsb_mask = 0xFC = 11111100  (clears last 2 bits)
```

---

### 3. Pillow (PIL) — `Pillow`

**Used in:** `src/simple_steg.py` and `server.py`

**What it does:**
- Opens any image format (JPEG, PNG, BMP, WebP) → `Image.open()`
- Forces RGB mode → `.convert("RGB")` — strips alpha channel and normalises colour space
- Creates the output stego image → `Image.fromarray(result_array, "RGB")`
- Saves as PNG → `stego.save(buf, format="PNG")`

**Why PNG is mandatory for output:** JPEG uses lossy compression — it slightly changes pixel values to reduce file size. Even one changed pixel destroys the hidden bits. PNG is lossless — every pixel value is preserved exactly as written. This is why the UI warns "save as PNG, do not re-save as JPEG."

---

### 4. pycryptodome — `Crypto.*`

**Used in:** `src/simple_steg.py` — all encryption/decryption

This is the most security-critical library. It provides:

#### a) `AES.MODE_GCM` — AES-256-GCM Encryption

**AES-256** = Advanced Encryption Standard with a 256-bit key. Military-grade. The US government uses it for top-secret data. Brute-forcing a 256-bit key would take longer than the age of the universe even with every computer on Earth.

**GCM (Galois/Counter Mode)** = an "authenticated encryption" mode. It simultaneously:
1. **Encrypts** the message so no one can read it without the key
2. **Produces an authentication tag** (16 bytes) — a cryptographic fingerprint of the ciphertext

```python
cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=16)
ciphertext, tag = cipher.encrypt_and_digest(message.encode("utf-8"))
```

During extraction, if the wrong password is used, the tag verification fails:

```python
cipher.decrypt_and_verify(ciphertext, tag)
# → raises ValueError if tag doesn't match (wrong key = wrong password)
```

Wrong password = **clean error**, not garbled output.

#### b) `PBKDF2` — Password Key Derivation

Passwords are not used directly as encryption keys. A user might type `"hello123"` — that's only 8 chars, far weaker than a 256-bit key. PBKDF2 solves this:

```python
PBKDF2(
    password.encode("utf-8"),
    salt,            # 16 random bytes — different every time
    dkLen=32,        # output = 32 bytes = 256 bits (perfect for AES-256)
    count=100_000,   # runs SHA-256 100,000 times
    hmac_hash_module=SHA256,
)
```

**Why 100,000 iterations:** An attacker brute-forcing the password must run PBKDF2 100,000 times for every single guess. At 100k iterations, even a GPU doing 1 billion guesses/second is slowed to 10,000 effective guesses/second — making dictionary attacks practically infeasible.

**Why the random salt:** Even if two users embed the same message with the same password, their salts are different random bytes → different keys → completely different ciphertexts. Prevents rainbow-table attacks.

#### c) `get_random_bytes` — Cryptographically Secure Randomness

Used to generate the 16-byte salt and 16-byte nonce. Unlike Python's `random` module (which is predictable), this pulls from the OS's cryptographic random number generator (`/dev/urandom` on Linux, `BCryptGenRandom` on Windows).

---

### 5. The Packet Format — Binary Protocol Design

This is the custom binary structure embedded in the image. Designed so extraction works without any side-channel (no separate file, no metadata):

```
Bytes  0–3  : MAGIC = b'SNUR'          (4 bytes) — identifies a PixelNur image
Byte   4    : BPC = 1 or 2             (1 byte)  — how many LSBs were used
Bytes  5–8  : MSG_LEN (big-endian u32) (4 bytes) — encrypted payload length
Bytes  9–24 : SALT                     (16 bytes) — for PBKDF2 key derivation
Bytes 25–40 : NONCE                    (16 bytes) — AES-GCM nonce
Bytes 41–N  : CIPHERTEXT               (variable) — encrypted message
Last 16     : GCM TAG                  (16 bytes) — authentication tag
```

**Total overhead:** 57 bytes minimum (header + salt + nonce + tag), regardless of message length.

The **MAGIC header** (`SNUR`) is a fast validity check — if the first 4 bytes extracted don't spell `SNUR`, it's the wrong image or wrong BPC. This is how auto-detection works: try BPC=1 first, if magic doesn't match, try BPC=2.

---

### 6. Flask — `flask`

**Used in:** `server.py`

Flask is a Python micro web framework. It turns Python functions into HTTP endpoints.

**Why Flask over Django/FastAPI:** Flask is minimal — no ORM, no admin panel, no unnecessary overhead. This app only needs 5 routes.

```python
app = Flask(__name__, template_folder="templates")
```

| Route | Method | Purpose |
|---|---|---|
| `GET /` | GET | Serves `index.html` |
| `POST /api/embed` | POST | Receives image+message+password, returns stego PNG as base64 JSON |
| `POST /api/extract` | POST | Receives stego PNG+password, returns decrypted message |
| `GET /api/search-images` | GET | Queries Pixabay API, returns list of image URLs |
| `GET /api/fetch-image` | GET | Proxies external image URLs to bypass browser CORS |

**Why the `/api/fetch-image` proxy exists:** Browsers block JavaScript from fetching images from other domains (CORS policy). When a user picks an image from Pixabay, the browser can't fetch `cdn.pixabay.com/...` and convert it to a file directly. The proxy fetches it server-side (Python has no CORS restriction), base64-encodes it, and returns it as a data URL.

**`MAX_CONTENT_LENGTH = 32 MB`:** Flask rejects uploaded files larger than this, preventing memory exhaustion attacks.

---

### 7. python-dotenv — `dotenv`

**Used in:** `server.py` — `load_dotenv()`

Loads the `.env` file into environment variables at startup:

```
# .env file
PIXABAY_API_KEY=56007047-b0cb75...
```

```python
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")
```

**Why:** Never hardcode API keys in source code. If you push to GitHub with a hardcoded key, bots scan GitHub in minutes and steal it. The `.env` file is gitignored — it only exists locally on the developer's machine.

---

### 8. Pixabay API + requests

**Used in:** `server.py` — `search_images()` route

The `requests` library makes outbound HTTP calls from the server to Pixabay's API:

```python
resp = http.get("https://pixabay.com/api/", params={
    "key":        PIXABAY_API_KEY,
    "q":          "kolhapur",
    "image_type": "photo",
    "per_page":   15,
})
hits = resp.json()["hits"]
```

Pixabay returns JSON with image objects. We extract `previewURL` (small thumbnail) and `webformatURL` (medium-res for actual use), and send them to the browser. The browser then calls `/api/fetch-image` to proxy the selected image.

---

### 9. Tailwind CSS (CDN)

**Used in:** `templates/index.html` — all styling

Tailwind is a utility-first CSS framework. Instead of writing custom CSS classes, you compose styles inline:

```html
<button class="bg-brand text-white px-4 py-2 rounded-md hover:bg-orange-700">
```

**Why CDN (no build step):** A single `<script src="https://cdn.tailwindcss.com">` tag is all that's needed. No Node.js, no npm, no webpack — perfect for a pure Python project.

**Custom brand colour:**
```javascript
tailwind.config = { theme: { extend: { colors: { brand: '#ff6600' } } } }
```
This registers `bg-brand`, `text-brand`, `border-brand` as usable Tailwind classes throughout the app.

---

### 10. Single-Page Application (SPA) Architecture

**Used in:** `templates/index.html`

There is only **one HTML file**, but it behaves like multiple pages:

```javascript
const PAGES = ['home', 'embed', 'extract', 'history', 'about'];
function showPage(name) {
  PAGES.forEach(p => document.getElementById('page-' + p).classList.add('hidden'));
  document.getElementById('page-' + name).classList.remove('hidden');
}
```

Every "page" is a `<div>` that is either hidden or visible. Navigation toggles CSS classes — no page reloads, no server requests. This is why the app feels instant.

---

### 11. localStorage — History Persistence

```javascript
const HIST_KEY = 'pixelnur_history';
localStorage.setItem(HIST_KEY, JSON.stringify(history));
```

The browser's `localStorage` is a key-value store that survives page refreshes and browser restarts. Every embed/extract session saves a record (timestamp, filename, message preview, metrics). Max 30 entries. Entirely client-side — no server or database needed.

---

### 12. Canvas API — Hero Background Animation

```javascript
const canvas = document.getElementById('hero-canvas');
const ctx = canvas.getContext('2d');
```

The animated pixel grid on the home page is drawn using the HTML5 Canvas API — pure JavaScript, no library. Each dot in the grid slowly fades in and out, alternating between orange and indigo, simulating the LSB bit pattern visually. Runs at 60 fps via `requestAnimationFrame`. Stops running when the home page is hidden (MutationObserver) to save CPU.

---

### 13. Web Share API + Clipboard API — Sharing

```javascript
// Mobile: native OS share sheet — image only, no text
navigator.share({ files: [file] })

// Desktop: copy image to clipboard, user pastes with Ctrl+V
navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
```

**Web Share API** — on Android/iOS, opens the native share sheet (WhatsApp, Instagram, Telegram, etc.) with just the image file. Requires HTTPS in production.

**Clipboard API** — on desktop (where Web Share with files requires HTTPS), copies the PNG directly to the system clipboard. The user presses `Ctrl+V` in WhatsApp Web or Telegram Web to paste and send it.

**Why Web Share API fails on localhost:** Browsers enforce a "secure context" rule — `navigator.share()` with files only works over HTTPS or on `localhost` for *text*, not files. Since the app runs on `http://localhost`, file sharing falls back to the clipboard approach.

---

### 14. PSNR / MSE Quality Metrics

After embedding, two image quality metrics are calculated and shown to the user:

**MSE (Mean Squared Error):**
```python
mse = np.mean((original_array - stego_array) ** 2)
```
Average squared difference per channel value. With 1 LSB changed, MSE ≈ 0.25. Essentially zero perceptual difference.

**PSNR (Peak Signal-to-Noise Ratio):**
```python
psnr = 10 * np.log10(255.0 ** 2 / mse)
```
Measured in dB. Higher = better quality.

| Scenario | PSNR | Perception |
|---|---|---|
| Identical images | ∞ dB | No difference |
| PixelNur 1-bit BPC | ~51 dB | Imperceptible |
| PixelNur 2-bit BPC | ~45 dB | Imperceptible |
| JPEG compression | 30–40 dB | Slight artifacts |

These numbers prove to anyone analysing the output that the stego image is visually indistinguishable from the original.

---

## Why We Replaced the Original Code

The original project used:
- **CNN (Convolutional Neural Network)** — adaptive embedding mask based on texture analysis
- **LWT (Lifting Wavelet Transform)** — work in the frequency domain instead of spatial domain
- **PyTorch, OpenCV, PyWavelets, reedsolo** (Reed-Solomon error correction)
- **Gradio** — web UI framework

It was theoretically more robust (survives JPEG compression better) but practically broken — PyTorch version conflicts, CNN mask dimension bugs, Gradio API changes between versions. The complexity made it unmaintainable.

| | Original PixelNur | Current PixelNur |
|---|---|---|
| Hiding method | CNN + LWT + spread spectrum | LSB substitution |
| Encryption | AES-256-CBC | AES-256-GCM (authenticated) |
| Dependencies | PyTorch, OpenCV, PyWavelets, reedsolo… | Pillow, NumPy, pycryptodome |
| GPU required | Yes (for CNN) | No |
| Reliability | Unstable | Stable |
| Core code lines | 5000+ | ~220 |
| Processing time | 2–15 seconds | < 1 second |

The trade-off: the current version doesn't survive JPEG re-compression (hence the "save as PNG" requirement), but it works reliably every time.

---

## Security Attack Resistance

| Attack | Result |
|---|---|
| Brute-force password | Infeasible — PBKDF2 with 100k iterations slows GPU attacks to ~10k guesses/sec |
| Rainbow table | Prevented — random 16-byte salt makes every embed unique |
| Tampered image | Detected — GCM authentication tag fails if any bit is changed |
| Wrong password | Clean error — GCM tag mismatch, no data returned |
| Statistical analysis (steganalysis) | 1-bit LSB changes are near-undetectable at PSNR ~51 dB |

---

## File Structure

```
pixelNur/
├── server.py              ← Flask backend (5 API routes)
├── run_simple.bat         ← Windows one-click launcher
├── .env                   ← API keys — gitignored, never commit
├── .env.example           ← Safe template to share with others
├── requirements.txt       ← 6 Python dependencies
├── README.md              ← Setup and usage instructions
├── TECHNICAL.md           ← This file — deep technical explanation
├── LICENSE
├── .gitignore
├── templates/
│   └── index.html         ← Entire frontend (~950 lines, single-page app)
└── src/
    ├── simple_steg.py     ← Core algorithm: LSB embed/extract + AES-256-GCM
    └── __init__.py        ← Python package marker
```

---

## Research Team

**KIT's College of Engineering, Kolhapur**

| Name | Roll No. |
|---|---|
| Aman Qureshi | 2223000503 |
| Rushikesh Randive | 2223000930 |
| Ankita Patil | 2223000302 |
| Madhura Patil | 2223000060 |

---

*MIT License — Built for privacy and information security research*
