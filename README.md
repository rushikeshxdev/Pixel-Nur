# PixelNur — Steganography Tool

Hide AES-256 encrypted messages inside images using LSB substitution.  
The stego image is visually indistinguishable from the original.

---

## What It Does

- **Embed** — Encrypts your message with AES-256-GCM and hides it in the pixel data of any image
- **Extract** — Reads the hidden bits from a stego image and decrypts the message back
- **Image Search** — Search and pick a cover image directly from inside the app (Pixabay)
- **History** — Every embed/extract session is saved in the browser (localStorage)
- **Share** — Share the stego image or extracted message to WhatsApp, Telegram, and other apps

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend | Python 3.11 + Flask | REST API, file handling |
| Steganography | NumPy | LSB bit manipulation on pixel arrays |
| Image I/O | Pillow | Open/save images, force RGB mode |
| Encryption | pycryptodome | AES-256-GCM + PBKDF2-HMAC-SHA256 |
| Image Search | Pixabay API | Free image search (API key required) |
| HTTP client | requests | Pixabay API calls + image proxy |
| Config | python-dotenv | Load API key from `.env` file |
| Frontend | HTML + Tailwind CSS | Single-page app, no build step |

---

## Prerequisites

- **Python 3.11** — [python.org/downloads](https://www.python.org/downloads/)
- **Pixabay API key** — free, no credit card required (see below)

---

## Getting a Pixabay API Key

The image search feature requires a free Pixabay API key.

1. Go to **[pixabay.com/api/docs](https://pixabay.com/api/docs/)**
2. Click **"Get a free API key"** and sign up with your email
3. After login, your API key is shown on that same page
4. Copy it — you will paste it into the `.env` file (step 3 below)

> Without the key, all other features (embed, extract, history, share) still work normally.  
> Only the "Search online" button in the Embed page will be disabled.

---

## Installation

### 1. Install dependencies

```bash
cd pixelNur
pip install -r requirements.txt
```

Or install manually:

```bash
pip install flask pillow numpy pycryptodome requests python-dotenv
```

### 2. Create the `.env` file

A template is already provided. Copy it:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### 3. Add your Pixabay API key

Open `.env` and replace the placeholder with your actual key:

```
PIXABAY_API_KEY=your_actual_key_here
```

---

## Running the App

### Option 1 — Windows launcher (easiest)

Double-click **`run_simple.bat`** in the `pixelNur` folder.

### Option 2 — Terminal

```bash
# Windows (Python 3.11 full path)
C:\Users\PRATHAMESH\AppData\Local\Programs\Python\Python311\python.exe server.py

# macOS / Linux (if python3.11 is on PATH)
python3.11 server.py
```

Then open **[http://localhost:7861](http://localhost:7861)** in your browser.

---

## How to Use

### Embed a message

1. Go to the **Embed** tab
2. Upload a cover image — drag & drop, browse, camera, or search online
3. Type your secret message
4. Enter a strong password (16+ characters recommended)
5. Choose a robustness level (None = best quality)
6. Click **Embed Message**
7. Download the output as **PNG** — never save as JPEG (JPEG compression destroys hidden bits)

### Extract a message

1. Go to the **Extract** tab
2. Upload the stego **PNG** image
3. Enter the same password used during embedding
4. Click **Extract Message**
5. The hidden message appears — use Copy or Share to send it

---

## Project Structure

```
pixelNur/
├── server.py              ← Flask backend (API routes)
├── run_simple.bat         ← Windows one-click launcher
├── .env                   ← Your API key (gitignored, never commit this)
├── .env.example           ← Template for the .env file
├── requirements.txt       ← Python dependencies
├── templates/
│   └── index.html         ← Entire frontend (single-page app)
└── src/
    ├── simple_steg.py     ← Core LSB + AES-256-GCM algorithm
    └── __init__.py
```

---

## Security Details

| Feature | Detail |
|---|---|
| Encryption | AES-256-GCM (authenticated encryption) |
| Key derivation | PBKDF2-HMAC-SHA256, 100,000 iterations |
| Salt | 16 bytes random, unique per message |
| Nonce | 16 bytes random, unique per message |
| Auth tag | 16-byte GCM tag — detects wrong password or tampering |
| Packet header | `SNUR` magic (4B) + BPC byte + 4B length |

Wrong password = clean error, not garbled output. The GCM tag fails authentication before any data is returned.

---

## Robustness Levels

| Level | Bits/Channel | Effect |
|---|---|---|
| None | 1 bit | Best image quality, ~375 KB capacity per megapixel |
| Low | 1 bit | Same as None |
| Medium | 2 bits | 2× capacity, slightly lower quality |
| High | 2 bits | Same as Medium |

---

## Team

**KIT's College of Engineering, Kolhapur**

| Name | Roll No. |
|---|---|
| Aman Qureshi | 2223000503 |
| Rushikesh Randive | 2223000930 |
| Ankita Patil | 2223000302 |
| Madhura Patil | 2223000060 |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
