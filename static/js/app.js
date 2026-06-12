// ── Pixel LSB canvas animation ────────────────────────
        (function () {
          const canvas = document.getElementById('hero-canvas');
          const ctx = canvas.getContext('2d');
          const SZ = 7, GAP = 3, STEP = SZ + GAP;
          let pixels = [], W, H, raf;

          function resize() {
            W = canvas.width = canvas.offsetWidth;
            H = canvas.height = canvas.offsetHeight;
            pixels = [];
            const cols = Math.ceil(W / STEP) + 1;
            const rows = Math.ceil(H / STEP) + 1;
            for (let r = 0; r < rows; r++) {
              for (let c = 0; c < cols; c++) {
                // Randomly give some pixels an "active" LSB color
                const active = Math.random() < 0.18;
                const bright = active ? 0.4 + Math.random() * 0.5 : Math.random() * 0.12;
                pixels.push({
                  x: c * STEP, y: r * STEP,
                  v: bright,
                  target: active ? 0.3 + Math.random() * 0.6 : Math.random() * 0.08,
                  speed: 0.002 + Math.random() * 0.012,
                  hue: Math.random() < 0.7 ? 24 : 220,   // orange (24) or blue (220)
                });
              }
            }
          }

          function draw() {
            ctx.clearRect(0, 0, W, H);
            for (const p of pixels) {
              if (Math.abs(p.v - p.target) < 0.005) {
                p.target = Math.random() < 0.15 ? 0.35 + Math.random() * 0.55 : Math.random() * 0.1;
                p.speed = 0.002 + Math.random() * 0.01;
              }
              p.v += (p.target - p.v) * p.speed;
              const a = Math.max(0, Math.min(1, p.v));
              if (a < 0.01) continue;
              ctx.fillStyle = p.hue === 24
                ? `rgba(255,102,0,${a})`
                : `rgba(99,102,241,${a * 0.6})`;
              ctx.fillRect(p.x, p.y, SZ, SZ);
            }
            raf = requestAnimationFrame(draw);
          }

          function start() {
            if (raf) cancelAnimationFrame(raf);
            resize(); draw();
          }

          // Start on first load (after layout settles) and on re-navigation
          const home = document.getElementById('page-home');
          const obs = new MutationObserver(() => {
            if (!home.classList.contains('hidden')) requestAnimationFrame(start);
            else { if (raf) { cancelAnimationFrame(raf); raf = null; } }
          });
          obs.observe(home, { attributes: true, attributeFilter: ['class'] });

          // Initial start (home is shown by showPage('home') above)
          requestAnimationFrame(start);

          window.addEventListener('resize', () => {
            if (!home.classList.contains('hidden')) start();
          });
        })();

// ── Navigation ───────────────────────────────────────────
    const PAGES = ['home', 'embed', 'extract', 'history', 'about'];
    function showPage(name) {
      PAGES.forEach(p => {
        document.getElementById('page-' + p).classList.add('hidden');
        document.getElementById('nav-' + p).classList.remove('active');
      });
      document.getElementById('page-' + name).classList.remove('hidden');
      document.getElementById('nav-' + name).classList.add('active');
      if (name === 'history') renderHistory();
    }
    showPage('home');


    // ── Drag-and-drop ────────────────────────────────────────
    function setupDrop(fileId, dropId, innerId) {
      const file = document.getElementById(fileId);
      const drop = document.getElementById(dropId);
      const inner = document.getElementById(innerId);
      file.addEventListener('change', () => { if (file.files[0]) previewImg(file.files[0], inner); });
      drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
      drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
      drop.addEventListener('drop', e => {
        e.preventDefault(); drop.classList.remove('drag-over');
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) {
          const dt = new DataTransfer(); dt.items.add(f); file.files = dt.files;
          previewImg(f, inner);
        }
      });
    }
    function previewImg(f, inner) {
      const r = new FileReader();
      r.onload = e => { inner.innerHTML = `<img src="${e.target.result}" class="h-24 rounded object-contain mb-1 pointer-events-none"><span class="text-xs text-gray-400 pointer-events-none">${f.name}</span>`; };
      r.readAsDataURL(f);
    }
    setupDrop('embed-file', 'embed-drop', 'embed-drop-inner');
    setupDrop('ext-file', 'ext-drop', 'ext-drop-inner');


    // ── Embed ────────────────────────────────────────────────
    async function doEmbed() {
      const file = document.getElementById('embed-file').files[0];
      const msg = document.getElementById('embed-msg').value.trim();
      const pw = document.getElementById('embed-pw').value;
      const rob = document.getElementById('embed-rob').value;
      const err = document.getElementById('embed-err');
      const spin = document.getElementById('embed-spin');
      const label = document.getElementById('embed-btn-label');
      err.classList.add('hidden');
      if (!file) { showErr(err, 'Upload a cover image first.'); return; }
      if (!msg) { showErr(err, 'Enter a secret message.'); return; }
      if (!pw) { showErr(err, 'Enter a password.'); return; }

      spin.classList.remove('hidden'); label.textContent = 'Embedding…';
      const form = new FormData();
      form.append('image', file); form.append('message', msg);
      form.append('password', pw); form.append('robustness', rob);

      try {
        const data = await post('/api/embed', form);
        if (data.error) { showErr(err, data.error); return; }

        document.getElementById('embed-out-img').src = data.image;
        document.getElementById('embed-dl').href = data.image;

        const table = document.getElementById('embed-metrics');
        table.innerHTML = Object.entries(data.metrics).map(([k, v]) =>
          `<tr class="border-b border-gray-50"><td class="py-1.5 pr-3 text-gray-400">${k}</td><td class="py-1.5 font-medium text-gray-800">${v}</td></tr>`
        ).join('');

        document.getElementById('embed-output').classList.remove('hidden');
        document.getElementById('embed-placeholder').classList.add('hidden');

        addHistory({
          type: 'embed', time: new Date().toISOString(),
          file: file.name, msgPreview: msg.substring(0, 60) + (msg.length > 60 ? '…' : ''),
          metrics: data.metrics
        });

      } catch (e) { showErr(err, 'Request failed — is the server running?'); }
      finally { spin.classList.add('hidden'); label.textContent = 'Embed Message'; }
    }


    // ── Extract ──────────────────────────────────────────────
    async function doExtract() {
      const file = document.getElementById('ext-file').files[0];
      const pw = document.getElementById('ext-pw').value;
      const err = document.getElementById('ext-err');
      const spin = document.getElementById('ext-spin');
      const label = document.getElementById('ext-btn-label');
      err.classList.add('hidden');
      if (!file) { showErr(err, 'Upload a stego image first.'); return; }
      if (!pw) { showErr(err, 'Enter the password.'); return; }

      spin.classList.remove('hidden'); label.textContent = 'Extracting…';
      const form = new FormData();
      form.append('image', file); form.append('password', pw);

      try {
        const data = await post('/api/extract', form);
        if (data.error) { showErr(err, data.error); return; }

        document.getElementById('ext-result').value = data.message;
        document.getElementById('ext-output').classList.remove('hidden');
        document.getElementById('ext-placeholder').classList.add('hidden');

        addHistory({
          type: 'extract', time: new Date().toISOString(),
          file: file.name, message: data.message
        });

      } catch (e) { showErr(err, 'Request failed — is the server running?'); }
      finally { spin.classList.add('hidden'); label.textContent = 'Extract Message'; }
    }


    // ── Share Modal ──────────────────────────────────────────
    let _shareMode = 'text'; // 'text' or 'image'

    const _platforms = [
      { name: 'WhatsApp', color: '#25D366', home: 'https://web.whatsapp.com/', icon: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/></svg>', url: t => `https://wa.me/?text=${encodeURIComponent(t)}` },
      { name: 'Telegram', color: '#229ED9', home: 'https://web.telegram.org/', icon: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/></svg>', url: t => `https://t.me/share/url?url=&text=${encodeURIComponent(t)}` },
      { name: 'Twitter', color: '#000000', home: 'https://twitter.com/', icon: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.744l7.73-8.835L1.254 2.25H8.08l4.259 5.63 5.905-5.63zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>', url: t => `https://twitter.com/intent/tweet?text=${encodeURIComponent(t)}` },
      { name: 'Facebook', color: '#1877F2', home: 'https://www.facebook.com/', icon: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>', url: t => `https://www.facebook.com/sharer/sharer.php?quote=${encodeURIComponent(t)}` },
      { name: 'LinkedIn', color: '#0A66C2', home: 'https://www.linkedin.com/', icon: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>', url: t => `https://www.linkedin.com/sharing/share-offsite/?url=&summary=${encodeURIComponent(t)}` },
      { name: 'Reddit', color: '#FF4500', home: 'https://www.reddit.com/', icon: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z"/></svg>', url: t => `https://reddit.com/submit?title=Hidden+Message&text=${encodeURIComponent(t)}` },
      { name: 'Email', color: '#6b7280', home: 'mailto:?subject=PixelNur+Stego+Image&body=See+attached+image.', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>', url: t => `mailto:?subject=Hidden%20Message%20via%20PixelNur&body=${encodeURIComponent(t)}` },
    ];

    function openShareModal(mode) {
      _shareMode = mode;
      const dlBtn = document.getElementById('share-dl-btn');
      const platEl = document.getElementById('share-platforms');

      if (mode === 'image') {
        dlBtn.style.display = '';
        document.getElementById('share-copy-label').textContent = 'Copy image to clipboard';
        platEl.innerHTML = _platforms.map(p => `
          <button onclick="shareStegoToApp('${p.home}')" title="Share on ${p.name}"
            style="display:flex;flex-direction:column;align-items:center;gap:6px;background:none;border:none;cursor:pointer;padding:0;">
            <div style="width:44px;height:44px;border-radius:12px;background:${p.color};display:flex;align-items:center;justify-content:center;color:#fff;transition:opacity .15s;"
              onmouseover="this.style.opacity='.8'" onmouseout="this.style.opacity='1'">
              <div style="width:22px;height:22px;">${p.icon}</div>
            </div>
            <span style="font-size:.65rem;color:#6b7280;">${p.name}</span>
          </button>`).join('');
      } else {
        dlBtn.style.display = 'none';
        document.getElementById('share-copy-label').textContent = 'Copy to clipboard';
        const msg = document.getElementById('ext-result').value || '';
        const preview = msg.length > 100 ? msg.slice(0, 100) + '…' : msg;
        platEl.innerHTML = _platforms.map(p => `
          <a href="${p.url(preview)}" target="_blank" rel="noopener"
            style="display:flex;flex-direction:column;align-items:center;gap:6px;text-decoration:none;cursor:pointer;"
            title="Share on ${p.name}">
            <div style="width:44px;height:44px;border-radius:12px;background:${p.color};display:flex;align-items:center;justify-content:center;color:#fff;">
              <div style="width:22px;height:22px;">${p.icon}</div>
            </div>
            <span style="font-size:.65rem;color:#6b7280;">${p.name}</span>
          </a>`).join('');
      }

      document.getElementById('share-modal').classList.remove('hidden');
    }

    function closeShareModal() {
      document.getElementById('share-modal').classList.add('hidden');
    }
    document.getElementById('share-modal').addEventListener('click', e => {
      if (e.target === e.currentTarget) closeShareModal();
    });

    async function shareModalCopy() {
      const lbl = document.getElementById('share-copy-label');
      if (_shareMode === 'image') {
        try {
          const dataUrl = document.getElementById('embed-out-img').src;
          const blob = await (await fetch(dataUrl)).blob();
          await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
          lbl.textContent = '✓ Image copied!';
        } catch (e) { lbl.textContent = 'Copy not supported — use Download instead.'; }
      } else {
        const msg = document.getElementById('ext-result').value;
        await navigator.clipboard.writeText(msg);
        lbl.textContent = '✓ Copied!';
      }
      setTimeout(() => { lbl.textContent = _shareMode === 'image' ? 'Copy image to clipboard' : 'Copy to clipboard'; }, 2500);
    }

    function shareModalDownload() {
      document.getElementById('embed-dl').click();
      closeShareModal();
    }

    async function shareStegoToApp(platformHome) {
      const dataUrl = document.getElementById('embed-out-img').src;
      if (!dataUrl) return;

      const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

      if (isMobile) {
        // Mobile: native OS share sheet — image only, no text
        try {
          const blob = await (await fetch(dataUrl)).blob();
          const file = new File([blob], 'stego.png', { type: 'image/png' });
          if (navigator.canShare && navigator.canShare({ files: [file] })) {
            await navigator.share({ files: [file] });
            closeShareModal();
            return;
          }
        } catch (e) {
          if (e.name === 'AbortError') { closeShareModal(); return; }
        }
      }

      // Desktop: copy image to clipboard → open platform → user pastes with Ctrl+V
      try {
        const blob = await (await fetch(dataUrl)).blob();
        // Convert to PNG for clipboard compatibility
        const pngBlob = await new Promise(res => {
          const img = new Image();
          img.onload = () => {
            const c = document.createElement('canvas');
            c.width = img.naturalWidth; c.height = img.naturalHeight;
            c.getContext('2d').drawImage(img, 0, 0);
            c.toBlob(res, 'image/png');
          };
          img.src = URL.createObjectURL(blob);
        });
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': pngBlob })]);
        if (platformHome) window.open(platformHome, '_blank');
        closeShareModal();
        showToast('Image copied to clipboard — open a chat and press Ctrl+V to send it.');
      } catch (e) {
        // Clipboard API unavailable — fall back to download
        document.getElementById('embed-dl').click();
        if (platformHome) window.open(platformHome, '_blank');
        closeShareModal();
        showToast('Image downloaded — attach it manually from your Downloads folder.');
      }
    }

    function showToast(msg) {
      let t = document.getElementById('share-toast');
      if (!t) {
        t = document.createElement('div');
        t.id = 'share-toast';
        t.style.cssText = 'position:fixed;bottom:28px;left:50%;transform:translateX(-50%);background:#1f2937;color:#fff;font-size:.78rem;padding:10px 20px;border-radius:8px;z-index:9999;box-shadow:0 4px 16px rgba(0,0,0,.25);pointer-events:none;transition:opacity .3s;';
        document.body.appendChild(t);
      }
      t.textContent = msg;
      t.style.opacity = '1';
      clearTimeout(t._timer);
      t._timer = setTimeout(() => { t.style.opacity = '0'; }, 3500);
    }

    function shareStego() { openShareModal('image'); }
    function shareMessage() { openShareModal('text'); }

    async function copyMessage() {
      const msg = document.getElementById('ext-result').value;
      if (!msg) return;
      await navigator.clipboard.writeText(msg);
      const lbl = document.getElementById('copy-label');
      lbl.textContent = 'Copied!';
      setTimeout(() => lbl.textContent = 'Copy', 2000);
    }


    // ── Image search ─────────────────────────────────────────
    function slideSearch(dir) {
      document.getElementById('search-results').scrollBy({ left: dir * 370, behavior: 'smooth' });
    }

    function openSearch() {
      document.getElementById('search-modal').classList.remove('hidden');
      document.getElementById('search-q').focus();
    }
    function closeSearch() {
      document.getElementById('search-modal').classList.add('hidden');
    }
    document.getElementById('search-modal').addEventListener('click', e => {
      if (e.target === e.currentTarget) closeSearch();
    });

    async function searchImages() {
      const q = document.getElementById('search-q').value.trim();
      if (!q) return;
      const status = document.getElementById('search-status');
      const results = document.getElementById('search-results');
      results.innerHTML = '';
      status.textContent = 'Searching…'; status.classList.remove('hidden');

      try {
        const res = await fetch('/api/search-images?q=' + encodeURIComponent(q));
        const data = await res.json();
        status.classList.add('hidden');

        if (data.error) { status.textContent = data.error; status.classList.remove('hidden'); return; }
        if (!data.images.length) { status.textContent = 'No results found.'; status.classList.remove('hidden'); return; }

        results.innerHTML = data.images.map(img => `
          <button onclick="selectImage('${img.full}')"
            style="flex:0 0 170px;height:120px;border-radius:10px;overflow:hidden;border:2px solid transparent;cursor:pointer;padding:0;background:#f3f4f6;transition:border-color .15s;"
            onmouseover="this.style.borderColor='#ff6600'" onmouseout="this.style.borderColor='transparent'">
            <img src="${img.thumb}" style="width:100%;height:100%;object-fit:cover;" loading="lazy">
          </button>`).join('');
      } catch (e) {
        status.textContent = 'Search failed. Check your connection.'; status.classList.remove('hidden');
      }
    }

    async function selectImage(url) {
      const status = document.getElementById('search-status');
      const inner = document.getElementById('embed-drop-inner');
      status.textContent = 'Loading image…'; status.classList.remove('hidden');
      document.getElementById('search-results').innerHTML = '';

      try {
        const data = await (await fetch('/api/fetch-image?url=' + encodeURIComponent(url))).json();
        if (data.error) { status.textContent = data.error; return; }

        // Load into embed file input
        const blob = await (await fetch(data.image)).blob();
        const file = new File([blob], 'searched.jpg', { type: blob.type });
        const dt = new DataTransfer(); dt.items.add(file);
        document.getElementById('embed-file').files = dt.files;

        inner.innerHTML = `<img src="${data.image}" class="h-24 rounded object-contain mb-1 pointer-events-none"><span class="text-xs text-gray-400 pointer-events-none">searched.jpg</span>`;
        closeSearch();
      } catch (e) {
        status.textContent = 'Failed to load image.'; status.classList.remove('hidden');
      }
    }


    // ── History (localStorage) ───────────────────────────────
    const HIST_KEY = 'pixelnur_history';

    function addHistory(entry) {
      const h = JSON.parse(localStorage.getItem(HIST_KEY) || '[]');
      h.unshift(entry);
      if (h.length > 30) h.length = 30;
      localStorage.setItem(HIST_KEY, JSON.stringify(h));
      updateHistBadge();
    }

    function clearHistory() {
      if (!confirm('Clear all history?')) return;
      localStorage.removeItem(HIST_KEY);
      updateHistBadge();
      renderHistory();
    }

    function updateHistBadge() {
      const h = JSON.parse(localStorage.getItem(HIST_KEY) || '[]');
      const badge = document.getElementById('hist-count');
      if (h.length) { badge.textContent = h.length; badge.classList.remove('hidden'); }
      else { badge.classList.add('hidden'); }
    }

    function renderHistory() {
      const list = document.getElementById('history-list');
      const h = JSON.parse(localStorage.getItem(HIST_KEY) || '[]');
      if (!h.length) {
        list.innerHTML = `<div class="text-center py-20"><div class="text-5xl opacity-20 mb-3">📋</div><p class="text-sm text-gray-400">No history yet. Embed or extract a message to get started.</p></div>`;
        return;
      }
      list.innerHTML = h.map((entry, i) => {
        const t = new Date(entry.time);
        const ts = t.toLocaleDateString() + ' ' + t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        if (entry.type === 'embed') {
          const metricsHtml = entry.metrics
            ? `<div class="mt-2 flex flex-wrap gap-x-4 gap-y-1">${Object.entries(entry.metrics).slice(0, 4).map(([k, v]) => `<span class="text-xs text-gray-400"><span class="text-gray-600">${k}:</span> ${v}</span>`).join('')}</div>`
            : '';
          return `
            <div class="border border-gray-100 rounded-lg px-5 py-4 mb-3 hover:border-gray-200 transition-colors">
              <div class="flex items-start justify-between">
                <div class="flex items-center gap-2">
                  <span class="hist-badge badge-embed">Embed</span>
                  <span class="text-xs text-gray-500">${entry.file || 'image'}</span>
                </div>
                <span class="text-xs text-gray-400">${ts}</span>
              </div>
              <p class="text-sm text-gray-700 mt-2">"${entry.msgPreview}"</p>
              ${metricsHtml}
            </div>`;
        } else {
          const preview = entry.message ? entry.message.substring(0, 120) + (entry.message.length > 120 ? '…' : '') : '';
          return `
            <div class="border border-gray-100 rounded-lg px-5 py-4 mb-3 hover:border-gray-200 transition-colors">
              <div class="flex items-start justify-between">
                <div class="flex items-center gap-2">
                  <span class="hist-badge badge-extract">Extract</span>
                  <span class="text-xs text-gray-500">${entry.file || 'image'}</span>
                </div>
                <span class="text-xs text-gray-400">${ts}</span>
              </div>
              <p class="text-sm text-gray-700 mt-2 font-mono">"${preview}"</p>
            </div>`;
        }
      }).join('');
    }

    // Init badge on load
    updateHistBadge();


    // ── Camera ───────────────────────────────────────────────
    let _camStream = null;
    let _camTarget = 'embed';
    let _capturedBlob = null;

    async function openCamera(target) {
      _camTarget = target;
      _capturedBlob = null;
      const video = document.getElementById('cam-video');
      const canvas = document.getElementById('cam-canvas');
      const err = document.getElementById('cam-err');
      const capBtn = document.getElementById('cam-capture-btn');
      const retBtn = document.getElementById('cam-retake-btn');
      const useBtn = document.getElementById('cam-use-btn');

      err.classList.add('hidden');
      video.style.display = 'block';
      canvas.style.display = 'none';
      capBtn.style.display = '';
      retBtn.style.display = 'none';
      useBtn.style.display = 'none';

      document.getElementById('camera-modal').classList.remove('hidden');

      try {
        _camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
        video.srcObject = _camStream;
      } catch (e) {
        const msg = e.name === 'NotAllowedError'
          ? 'Camera permission denied. Allow camera access in your browser settings.'
          : 'Camera not available: ' + e.message;
        document.getElementById('cam-err').textContent = msg;
        document.getElementById('cam-err').classList.remove('hidden');
        document.getElementById('cam-capture-btn').disabled = true;
      }
    }

    function closeCamera() {
      document.getElementById('camera-modal').classList.add('hidden');
      if (_camStream) { _camStream.getTracks().forEach(t => t.stop()); _camStream = null; }
      _capturedBlob = null;
      document.getElementById('cam-capture-btn').disabled = false;
    }

    function capturePhoto() {
      const video = document.getElementById('cam-video');
      const canvas = document.getElementById('cam-canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      canvas.getContext('2d').drawImage(video, 0, 0);

      canvas.toBlob(blob => {
        _capturedBlob = blob;
        // Show snapshot preview
        video.style.display = 'none';
        canvas.style.display = 'block';
        document.getElementById('cam-capture-btn').style.display = 'none';
        document.getElementById('cam-retake-btn').style.display = '';
        document.getElementById('cam-use-btn').style.display = '';
      }, 'image/jpeg', 0.95);
    }

    function retakePhoto() {
      _capturedBlob = null;
      const video = document.getElementById('cam-video');
      const canvas = document.getElementById('cam-canvas');
      video.style.display = 'block';
      canvas.style.display = 'none';
      document.getElementById('cam-capture-btn').style.display = '';
      document.getElementById('cam-retake-btn').style.display = 'none';
      document.getElementById('cam-use-btn').style.display = 'none';
    }

    function usePhoto() {
      if (!_capturedBlob) return;
      const file = new File([_capturedBlob], 'camera.jpg', { type: 'image/jpeg' });
      const dt = new DataTransfer(); dt.items.add(file);
      const innerId = _camTarget === 'embed' ? 'embed-drop-inner' : 'ext-drop-inner';
      const fileId = _camTarget === 'embed' ? 'embed-file' : 'ext-file';
      document.getElementById(fileId).files = dt.files;
      previewImg(file, document.getElementById(innerId));
      closeCamera();
    }


    // ── Utils ────────────────────────────────────────────────
    async function post(url, form) {
      const res = await fetch(url, { method: 'POST', body: form });
      return res.json();
    }
    function showErr(el, msg) { el.textContent = msg; el.classList.remove('hidden'); }

