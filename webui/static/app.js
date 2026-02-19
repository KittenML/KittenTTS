// KittenTTS WebUI - Cute kitten-themed interactions 🐱

const models = [
    { id: 'kitten-tts-nano', name: 'Nano (FP32)', params: '15M', size: '56MB', precision: 'FP32', quality: 'best', description: '⭐ Best quality - Full 32-bit precision', emoji: '⭐' },
    { id: 'kitten-tts-mini', name: 'Mini (INT8)', params: '80M', size: '80MB', precision: 'INT8', quality: 'good', description: 'Largest model, INT8 quantized', emoji: '🐱' },
    { id: 'kitten-tts-micro', name: 'Micro (INT8)', params: '40M', size: '41MB', precision: 'INT8', quality: 'good', description: 'Balanced size, INT8 quantized', emoji: '🐈' },
    { id: 'kitten-tts-nano-int8', name: 'Nano (INT8)', params: '15M', size: '19MB', precision: 'INT8', quality: 'basic', description: 'Smallest, INT8 quantized', emoji: '💫' }
];

const voices = [
    { id: 'Bella', name: 'Bella', gender: 'female', description: 'Warm & gentle', emoji: '👩' },
    { id: 'Jasper', name: 'Jasper', gender: 'male', description: 'Clear & professional', emoji: '👨' },
    { id: 'Luna', name: 'Luna', gender: 'female', description: 'Soft & melodic', emoji: '🌙' },
    { id: 'Bruno', name: 'Bruno', gender: 'male', description: 'Deep & resonant', emoji: '🐻' },
    { id: 'Rosie', name: 'Rosie', gender: 'female', description: 'Bright & cheerful', emoji: '🌸' },
    { id: 'Hugo', name: 'Hugo', gender: 'male', description: 'Confident & steady', emoji: '💼' },
    { id: 'Kiki', name: 'Kiki', gender: 'female', description: 'Playful & energetic', emoji: '🎀' },
    { id: 'Leo', name: 'Leo', gender: 'male', description: 'Friendly & warm', emoji: '🦁' }
];

const sampleTexts = [
    "Hello! I'm KittenTTS, your cute and lightweight text-to-speech companion! 🐱",
    "The quick brown fox jumps over the lazy dog. Meow! 🐾",
    "Welcome to KittenTTS! For kittens, by kittens. 💕",
    "Did you know? KittenTTS can run entirely on your CPU without a GPU! ✨",
    "Purrr-fect speech synthesis at your fingertips! 🎙️"
];

let audioContext = null;
let meowSounds = [];

function init() {
    initializeTheme();
    populateModels();
    populateVoices();
    setupEventListeners();
    addFloatingPaws();

    // Randomly select a sample text
    const textarea = document.getElementById('textInput');
    if (textarea && !textarea.value) {
        textarea.placeholder = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];
    }

    // Load initial stats
    loadDebugStats();
}

// Add floating paw decorations
function addFloatingPaws() {
    const container = document.querySelector('.container');
    for (let i = 0; i < 5; i++) {
        const paw = document.createElement('div');
        paw.className = 'paw-bg';
        paw.innerHTML = '🐾';
        paw.style.left = `${Math.random() * 90}%`;
        paw.style.top = `${Math.random() * 90}%`;
        paw.style.animationDelay = `${Math.random() * -20}s`;
        paw.style.fontSize = `${6 + Math.random() * 6}rem`;
        document.body.insertBefore(paw, container);
    }
}

function initializeTheme() {
    const savedTheme = localStorage.getItem('kitten-tts-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = savedTheme || (prefersDark ? 'dark' : 'light');
    setTheme(theme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('kitten-tts-theme', theme);
    updateThemeIcon(theme);
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggle');
    btn.innerHTML = theme === 'dark' ? '☀️' : '🌙';
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    setTheme(current === 'dark' ? 'light' : 'dark');
}

function populateModels() {
    const select = document.getElementById('modelSelect');
    select.innerHTML = models.map(m => {
        const qualityIcon = m.quality === 'best' ? '⭐' : m.quality === 'good' ? '✓' : '○';
        return `<option value="${m.id}">${qualityIcon} ${m.name} (${m.size}) — ${m.description}</option>`;
    }).join('');

    // Add change handler to update info display
    select.addEventListener('change', updateModelInfo);
    updateModelInfo();
}

function updateModelInfo() {
    const select = document.getElementById('modelSelect');
    const model = models.find(m => m.id === select.value);
    const infoEl = document.getElementById('modelInfo');
    if (infoEl && model) {
        const qualityBadge = model.quality === 'best' ? '⭐ Best Quality' :
            model.quality === 'good' ? '✓ Good' : '○ Basic';
        infoEl.textContent = `${model.params} params • ${model.size} • ${model.precision} precision • ${qualityBadge}`;
    }
}

function populateVoices() {
    const select = document.getElementById('voiceSelect');
    select.innerHTML = voices.map(v =>
        `<option value="${v.id}">${v.emoji} ${v.name} — ${v.description}</option>`
    ).join('');

    select.addEventListener('change', updateVoiceInfo);
    updateVoiceInfo();
}

function updateVoiceInfo() {
    const select = document.getElementById('voiceSelect');
    const voice = voices.find(v => v.id === select.value);
    const infoEl = document.getElementById('voiceInfo');
    if (infoEl && voice) {
        const genderEmoji = voice.gender === 'female' ? '♀️' : '♂️';
        infoEl.textContent = `${genderEmoji} ${voice.gender} voice • ${voice.description}`;
    }
}

function setupEventListeners() {
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
    document.getElementById('speedSlider').addEventListener('input', updateSpeedDisplay);
    document.getElementById('generateBtn').addEventListener('click', handleGenerate);

    // Debug panel toggle
    const debugToggle = document.getElementById('debugToggle');
    if (debugToggle) {
        debugToggle.addEventListener('click', toggleDebugPanel);
    }

    // Debug refresh button
    const refreshBtn = document.getElementById('refreshStatsBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadDebugStats);
    }
}

function updateSpeedDisplay() {
    const slider = document.getElementById('speedSlider');
    const value = slider.value;
    document.getElementById('speedValue').childNodes[0].textContent = `${value}x `;

    // Change emoji based on speed
    let emoji = '🐱';
    if (value < 0.8) emoji = '🐢'; // Slow
    else if (value > 1.5) emoji = '⚡'; // Fast
    else if (value > 1.2) emoji = '🐇'; // Quick

    const emojiEl = document.getElementById('speedEmoji');
    if (emojiEl) emojiEl.textContent = emoji;
}

function showLoading(show) {
    const btn = document.getElementById('generateBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('btnSpinner');

    btn.disabled = show;
    btnText.style.display = show ? 'none' : 'inline';
    spinner.style.display = show ? 'inline-block' : 'none';

    if (show) {
        btn.classList.add('generating');
    } else {
        btn.classList.remove('generating');
    }
}

function showError(message) {
    const el = document.getElementById('errorMessage');
    el.textContent = '😿 ' + message;
    el.classList.add('visible');
    setTimeout(() => el.classList.remove('visible'), 5000);
}

function showOutput(audioBase64, duration) {
    const section = document.getElementById('outputSection');
    const audio = document.getElementById('audioPlayer');
    const durationEl = document.getElementById('audioDuration');
    const downloadBtn = document.getElementById('downloadBtn');

    audio.src = `data:audio/wav;base64,${audioBase64}`;
    durationEl.textContent = `⏱️ Duration: ${duration}s`;

    const blob = base64ToBlob(audioBase64, 'audio/wav');
    const url = URL.createObjectURL(blob);
    downloadBtn.href = url;
    downloadBtn.download = `kitten-tts-${Date.now()}.wav`;

    section.classList.add('visible');
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Auto-play the audio
    audio.play().catch(() => {
        // Auto-play blocked, user will need to click
    });
}

function base64ToBlob(base64, mimeType) {
    const byteChars = atob(base64);
    const byteNumbers = new Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++) {
        byteNumbers[i] = byteChars.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

async function handleGenerate() {
    const text = document.getElementById('textInput').value.trim();
    const model = document.getElementById('modelSelect').value;
    const voice = document.getElementById('voiceSelect').value;
    const speed = parseFloat(document.getElementById('speedSlider').value);

    if (!text) {
        showError('Please enter some text to generate speech! 🐾');
        return;
    }

    showLoading(true);
    document.getElementById('errorMessage').classList.remove('visible');

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, model, voice, speed })
        });

        const data = await response.json();
        console.log('Generation response:', data);

        if (!response.ok) {
            throw new Error(data.detail || 'Generation failed 😿');
        }

        showOutput(data.audio_base64, data.duration);

        // Update debug stats if available
        if (data.debug_info) {
            console.log('Debug info received:', data.debug_info);
            updateCurrentStats(data.debug_info);
        } else {
            console.log('No debug_info in response');
        }

        // Refresh session stats
        await loadDebugStats();

        // Success animation on button
        const btn = document.getElementById('generateBtn');
        btn.style.transform = 'scale(0.98)';
        setTimeout(() => btn.style.transform = '', 200);

    } catch (error) {
        showError(error.message || 'An error occurred during generation 😿');
        console.error('Generation error:', error);
    } finally {
        showLoading(false);
    }
}

// Debug Panel Functions
function updateCurrentStats(debugInfo) {
    console.log('Updating current stats:', debugInfo);
    const modelLoadEl = document.getElementById('statModelLoad');
    const genTimeEl = document.getElementById('statGenTime');
    const totalTimeEl = document.getElementById('statTotalTime');
    const rtfEl = document.getElementById('statRTF');

    if (modelLoadEl) modelLoadEl.textContent = debugInfo.model_load_time ? `${debugInfo.model_load_time}s` : '-';
    if (genTimeEl) genTimeEl.textContent = debugInfo.generation_time ? `${debugInfo.generation_time}s` : '-';
    if (totalTimeEl) totalTimeEl.textContent = debugInfo.total_time ? `${debugInfo.total_time}s` : '-';
    if (rtfEl) rtfEl.textContent = debugInfo.real_time_factor || '-';
}

async function loadDebugStats() {
    try {
        console.log('Loading debug stats...');
        const response = await fetch('/api/stats');
        if (!response.ok) {
            console.error('Stats fetch failed:', response.status);
            return;
        }

        const data = await response.json();
        console.log('Stats received:', data);

        // Update session stats
        if (data.generation_stats) {
            const sessionRequests = document.getElementById('sessionRequests');
            const sessionAvgGen = document.getElementById('sessionAvgGen');
            const sessionAvgRTF = document.getElementById('sessionAvgRTF');
            const sessionAudio = document.getElementById('sessionAudio');

            if (sessionRequests) sessionRequests.textContent = data.generation_stats.total_requests || 0;
            if (sessionAvgGen) sessionAvgGen.textContent = `${data.generation_stats.avg_generation_time || 0}s`;
            if (sessionAvgRTF) sessionAvgRTF.textContent = data.generation_stats.avg_rtf || 0;
            if (sessionAudio) sessionAudio.textContent = `${data.generation_stats.total_audio_generated || 0}s`;
        }

        // Update system info
        if (data.system) {
            const systemInfoEl = document.getElementById('systemInfo');
            if (systemInfoEl) {
                systemInfoEl.innerHTML = `
                    <div class="debug-info-item">
                        <span class="debug-info-key">Cache Directory</span>
                        <span class="debug-info-value">${data.system.cache_directory || 'N/A'}</span>
                    </div>
                    <div class="debug-info-item">
                        <span class="debug-info-key">Cache Size</span>
                        <span class="debug-info-value">${data.system.cache_size_mb || 0} MB</span>
                    </div>
                    <div class="debug-info-item">
                        <span class="debug-info-key">Memory Usage</span>
                        <span class="debug-info-value">${data.system.memory_usage_mb || 'N/A'} MB</span>
                    </div>
                    <div class="debug-info-item">
                        <span class="debug-info-key">Python Version</span>
                        <span class="debug-info-value">${data.system.python_version || 'N/A'}</span>
                    </div>
                    <div class="debug-info-item">
                        <span class="debug-info-key">Loaded Models</span>
                        <span class="debug-info-value">${(data.system.loaded_models || []).join(', ') || 'None'}</span>
                    </div>
                `;
            }
        }

        // Update recent requests
        const requestsEl = document.getElementById('recentRequests');
        if (requestsEl && data.generation_stats && data.generation_stats.recent_requests) {
            if (data.generation_stats.recent_requests.length === 0) {
                requestsEl.innerHTML = '<div class="debug-request-item">No recent generations</div>';
            } else {
                requestsEl.innerHTML = data.generation_stats.recent_requests.map(req => `
                    <div class="debug-request-item">
                        <span class="debug-request-model">${req.model || '?'}</span>
                        <span class="debug-request-voice">${req.voice || '?'}</span>
                        <span class="debug-request-time">${req.generation_time || 0}s</span>
                        <span class="debug-request-rtf">RTF: ${req.rtf || 0}</span>
                    </div>
                `).join('');
            }
        }
        console.log('Debug stats updated successfully');
    } catch (error) {
        console.error('Failed to load debug stats:', error);
    }
}

function toggleDebugPanel() {
    const panel = document.getElementById('debugPanel');
    const content = document.getElementById('debugContent');

    console.log('Toggling debug panel');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        panel.classList.add('expanded');
        loadDebugStats();
    } else {
        content.style.display = 'none';
        panel.classList.remove('expanded');
    }
}

// Easter egg: Meow on logo click
document.addEventListener('DOMContentLoaded', () => {
    init();

    const logoIcon = document.querySelector('.logo-icon');
    if (logoIcon) {
        logoIcon.addEventListener('click', () => {
            // Visual feedback
            logoIcon.style.transform = 'scale(1.1) rotate(-10deg)';
            setTimeout(() => {
                logoIcon.style.transform = '';
            }, 300);

            // Create a cute popup
            const popup = document.createElement('div');
            popup.textContent = 'Meow! 🐱';
            popup.style.cssText = `
                position: fixed;
                top: 100px;
                left: 50%;
                transform: translateX(-50%);
                background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 9999px;
                font-weight: 600;
                font-size: 1rem;
                box-shadow: 0 4px 12px rgba(232, 146, 160, 0.4);
                z-index: 1000;
                animation: popIn 0.3s ease, fadeOut 0.3s ease 1s forwards;
                pointer-events: none;
            `;
            document.body.appendChild(popup);
            setTimeout(() => popup.remove(), 1500);
        });
    }
});

// Add popIn animation
const style = document.createElement('style');
style.textContent = `
    @keyframes popIn {
        from { opacity: 0; transform: translateX(-50%) scale(0.8) translateY(10px); }
        to { opacity: 1; transform: translateX(-50%) scale(1) translateY(0); }
    }
    @keyframes fadeOut {
        from { opacity: 1; transform: translateX(-50%) scale(1); }
        to { opacity: 0; transform: translateX(-50%) scale(0.8) translateY(-10px); }
    }
`;
document.head.appendChild(style);
