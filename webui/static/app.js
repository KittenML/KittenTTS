const models = [
    { id: 'kitten-tts-mini', name: 'Mini', params: '80M', size: '80MB', description: 'Highest quality, larger model' },
    { id: 'kitten-tts-micro', name: 'Micro', params: '40M', size: '41MB', description: 'Balanced quality & speed' },
    { id: 'kitten-tts-nano', name: 'Nano', params: '15M', size: '56MB', description: 'Lightweight & fast' },
    { id: 'kitten-tts-nano-int8', name: 'Nano (INT8)', params: '15M', size: '19MB', description: 'Smallest, quantized' }
];

const voices = [
    { id: 'Bella', name: 'Bella', gender: 'female', description: 'Warm & gentle' },
    { id: 'Jasper', name: 'Jasper', gender: 'male', description: 'Clear & professional' },
    { id: 'Luna', name: 'Luna', gender: 'female', description: 'Soft & melodic' },
    { id: 'Bruno', name: 'Bruno', gender: 'male', description: 'Deep & resonant' },
    { id: 'Rosie', name: 'Rosie', gender: 'female', description: 'Bright & cheerful' },
    { id: 'Hugo', name: 'Hugo', gender: 'male', description: 'Confident & steady' },
    { id: 'Kiki', name: 'Kiki', gender: 'female', description: 'Playful & energetic' },
    { id: 'Leo', name: 'Leo', gender: 'male', description: 'Friendly & warm' }
];

const pawIcon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 10c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm-6 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm12 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm-6 6c-2.2 0-4 1.8-4 4h8c0-2.2-1.8-4-4-4zm-6-10c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm12 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>`;

function init() {
    initializeTheme();
    populateModels();
    populateVoices();
    setupEventListeners();
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
    select.innerHTML = models.map(m => 
        `<option value="${m.id}">${m.name} (${m.size}) - ${m.description}</option>`
    ).join('');
}

function populateVoices() {
    const select = document.getElementById('voiceSelect');
    select.innerHTML = voices.map(v => 
        `<option value="${v.id}">${v.name} (${v.gender === 'female' ? '♀' : '♂'}) - ${v.description}</option>`
    ).join('');
}

function setupEventListeners() {
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
    document.getElementById('speedSlider').addEventListener('input', updateSpeedDisplay);
    document.getElementById('generateBtn').addEventListener('click', handleGenerate);
}

function updateSpeedDisplay() {
    const slider = document.getElementById('speedSlider');
    document.getElementById('speedValue').textContent = `${slider.value}x`;
}

function showLoading(show) {
    const btn = document.getElementById('generateBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('btnSpinner');
    
    btn.disabled = show;
    btnText.style.display = show ? 'none' : 'inline';
    spinner.style.display = show ? 'inline-block' : 'none';
}

function showError(message) {
    const el = document.getElementById('errorMessage');
    el.textContent = message;
    el.classList.add('visible');
    setTimeout(() => el.classList.remove('visible'), 5000);
}

function showOutput(audioBase64, duration) {
    const section = document.getElementById('outputSection');
    const audio = document.getElementById('audioPlayer');
    const durationEl = document.getElementById('audioDuration');
    const downloadBtn = document.getElementById('downloadBtn');
    
    audio.src = `data:audio/wav;base64,${audioBase64}`;
    durationEl.textContent = `Duration: ${duration}s`;
    
    const blob = base64ToBlob(audioBase64, 'audio/wav');
    const url = URL.createObjectURL(blob);
    downloadBtn.href = url;
    downloadBtn.download = `kitten-tts-${Date.now()}.wav`;
    
    section.classList.add('visible');
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
        showError('Please enter some text to generate speech.');
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
        
        if (!response.ok) {
            throw new Error(data.detail || 'Generation failed');
        }
        
        showOutput(data.audio_base64, data.duration);
    } catch (error) {
        showError(error.message || 'An error occurred during generation.');
        console.error('Generation error:', error);
    } finally {
        showLoading(false);
    }
}

document.addEventListener('DOMContentLoaded', init);
