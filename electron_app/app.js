'use strict';

const app = {
  brandLink: document.querySelector('.brand-link'),
  recorder: {
    containerSiri: document.getElementById('siri-container'),
    canvasSiri: document.querySelector('#siri-canvas'),
    buttonRecorder: document.querySelector('.recorder__button'),
    panelTitle: document.querySelector('.panel__title'),
    progressBar: document.querySelector('.progress__cover-bar'),
  },
  transcription: {
    transcriptionElement: document.querySelector('.transcriptor__body'),
    copyButton: document.querySelector('.transcriptor__copyBtn'),
  },
  preloader: {
    preloaderWrapper: document.getElementById('preloader'),
    preloaderMessage: document.getElementById('preloaderMessage'),
  },
  languageSelector: {
    trigger: document.querySelector('.lang-trigger'),
    dropdown: document.querySelector('.lang-dropdown'),
    wrapper: document.querySelector('.lang-select-wrapper'),
  },
};

const appState = {
  recordingActive: false,
  sessionActive: false,
  audioInitialized: false,
  processActive: false,
  isValidToken: false,
};

const SAMPLE_RATE = 16000;
// Duration (in seconds) of each small audio chunk sent to the backend.
// At 16 kHz, 0.05 s ≈ 800 samples.
const CHUNK_DURATION_S = 0.05;
const BUFFER_SIZE = Math.round(SAMPLE_RATE * CHUNK_DURATION_S);
const FRAME_SIZE_ANALYSER = 512;
const STEP_SIZE_SEND_CHUNK = 500;
const MIN_PROCESS_INTERVAL = 100;
let SESSION_ID = null;

let audioContext = null;
let audioWorkletNode = null;
let analyser = null;
let mediaStream = null;
let siriWaveInstance = null;
let animationFrameId = null;
let pcmFrames = [];
let currentText = '';
let uncommittedText = '';
let uncommittedLength = 0;
let audioSource = null;
let processingLoopActive = false;

/////////////////////////////////////// API SERVER ////////////////////////////////////////////////
const api = {
  BASE_URL: null,

  async init() {
    const config = await window.electronAPI.getConfig();
    this.BASE_URL = config.baseUrl;
    window.electronAPI.log.info('API initialized with BASE_URL:', this.BASE_URL);
  },

  // SESSION CREATE
  async createSession() {
    while (true) {
      try {
        const response = await fetch(`${this.BASE_URL}/session/create`, {
          method: 'POST',
        });

        if (response.status === 401) {
          showErrorModal(`Error: ${response.status}`);
          return false;
        }

        if (response.status === 500) {
          showErrorModal(`Error: ${response.status}`);
          return false;
        }

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        SESSION_ID = data.session_id;
        window.electronAPI.log.info('Session created:', SESSION_ID);
        return true;
      } catch (error) {
        window.electronAPI.log.warn('Session created error:', error.message);
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
  },

  // SESSION END
  async endSession() {
    if (!SESSION_ID) return;
    try {
      await fetch(`${this.BASE_URL}/session/${SESSION_ID}/end`, {
        method: 'POST',
      });
      window.electronAPI.log.info('Session ended');
    } catch (error) {
      showErrorModal(`Error: ${error.message}`);
      window.electronAPI.log.error(`Error ending session: ${error.message}`);
      statusElement.textContent = `Status: Error ending session - ${error.message}`;
    }
  },

  // SESSION CLEAR
  async clearSession() {
    try {
      const response = await fetch(`${this.BASE_URL}/session/${SESSION_ID}/clear`, {
        method: 'POST',
      });

      window.electronAPI.log.info('Session cleared');

      if (!response.ok) {
        const errorText = await response.text();
        window.electronAPI.log.error('Clear session error details:', errorText);
      }
    } catch (error) {
      window.electronAPI.log.error('Error clearing session:', error);
    }
  },

  // SESSION ADD_CHUNK
  async sendAudioChunk(chunksFloatArray) {
    if (!SESSION_ID || !appState.recordingActive) return;
    try {
      await fetch(
        `${this.BASE_URL}/session/${SESSION_ID}/add_chunk?audio_data=${encodeURIComponent(chunksFloatArray)}`,
        {
          method: 'POST',
        }
      );
    } catch (err) {
      window.electronAPI.log.error('Error sending chunk:', err);
    }
  },

  // SESSION PROCESS START
  startProcessChunks(interval = MIN_PROCESS_INTERVAL) {
    processingLoopActive = true;
    appState.processActive = true;

    async function runLoop() {
      while (processingLoopActive && SESSION_ID && appState.recordingActive) {
        const start = Date.now();

        try {
          const response = await fetch(`${api.BASE_URL}/session/${SESSION_ID}/process`, {
            method: 'POST',
          });
          const wordsObject = await response.json();
          updateTranscription(wordsObject);
        } catch (err) {
          window.electronAPI.log.error('Process error:', err);
        }

        const elapsed = Date.now() - start;
        const delay = Math.max(0, interval - elapsed);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
    runLoop();
  },

  // SESSION PROCESS STOP
  stopProcessChunks() {
    processingLoopActive = false;
    appState.processActive = false;
  },
};

/////////////////////////////////////// AUDIO PLUGIN //////////////////////////////////////////////
async function initAudioPlugin() {
  try {
    if (appState.audioInitialized) return true;

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    audioContext = new AudioContext({
      sampleRate: SAMPLE_RATE,
      latencyHint: 'interactive',
    });

    analyser = audioContext.createAnalyser();
    analyser.fftSize = FRAME_SIZE_ANALYSER;
    analyser.smoothingTimeConstant = 0.8;

    await audioContext.audioWorklet.addModule('audio-processor.js');
    audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      channelCountMode: 'explicit',
      channelInterpretation: 'speakers',
      processorOptions: { frameSize: BUFFER_SIZE },
    });

    audioWorkletNode.port.onmessage = (e) => {
      if (e.data.type === 'audioData') {
        const floatArray = new Float32Array(e.data.data);
        const base64Audio = btoa(String.fromCharCode(...new Uint8Array(floatArray.buffer)));
        api.sendAudioChunk(base64Audio);
      }
    };
    await audioContext.suspend();
    appState.audioInitialized = true;
    window.electronAPI.log.info('Audio plugin initialized!');
    return true;
  } catch (error) {
    window.electronAPI.log.error('Audio initialization error:', error);
    removeAudioPlugin();
    return false;
  }
}

function removeAudioPlugin() {
  audioSource?.disconnect();
  audioSource = null;
  audioWorkletNode?.disconnect();
  analyser?.disconnect();
  cancelAnimationFrame(animationFrameId);
  api.stopPolling();
  if (audioContext?.state !== 'closed') {
    audioContext?.close();
  }
  mediaStream?.getTracks().forEach((t) => t.stop());
  pcmFrames = [];
  currentText = '';
  uncommittedText = '';

  window.electronAPI.log.info('Audio plugin cleaned up');
}

/////////////////////////////////////// SIRI //////////////////////////////////////////////////////
function initSiriWave(container) {
  if (siriWaveInstance) siriWaveInstance.stop();
  container.innerHTML = '';
  const containerWidth = container.offsetWidth;
  siriWaveInstance = new SiriWave({
    container: container,
    width: containerWidth,
    height: 350,
    style: 'ios9',
    curveDefinition: [
      { attenuation: -2, lineWidth: 1, opacity: 0.1 },
      { attenuation: -6, lineWidth: 1, opacity: 0.2 },
      { attenuation: 4, lineWidth: 1, opacity: 0.4 },
      { attenuation: 2, lineWidth: 1, opacity: 0.6 },
      { attenuation: 1, lineWidth: 1.5, opacity: 1 },
    ],
  });
  siriWaveInstance.setAmplitude(0);
  siriWaveInstance.draw();
}

function updateWaveform() {
  if (!analyser || !siriWaveInstance) return;
  const dataArray = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(dataArray);
  const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
  siriWaveInstance.setAmplitude(average / 128);
  animationFrameId = requestAnimationFrame(updateWaveform);
}

/////////////////////////////////////// TRANSCRIPTION /////////////////////////////////////////////
function updateTranscription(wordsObject) {
  const transcriptionElement = app.transcription.transcriptionElement;
  if (!wordsObject || (wordsObject.words.length === 0 && wordsObject.uncommited_words.length === 0)) {
    return;
  }

  if (wordsObject.words && wordsObject.words.length > 0) {
    uncommittedText = '';
    const wordsText = wordsObject.words.map(word => word.text || word.word || word.value || '').filter(Boolean).join(' ');
    currentText = currentText
      ? `${currentText} ${wordsText}`
      : wordsText;
  }
  const uncommittedWordsText = wordsObject.uncommited_words.map(word => word.text || word.word || word.value || '').filter(Boolean).join(' ');
  uncommittedText = uncommittedWordsText;

  requestAnimationFrame(() => {
    transcriptionElement.innerHTML = '';

    if (currentText) {
      const committedSpan = document.createElement('span');
      committedSpan.className = 'committed';
      committedSpan.textContent = currentText;
      transcriptionElement.appendChild(committedSpan);
    }

    if (uncommittedText) {
      const uncommittedSpan = document.createElement('span');
      uncommittedSpan.className = 'uncommitted';
      uncommittedSpan.style.opacity = '0.7';
      uncommittedSpan.textContent = uncommittedText;
      transcriptionElement.appendChild(uncommittedSpan);
    }
    transcriptionElement.scrollTop = transcriptionElement.scrollHeight;
  });
}

/////////////////////////////////////// TOGGLE BUTTON /////////////////////////////////////////////
async function toggleButton() {
  const isStart = !appState.recordingActive;
  appState.recordingActive = isStart;
  app.recorder.buttonRecorder.classList.toggle('is-recording', isStart);
  app.recorder.progressBar.classList.toggle('is-recording', isStart);
  app.recorder.panelTitle.textContent = isStart ? 'Recording...' : 'Paused...';
  app.recorder.canvasSiri.style.opacity = isStart ? 1 : 0;
  if (isStart) {
    if (!appState.sessionActive) {
      appState.sessionActive = await api.createSession();
      if (!appState.sessionActive) return;
    }
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }
    audioSource = audioContext.createMediaStreamSource(mediaStream);
    audioSource.connect(analyser);
    audioSource.connect(audioWorkletNode);
    siriWaveInstance.start();
    updateWaveform();
    api.startProcessChunks(MIN_PROCESS_INTERVAL);
    clearText();
  } else {
    audioContext.suspend();
    await api.clearSession();
    await api.endSession();
    appState.sessionActive = false;
    appState.recordingActive = false;
    api.stopProcessChunks();
    appState.processActive = false;
  }
}

/////////////////////////////////////// UI ELEMENTS ///////////////////////////////////////////////
function progressBarAnimation(isRecording = false) {
  const progressBar = app.recorder.progressBar;
  if (!progressBar) return;

  if (appState.recordingActive) {
    progressBar.classList.add('is-recording');
  } else {
    progressBar.classList.remove('is-recording');
  }
}

function copyText() {
  const textElement = app.transcription.transcriptionElement;
  const copyButton = app.transcription.copyButton;

  if (!textElement || !copyButton) {
    window.electronAPI.log.error('Text or copy button elements not found');
    return;
  }

  const text = textElement.textContent;

  copyButton.classList.add('is-copied');
  copyButton.innerHTML = `
    <svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  `;

  navigator.clipboard
    .writeText(text)
    .then(() => {
      setTimeout(() => {
        copyButton.classList.remove('is-copied');
        copyButton.innerHTML = `
          <svg width="100%" height="100%" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M7 7V4.2002C7 3.08009 7 2.51962 7.21799 2.0918C7.40973 1.71547 7.71547 1.40973 8.0918 1.21799C8.51962 1 9.08009 1 10.2002 1H15.8002C16.9203 1 17.4801 1 17.9079 1.21799C18.2842 1.40973 18.5905 1.71547 18.7822 2.0918C19.0002 2.51962 19.0002 3.07967 19.0002 4.19978V9.7998C19.0002 10.9199 19.0002 11.48 18.7822 11.9078C18.5905 12.2841 18.2839 12.5905 17.9076 12.7822C17.4802 13 16.921 13 15.8031 13H13M7 7H4.2002C3.08009 7 2.51962 7 2.0918 7.21799C1.71547 7.40973 1.40973 7.71547 1.21799 8.0918C1 8.51962 1 9.08009 1 10.2002V15.8002C1 16.9203 1 17.4801 1.21799 17.9079C1.40973 18.2842 1.71547 18.5905 2.0918 18.7822C2.5192 19 3.07899 19 4.19691 19H9.80355C10.9215 19 11.4805 19 11.9079 18.7822C12.2842 18.5905 12.5905 18.2839 12.7822 17.9076C13 17.4802 13 16.921 13 15.8031V13M7 7H9.8002C10.9203 7 11.4801 7 11.9079 7.21799C12.2842 7.40973 12.5905 7.71547 12.7822 8.0918C13 8.51921 13 9.079 13 10.1969L13 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        `;
      }, 2000);
    })
    .catch((err) => {
      window.electronAPI.log.error('Error:', err);

      copyButton.classList.remove('is-copied');
      copyButton.innerHTML = `
        <svg width="100%" height="100%" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M7 7V4.2002C7 3.08009 7 2.51962 7.21799 2.0918C7.40973 1.71547 7.71547 1.40973 8.0918 1.21799C8.51962 1 9.08009 1 10.2002 1H15.8002C16.9203 1 17.4801 1 17.9079 1.21799C18.2842 1.40973 18.5905 1.71547 18.7822 2.0918C19.0002 2.51962 19.0002 3.07967 19.0002 4.19978V9.7998C19.0002 10.9199 19.0002 11.48 18.7822 11.9078C18.5905 12.2841 18.2839 12.5905 17.9076 12.7822C17.4802 13 16.921 13 15.8031 13H13M7 7H4.2002C3.08009 7 2.51962 7 2.0918 7.21799C1.71547 7.40973 1.40973 7.71547 1.21799 8.0918C1 8.51962 1 9.08009 1 10.2002V15.8002C1 16.9203 1 17.4801 1.21799 17.9079C1.40973 18.2842 1.71547 18.5905 2.0918 18.7822C2.5192 19 3.07899 19 4.19691 19H9.80355C10.9215 19 11.4805 19 11.9079 18.7822C12.2842 18.5905 12.5905 18.2839 12.7822 17.9076C13 17.4802 13 16.921 13 15.8031V13M7 7H9.8002C10.9203 7 11.4801 7 11.9079 7.21799C12.2842 7.40973 12.5905 7.71547 12.7822 8.0918C13 8.51921 13 9.079 13 10.1969L13 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      `;
    });
}

function clearText() {
  const textElement = app.transcription.transcriptionElement;
  if (textElement) {
    textElement.textContent = '';
    currentText = '';
    uncommittedText = '';
    uncommittedLength = 0;
    window.electronAPI.log.info('Text cleared');
  } else {
    window.electronAPI.log.error('Transcriptor element not found');
  }
}

function langSelectButton() {
  const trigger = app.languageSelector.trigger;
  const dropdown = app.languageSelector.dropdown;
  const wrapper = app.languageSelector.wrapper;

  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('lang-dropdown_hidden');
  });

  dropdown.addEventListener('click', (e) => {
    const target = e.target;

    const li = target.closest('li');
    if (!li || li.classList.contains('disabled')) return;

    document.querySelectorAll('.lang-dropdown li').forEach((el) => {
      el.classList.remove('active');
    });
    li.classList.add('active');

    trigger.textContent = li.textContent.slice(0, 2);
    dropdown.classList.add('lang-dropdown_hidden');
  });

  document.addEventListener('click', (e) => {
    if (!wrapper.contains(e.target)) {
      dropdown.classList.add('lang-dropdown_hidden');
    }
  });
}

function preloaderOpen() {
  const preloader = app.preloader.preloaderWrapper;
  const messageElement = app.preloader.preloaderMessage;
  preloader.classList.add('preloader_open');
  messageElement.textContent = 'Setting everything up for you...';
  setTimeout(() => {
    messageElement.textContent = 'Initializing speech recognition model...';
  }, 4000);
  setTimeout(() => {
    messageElement.textContent = 'Finishing setup...';
  }, 8000);
}

function preloaderClose() {
  const preloader = app.preloader.preloaderWrapper;
  preloader.classList.remove('preloader_open');
}

function showErrorModal(message) {
  const modal = document.querySelector('.error-wrapper');
  const text = modal.querySelector('.text-error-caption');
  text.textContent = message;
  modal.classList.add('error-wrapper_open');
}

/////////////////////////////////////// INITIALIZATION APP ////////////////////////////////////////
async function initializeApp() {
  preloaderOpen();
  await api.init();
  const initialized = await api.createSession();
  if (!initialized) {
    window.electronAPI.log.error('Failed to initialize application');
    appState.sessionActive = false;
    preloaderClose();
    return;
  }
  preloaderClose();
  appState.sessionActive = true;
  await initAudioPlugin();
  initSiriWave(app.recorder.canvasSiri);
  siriWaveInstance.stop();
}

document.addEventListener('DOMContentLoaded', () => {
  initializeApp();
  langSelectButton();

  document.querySelector('.error-button')?.addEventListener('click', () => {
    if (window.electronAPI?.quitApp) {
      window.electronAPI.quitApp();
    } else {
      window.electronAPI.log.warn('quitApp not available');
    }
  });
});

window.addEventListener('beforeunload', (event) => {
  removeAudioPlugin();
});
