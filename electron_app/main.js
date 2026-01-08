const { app, BrowserWindow, ipcMain, Menu } = require('electron');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
const isDev = process.env.NODE_ENV === 'development';

const BACKEND_MODE = process.env.BACKEND_MODE || 'local';
const LOCAL_BACKEND_URL = process.env.LOCAL_BACKEND_URL || 'http://localhost:8800';
const REMOTE_BACKEND_URL = process.env.REMOTE_BACKEND_URL || 'https://api.thestage.ai';
const BASE_URL = BACKEND_MODE === 'remote' ? REMOTE_BACKEND_URL : LOCAL_BACKEND_URL;

// ASR Backend Type: "apple" (local MLX) or "whisper" (remote Triton API)
const ASR_BACKEND_TYPE = process.env.ASR_BACKEND_TYPE || 'apple';

const log = require('electron-log');
log.transports.file.level = false; // Logs off
log.transports.file.resolvePath = () =>
  path.join(app.getPath('logs'), 'speech-to-text-frontend', 'logs-' + new Date().toISOString().slice(0, 10) + '.txt');
log.info('Speech to Text Frontend starting...');

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    maxWidth: 1440,
    maxHeight: 900,
    minWidth: 500,
    minHeight: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      devTools: isDev,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  Menu.setApplicationMenu(null);
  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          `default-src 'self' ${BASE_URL};`,
          "script-src 'self' 'unsafe-inline';",
          "style-src 'self' 'unsafe-inline';",
          "img-src 'self' data: https:;",
          `connect-src 'self' ${BASE_URL};`,
          "media-src 'self' mediastream:;",
        ].join(' '),
      },
    });
  });

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

ipcMain.on('app-quit', () => {
  log.info('User requested to quit the app via UI');
  app.quit();
});

ipcMain.handle('get-config', () => {
  return {
    baseUrl: BASE_URL,
    asrBackendType: ASR_BACKEND_TYPE,
  };
});

ipcMain.on('log-info', (event, ...args) => {
  log.info(...args);
});

ipcMain.on('log-warn', (event, ...args) => {
  log.warn(...args);
});

ipcMain.on('log-error', (event, ...args) => {
  log.error(...args);
});
