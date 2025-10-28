const { app, BrowserWindow, ipcMain, Menu } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';

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
          "default-src 'self' http://localhost:8000;",
          "script-src 'self' 'unsafe-inline';",
          "style-src 'self' 'unsafe-inline';",
          "img-src 'self' data: https:;",
          "connect-src 'self' http://localhost:8000;",
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

ipcMain.on('log-info', (event, ...args) => {
  log.info(...args);
});

ipcMain.on('log-warn', (event, ...args) => {
  log.warn(...args);
});

ipcMain.on('log-error', (event, ...args) => {
  log.error(...args);
});
