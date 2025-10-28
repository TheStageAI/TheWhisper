const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  quitApp: () => ipcRenderer.send('app-quit'),
  log: {
    info: (...args) => ipcRenderer.send('log-info', ...args),
    warn: (...args) => ipcRenderer.send('log-warn', ...args),
    error: (...args) => ipcRenderer.send('log-error', ...args),
  },
});