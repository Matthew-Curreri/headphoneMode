const config = {
    host: 'localhost',
    port: 3500,
    frontendDir: '/home/mcurreri/Projects/headphone-mode/frontend',
    localAudioServices: '/home/mcurreri/Projects/headphone-mode/localAudioProc',
    useHttps: false,
    httpsOptions: {
        key: null,
        cert: null
    }
};

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

const serverCallback = (req, res) => {
    let filePath = path.join(config.frontendDir, req.url);
    let filePath2 = path.join(config.localAudioProc, req.url);
  
    const checkFile = (filePath) => {
      return new Promise((resolve, reject) => {
        fs.stat(filePath, (err, stats) => {
          if (!err && stats.isFile()) resolve(filePath);
          else resolve(null); // Resolve with null if the file doesn't exist
        });
      });
    };
  
    Promise.all([checkFile(filePath), checkFile(filePath2)]).then((results) => {
      for (let i = 0; i < results.length; i++) {
        if (results[i]) {
          serveFile(results[i], res);
          return;
        }
      }
      // If no files are found, default to serving index.html
      serveFile(path.join(config.frontendDir, 'index.html'), res);
    });
  };

function getContentType(filePath) {
    const ext = path.extname(filePath);
    const map = {
        '.html': 'text/html',
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml'
    };
    return map[ext] || 'application/octet-stream';
}

if (config.useHttps && config.httpsOptions.key && config.httpsOptions.cert) {
    https.createServer(config.httpsOptions, serverCallback).listen(config.port, config.host, () => 
        console.log(`HTTPS Server running at https://${config.host}:${config.port}`)
    );
} else {
    http.createServer(serverCallback).listen(config.port, config.host, () => 
        console.log(`HTTP Server running at http://${config.host}:${config.port}`)
    );
}
