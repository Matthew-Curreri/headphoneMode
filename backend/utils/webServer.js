const config = {
    host: 'localhost',
    port: 3000,
    frontendDir: '/home/mcurreri/Projects/headphone-mode/frontend',
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

    fs.stat(filePath, (err, stats) => {
        if (err || !stats.isFile()) {
            filePath = path.join(config.frontendDir, 'index.html');
        }
        
        fs.readFile(filePath, (err, content) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'text/plain' });
                return res.end('500 Internal Server Error');
            }
            
            res.writeHead(200, { 'Content-Type': getContentType(filePath) });
            res.end(content);
        });
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
