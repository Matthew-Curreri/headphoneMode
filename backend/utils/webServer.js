"use strict";

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");

/**
 * Application configuration
 */
const config = {
  host: "localhost",
  port: 3500,
  frontendDir: "/home/mcurreri/Projects/headphone-mode/frontend",
  localAudioServices: "/home/mcurreri/Projects/headphone-mode/localAudioProc",
  useHttps: false,
  httpsOptions: {
    key: null,
    cert: null
  }
};

/**
 * Return the correct content type for a given file based on its extension.
 *
 * @param {string} filePath - The file path to determine MIME type from
 * @returns {string} - The MIME/content type
 */
const getContentType = (filePath) => {
  const ext = path.extname(filePath).toLowerCase();
  const map = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml"
  };
  return map[ext] || "application/octet-stream";
};

/**
 * Read and serve a file to the client.
 *
 * @param {string} filePath - The fully resolved path to a file
 * @param {import("http").ServerResponse} res - The response object
 */
const serveFile = (filePath, res) => {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end(JSON.stringify(err));
      return;
    }
    res.setHeader("Content-Type", getContentType(filePath));
    res.writeHead(200);
    res.end(data);
  });
};

/**
 * Check if a given path references an existing file.
 *
 * @param {string} potentialPath - File path to verify
 * @returns {Promise<string|null>} - Resolves with the path if the file exists, otherwise null
 */
const checkFileExists = (potentialPath) => {
  return new Promise((resolve) => {
    fs.stat(potentialPath, (err, stats) => {
      if (!err && stats.isFile()) {
        resolve(potentialPath);
      } else {
        resolve(null);
      }
    });
  });
};

/**
 * Attempt to find the first valid file from a list of potential paths.
 *
 * @param {string[]} paths - An array of file paths to check in order
 * @returns {Promise<string|null>} - The first existing file path or null
 */
const findFirstValidFile = async (paths) => {
  for (const p of paths) {
    const result = await checkFileExists(p);
    if (result) {
      return result;
    }
  }
  return null;
};

/**
 * Return a list of potential file paths based on the incoming URL.
 * If the URL starts with "/localAudioProc", we try the localAudioServices folder first,
 * then fall back to the frontend folder.
 *
 * @param {string} requestUrl - The raw URL from the request
 * @param {object} config - The server configuration
 * @returns {string[]} - An ordered list of potential file paths
 */
const getPotentialFilePaths = (requestUrl, config) => {
  const paths = [];

  // If the request is intended for the local audio services, check that folder first
  if (requestUrl.startsWith("/localAudioProc")) {
    const strippedUrl = requestUrl.slice("/localAudioProc".length) || "/";
    paths.push(path.join(config.localAudioServices, strippedUrl));
  }

  // Always also check the frontend directory
  paths.push(path.join(config.frontendDir, requestUrl));

  return paths;
};

/**
 * Main server callback to handle each incoming request.
 * Checks all relevant paths and serves the first existing file,
 * or falls back to index.html if no files are found.
 *
 * @param {import("http").IncomingMessage} req - The incoming request
 * @param {import("http").ServerResponse} res - The outgoing response
 */
const serverCallback = async (req, res) => {
  try {
    const potentialPaths = getPotentialFilePaths(req.url, config);
    const foundFile = await findFirstValidFile(potentialPaths);

    if (foundFile) {
      // Serve the first file that exists
      return serveFile(foundFile, res);
    } else {
      // Default to serving index.html if nothing was found
      const indexPath = path.join(config.frontendDir, "index.html");
      serveFile(indexPath, res);
    }
  } catch (error) {
    console.error("Server Error:", error);
    res.writeHead(500);
    res.end("Internal Server Error");
  }
};

/**
 * Create and start the server (HTTP or HTTPS depending on config).
 */
if (config.useHttps && config.httpsOptions.key && config.httpsOptions.cert) {
  https.createServer(config.httpsOptions, serverCallback).listen(
    config.port,
    config.host,
    () => {
      console.log(`HTTPS Server running at https://${config.host}:${config.port}`);
    }
  );
} else {
  http.createServer(serverCallback).listen(config.port, config.host, () => {
    console.log(`HTTP Server running at http://${config.host}:${config.port}`);
  });
}
