/**
 * @file micAccess.js
 *
 * This file provides two functions for microphone capture:
 *   1. nodeMicAccess: For Node.js environments using the `node-record-lpcm16` library.
 *      - Input: None
 *      - Output: A microphone instance that can be used to record and then stop the recording.
 *
 *   2. webMicAccess: For browser environments using the MediaDevices API.
 *      - Input: None
 *      - Output: A Promise that resolves to a MediaStream representing the microphone audio.
 *
 * Example usage for both Node.js and the web is included below. Nothing is attached to the global scope.
 */

 const micAccess = {
  // --------------------- //
  // Node.js Microphone    //
  // --------------------- //
  nodeMicAccess: function () {
    // Load a Node audio capture library (e.g., 'node-record-lpcm16')
    const record = require('node-record-lpcm16');

    // Start recording from the system’s default microphone
    const micInstance = record.start({
      sampleRateHertz: 16000,
      verbose: false,
    });

    // Listen for data events from the microphone
    micInstance.on('data', (chunk) => {
      console.log('Mic chunk received:', chunk.length, 'bytes');
      // Process & store this chunk
    });

    // Return the micInstance so the caller can stop/cleanup later
    return micInstance;
  },

  // Example usage in Node.js:
  usageExampleNode: function () {
    const micInstance = micAccess.nodeMicAccess();

    // Stop recording after 5 seconds
    setTimeout(() => {
      micInstance.stop();
      console.log('Stopped recording from mic');
    }, 5000);
  },

  // --------------------- //
  // Web Microphone        //
  // --------------------- //
  webMicAccess: function () {
    // Request permission to access the microphone
    return navigator.mediaDevices.getUserMedia({ audio: true })
      .then((stream) => {
        console.log('Microphone access granted:', stream);
        // You can process the audio stream here, e.g., MediaRecorder or AudioContext for analysis
        return stream;
      })
      .catch((err) => {
        console.error('Mic access error:', err);
        throw err;
      });
  },
};

// Example usage in the browser:
// function usageExampleWeb() {
//   let currentStream = null;
//   const startBtn = document.getElementById('startBtn');
//   const stopBtn = document.getElementById('stopBtn');

//   startBtn.addEventListener('click', () => {
//     micAccess.webMicAccess()
//       .then((stream) => {
//         currentStream = stream;
//         console.log('Recording started...');
//       });
//   });

//   stopBtn.addEventListener('click', () => {
//     if (currentStream) {
//       currentStream.getTracks().forEach(track => track.stop());
//       console.log('Recording stopped.');
//     }
//   });
// }

