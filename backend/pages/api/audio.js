// audio.js - Handles incoming audio processing

const fs = require('fs')
const path = require('path')
const config = require(path.join(process.cwd(), 'backend/config.js'))
const formidable = require('formidable')
const { exec } = require('child_process')
const fetch = require('node-fetch')
const FormData = require('form-data')
const keywordScan = require(path.join(
  process.cwd(),
  'localAudioProc/keywordScan.js'
))
const micAccess = require(path.join(
  process.cwd(),
  'localAudioProc/micAccess.js'
))

function handler (req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' })
  }

  const form = new formidable.IncomingForm()
  form.uploadDir = path.join(process.cwd(), 'uploads')
  form.keepExtensions = true

  form.parse(req, (err, fields, files) => {
    if (err) {
      return res.status(500).json({ error: 'File upload failed' })
    }

    const audioFile = files.audio?.filepath
    if (!audioFile) {
      return res.status(400).json({ error: 'No audio file received' })
    }

    // Use 'conda run -n whisper' to ensure we are in the correct conda env
    // Also use the --model turbo flag by default
    const transcriptionCommand = `conda run -n whisper whisper "${audioFile}" --model turbo --output_format json`

    exec(transcriptionCommand, (error, stdout) => {
      if (error) {
        console.error('Error during Whisper transcription:', error)
        return res.status(500).json({ error: 'Speech-to-text failed' })
      }
      res.status(200).json({ transcript: stdout })
    })
  })
}

module.exports = handler

function testAudioAPI () {
  // Provide a path to a test WAV file that exists
  const testAudioPath = 'test-audio.wav'
  const formData = new FormData()
  formData.append('audio', fs.createReadStream(testAudioPath))

  fetch(`${config.baseUri}:${config.port}/api/audio`, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => console.log('Test Audio API Response:', data))
    .catch(error => console.error('Test failed:', error))
}

testAudioAPI()
