let currentStream = null
let mediaRecorder = null
let audioChunks = []
let audioElement = null
let micSelect = null

function initUI () {
  const contentDiv = document.getElementById('main-content')
  const startBtn = document.createElement('button')
  const stopBtn = document.createElement('button')
  const br = document.createElement('br')
  const recordingIndicator = document.createElement('div')
  const controlsElm = document.getElementById('controls')
  startBtn.classList.add('button', 'start-btn')
  stopBtn.classList.add('button', 'stop-btn')
  br.classList.add('break')
  recordingIndicator.classList.add('recordingIndicator')
  recordingIndicator.classList.add('recordingIndicator')
  //recordingIndicator.classList.add('breathing-fast');
  recordingIndicator.classList.add('spinning')
  //recordingIndicator.classList.add('cloud');

  startBtn.textContent = 'Start'
  startBtn.id = 'startBtn'
  stopBtn.textContent = 'Stop'
  micSelect = document.createElement('select')
  micSelect.id = 'micSelect'

  audioElement = document.createElement('audio')
  audioElement.controls = true

  controlsElm.appendChild(audioElement)
  controlsElm.appendChild(br)
  controlsElm.appendChild(micSelect)
  controlsElm.appendChild(startBtn)
  controlsElm.appendChild(stopBtn)
  document.body.appendChild(recordingIndicator)

  startBtn.addEventListener('click', startRecording)
  stopBtn.addEventListener('click', () => stopRecording(true))

  updateMicList()
  setTimeout(updateMicList, 200)
  setInterval(updateMicList, 2000)
}

function updateMicList () {
  navigator.mediaDevices.enumerateDevices().then(devices => {
    const mics = devices.filter(device => device.kind === 'audioinput')
    const currentOptions = Array.from(micSelect.options).map(
      option => option.value
    )
    const newOptions = mics.map(mic => mic.deviceId)

    if (JSON.stringify(currentOptions) === JSON.stringify(newOptions)) return

    const selectedValue = micSelect.value
    micSelect.innerHTML = ''
    mics.forEach((mic, index) => {
      const option = document.createElement('option')
      option.value = mic.deviceId
      option.textContent = mic.label || `Microphone ${index + 1}`
      micSelect.appendChild(option)
    })

    if (selectedValue && newOptions.includes(selectedValue))
      micSelect.value = selectedValue
  })
}

function startRecording () {
  micAccess.webMicAccess(micSelect.value).then(stream => {
    currentStream = stream
    mediaRecorder = new MediaRecorder(stream)
    audioChunks = []

    mediaRecorder.ondataavailable = event => audioChunks.push(event.data)
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
      const fileName = `recording_${Date.now()}.webm`
      audioElement.src = URL.createObjectURL(audioBlob)
      saveRecordingToIndexedDB(fileName, audioBlob, currentStream.id)
    }

    mediaRecorder.start()
    console.log('Recording started...')
  })
}

function stopRecording (saveToLocal = true) {
  if (mediaRecorder) {
    mediaRecorder.stop()
    currentStream.getTracks().forEach(track => track.stop())
    console.log('Recording stopped.')
  }
}

function saveRecordingToIndexedDB (fileName, audioBlob, streamId) {
  const request = indexedDB.open('AudioDatabase', 1)

  request.onupgradeneeded = event => {
    const db = event.target.result
    if (!db.objectStoreNames.contains('recordings')) {
      const objectStore = db.createObjectStore('recordings', {
        keyPath: 'id',
        autoIncrement: true
      })
      objectStore.createIndex('fileName', 'fileName', { unique: true })
      objectStore.createIndex('streamId', 'streamId', { unique: false })
    }
  }

  request.onsuccess = event => {
    const db = event.target.result

    // **Check if the object store exists before using it**
    if (!db.objectStoreNames.contains('recordings')) {
      console.error(
        "Object store 'recordings' is missing. Try clearing the database and reloading the page."
      )
      return
    }

    const transaction = db.transaction('recordings', 'readwrite')
    const store = transaction.objectStore('recordings')

    const data = { fileName, streamId, blob: audioBlob }
    store.add(data)

    transaction.oncomplete = () => {
      console.log(`Recording saved: ${fileName} (Stream ID: ${streamId})`)
    }
  }

  request.onerror = () => {
    console.error('IndexedDB error. Failed to open database.')
  }
}

function getRecordingsFromIndexedDB (callback) {
  const request = indexedDB.open('AudioDatabase', 1)

  request.onsuccess = event => {
    const db = event.target.result
    const transaction = db.transaction('recordings', 'readonly')
    const store = transaction.objectStore('recordings')
    const query = store.getAll()

    query.onsuccess = () => {
      callback(query.result)
    }
  }

  request.onerror = () => {
    console.error('Failed to retrieve recordings.')
  }
}

// Example usage:
//getRecordingsFromIndexedDB(recordings => console.log(recordings));
