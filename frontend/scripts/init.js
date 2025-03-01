let currentStream = null;
let mediaRecorder = null;
let audioChunks = [];
let audioElement = null;
let micSelect = null;


function initUI() {
  const startBtn = document.createElement('button');
  const stopBtn = document.createElement('button');
  const br = document.createElement("br");

  startBtn.textContent = 'Start';
  startBtn.id = 'startBtn';
  stopBtn.textContent = 'Stop';
  micSelect = document.createElement('select');
  micSelect.id = 'micSelect';
  
  audioElement = document.createElement('audio');
  audioElement.controls = true;
  

  document.body.appendChild(audioElement);
  document.body.appendChild(br);
  document.body.appendChild(micSelect);
  document.body.appendChild(startBtn);
  document.body.appendChild(stopBtn);
  
  startBtn.addEventListener('click', startRecording);
  stopBtn.addEventListener('click', stopRecording);
  
  updateMicList();
  setTimeout(updateMicList, 200);
  setInterval(updateMicList, 2000);

}

function updateMicList() {
  navigator.mediaDevices.enumerateDevices().then((devices) => {
    const mics = devices.filter(device => device.kind === 'audioinput');
    const selectedValue = micSelect.value;
    micSelect.innerHTML = '';
    mics.forEach(mic => {
      const option = document.createElement('option');
      option.value = mic.deviceId;
      option.textContent = mic.label || `Microphone ${micSelect.length + 1}`;
      micSelect.appendChild(option);
    });
    if (selectedValue) micSelect.value = selectedValue;
  });
}

function startRecording() {
    micAccess.webMicAccess(micSelect.value).then(stream => {
    currentStream = stream;
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    
    mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      audioElement.src = URL.createObjectURL(audioBlob);
    };
    
    mediaRecorder.start();
    console.log('Recording started...');
  });
}

function stopRecording() {
  if (mediaRecorder) {
    mediaRecorder.stop();
    currentStream.getTracks().forEach(track => track.stop());
    console.log('Recording stopped.');
  }
}

