const chatContainer = document.getElementById('chat-container')
const input = document.getElementById('chat-input')
const sendBtn = document.getElementById('send-btn')

let chatHistory = [] // to hold the conversation turns for context

// Append a message to the chat UI and history
function addMessage (author, text) {
  const msgDiv = document.createElement('div')
  msgDiv.className = `message ${author}` // e.g. "message user" or "message assistant"
  msgDiv.innerText = text
  // Optionally add timestamp
  const time = new Date().toLocaleTimeString()
  msgDiv.setAttribute('data-time', time)
  chatContainer.appendChild(msgDiv)
  chatContainer.scrollTop = chatContainer.scrollHeight // scroll to bottom
  // Record in history (role could be 'user' or 'assistant')
  chatHistory.push({ role: author, content: text })
}

// Handle send button click
sendBtn.addEventListener('click', async () => {
  const userText = input.value.trim()
  if (!userText) return
  addMessage('user', userText)
  input.value = '' // clear input
  // Send user's message to backend for storage
  fetch('/api/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: userText, role: 'user' })
  })
  // Generate assistant's reply using the ML model in-browser
  const reply = await generateReply(chatHistory)
  addMessage('assistant', reply)
  // Store assistant's reply in backend as well
  fetch('/api/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: reply, role: 'assistant' })
  })
})

document.getElementById('sendButton').addEventListener('click', async () => {
    const messages = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is the weather today?' }
    ];
    const reply = await generateReply(messages);
    console.log(reply);
});

document.getElementById('loadModelButton').addEventListener('click', async () => {
  const fileInput = document.getElementById('binaryUpload');
  if (fileInput.files.length === 0) {
    alert('Please upload a binary file.');
    return;
  }
  const binaryFile = fileInput.files[0];
  const binaryData = await binaryFile.arrayBuffer();
  const model = await LlamaModel.loadFromBinary('/path/to/config.json', binaryData);
  window.model = model; // Make the model globally accessible
  alert('Model loaded successfully.');
});

// Modify the generateReply function to use the loaded model
async function generateReply(messages) {
  if (!window.model) {
    throw new Error('Model not loaded. Please upload a binary file and load the model.');
  }
  const response = await window.model.generateChatCompletion(messages);
  return response.choices[0].message.content;
}
