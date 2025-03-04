import { LlamaModel } from './generation.js';

document.getElementById('runTestButton').addEventListener('click', async () => {
  // Request the GPU adapter and device
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error("WebGPU not supported or available in this environment.");
    return;
  }
  const device = await adapter.requestDevice();

  const mockConfig = { model_name: 'LlamaModel' };

  const model = new LlamaModel(mockConfig, device);
  model.generateTextCompletion = async (messages) => {
    return {
      choices: [{ message: { content: 'This is a test response.' } }]
    };
  };

  const messages = [
    { role: 'user', content: 'Hello' }
  ];
  const response = await model.generateChatCompletion(messages);
  const resultContainer = document.getElementById('testResult');
  resultContainer.textContent = response.choices[0].message.content;
});