import { LlamaModel } from './generation.js';

document.getElementById('runTestButton').addEventListener('click', async () => {
  const mockConfig = { model_name: 'LlamaModel' };
  const device = window.device;

  if (!device) {
    console.error("WebGPU device not initialized.");
    return;
  }

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