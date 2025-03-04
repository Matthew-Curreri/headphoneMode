const { LlamaModel } = require('./generation.js'); // Adjust the path if necessary

// Mock GPUShaderStage constant
const GPUShaderStage = {
    COMPUTE: 1
};

test('generateChatCompletion should format messages and generate a response', async () => {
    const mockConfig = { model_name: 'LlamaModel' };
    const mockDevice = {
        createShaderModule: jest.fn().mockReturnValue({}),
        createBindGroupLayout: jest.fn().mockReturnValue({}),
        createPipelineLayout: jest.fn().mockReturnValue({}),
        createComputePipeline: jest.fn().mockReturnValue({}),
        createBindGroup: jest.fn().mockReturnValue({}),
        createBuffer: jest.fn().mockReturnValue({
            getMappedRange: jest.fn().mockReturnValue(new ArrayBuffer(8)),
            unmap: jest.fn()
        }),
        queue: {
            writeBuffer: jest.fn()
        }
    };

    const model = new LlamaModel(mockConfig, mockDevice); // Pass mock config and device
    model.generateTextCompletion = jest.fn().mockResolvedValue({
        choices: [{ text: 'This is a test response.' }]
    });

    const messages = [
        { role: 'user', content: 'Hello' }
    ];
    const response = await model.generateChatCompletion(messages);
    expect(response.choices[0].message.content).toBe('This is a test response.');
});