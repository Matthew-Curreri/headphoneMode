// openai.js - Handles LLM interactions

const OpenAI = require('openai');
const config = require('/home/mcurreri/Projects/headphone-mode/backend/config.js');

const openai = new OpenAI({ apiKey: config.api.openaiApiKey });

async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    const { prompt, sessionToken } = req.body;
    if (!prompt || !sessionToken) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    try {
        const completion = await openai.chat.completions.create({
            model: 'gpt-4',
            messages: [{ role: 'user', content: prompt }],
            max_tokens: config.api.rateLimit,
        });
        
        res.status(200).json({ response: completion.choices[0].message.content });
    } catch (error) {
        console.error('OpenAI API Error:', error);
        res.status(500).json({ error: 'Failed to process request' });
    }
}

module.exports = handler;
/*
// Test the OpenAI API handler
async function testOpenAI() {
    const testPrompt = 'Hello, how are you?';
    const testSessionToken = 'test-session-token';

    try {
        const response = await fetch(`${config.baseUri}:${config.port}/api/openai`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: testPrompt, sessionToken: testSessionToken }),
        });
        
        const data = await response.json();
        console.log('Test OpenAI API Response:', data);
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Uncomment to run test
 testOpenAI();
*/