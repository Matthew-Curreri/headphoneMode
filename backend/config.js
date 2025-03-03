// config.js - Centralized Configuration

const path = require('path');
require('dotenv').config();

const config = {
    // Base URI & Networking
    baseUri: process.env.BASE_URI || 'http://localhost',
    backendPort: process.env.PORTBACKEND || 3000,

    // Database Configuration
    dbPath: path.join(__dirname, 'headphoneMode.db'),
    
    // API Settings
    api: {
        openaiApiKey: process.env.OPENAI_API_KEY,
        openaiModel: 'gpt-4',
        maxTokens: 500,
        rateLimit: 1000,
        premiumRateLimit: 5000,
    },
    
    // Session Settings
    session: {
        expiration: 86400, // Session expiration in seconds (24 hours)
    },

    // Default User Preferences
    defaults: {
        theme: 'dark',
        language: 'en',
        notifications: true,
    },

    // Security Settings
    security: {
        hashAlgorithm: 'argon2',
        saltRounds: 12,
    },
};

module.exports = config;