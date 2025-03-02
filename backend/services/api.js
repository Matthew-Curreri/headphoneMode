const express = require('express');
const bodyParser = require('body-parser');
const db = require('../database');
const register = require('../api/register');
const login = require('../api/login');
const messages = require('../api/messages');
const feedback = require('../api/feedback');

const app = express();
app.use(bodyParser.json());

// Register API routes
app.use('/api/register', register);
app.use('/api/login', login);
app.use('/api/messages', messages);
app.use('/api/feedback', feedback);

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Backend running on port ${PORT}`);
});
