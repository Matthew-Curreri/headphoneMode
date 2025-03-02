const express = require('express');
const sqlite3 = require('sqlite3');
const bcrypt = require('bcrypt');
const { v4: uuidv4 } = require('uuid');
const app = express();
app.use(express.json());

// Connect to SQLite
const db = new sqlite3.Database('chatapp.db');

// Session middleware
const sessions = {};  // in-memory map (token -> user_id) for simplicity
app.use((req, res, next) => {
  if (req.path.startsWith('/api')) {
    const token = req.headers['x-session-token'] || req.cookies.session;
    if (!token || !sessions[token]) {
      if (req.path === '/api/login' || req.path === '/api/register') {
        return next(); // allow auth routes without session
      }
      return res.status(401).json({ error: 'Unauthorized' });
    }
    req.userId = sessions[token];
  }
  next();
});

// Registration endpoint
app.post('/api/register', (req, res) => {
  const { username, password } = req.body;
  // (Validate inputs...)
  const passHash = bcrypt.hashSync(password, 10);
  db.run(`INSERT INTO Users (username, password_hash) VALUES (?, ?)`, 
         [username, passHash], function(err) {
    if (err) {
      return res.status(500).json({ error: 'User registration failed' });
    }
    // Auto-login after register
    const token = uuidv4();
    sessions[token] = this.lastID; // lastID is the new user's ID
    res.cookie('session', token, { httpOnly: true });
    res.json({ success: true, userId: this.lastID });
  });
});

// Login endpoint
app.post('/api/login', (req, res) => {
  const { username, password } = req.body;
  db.get(`SELECT user_id, password_hash FROM Users WHERE username = ?`, [username], (err, row) => {
    if (err || !row) {
      return res.status(400).json({ error: 'Invalid username or password' });
    }
    if (!bcrypt.compareSync(password, row.password_hash)) {
      return res.status(400).json({ error: 'Invalid username or password' });
    }
    // Create session
    const token = uuidv4();
    sessions[token] = row.user_id;
    res.cookie('session', token, { httpOnly: true });
    res.json({ success: true });
  });
});

// Message endpoint (store message)
app.post('/api/message', (req, res) => {
  const userId = req.userId;
  const { chatId, role, text } = req.body;
  const chat_id = chatId || getDefaultChatForUser(userId); // implement as needed
  const timestamp = Date.now();
  db.run(`INSERT INTO Messages (chat_id, sender, content, timestamp) VALUES (?, ?, ?, ?)`,
         [chat_id, role, text, timestamp], function(err) {
    if (err) return res.status(500).json({ error: 'DB insert failed' });
    res.json({ messageId: this.lastID, timestamp });
  });
});
