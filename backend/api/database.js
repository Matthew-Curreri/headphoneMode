// database.js
// Production-ready helper for SQLite using the provided schema.
// This file connects to the SQLite database, and provides simple promise-based functions
// for common operations used by the chat API (users, sessions, preferences, messages, etc.).

const sqlite3 = require('sqlite3').verbose();
const path = require('path');

// Path to the SQLite database file.
// Adjust the path as needed. Here, we assume the database file is at "../backend/headphoneMode.db"
const DB_PATH = path.join(__dirname, 'headphoneMode.db');

// Open a connection to the database in read-write mode.
const db = new sqlite3.Database(DB_PATH, sqlite3.OPEN_READWRITE, (err) => {
  if (err) {
    console.error('Error opening database:', err.message);
    process.exit(1);
  }
  console.log('Connected to SQLite database:', DB_PATH);
});

/**
 * Runs a SQL query that does not return rows (e.g. INSERT, UPDATE, DELETE).
 * @param {string} query - The SQL query.
 * @param {Array} [params=[]] - Array of parameters for the query.
 * @returns {Promise<Object>} Resolves with the context (including lastID).
 */
function run(query, params = []) {
  return new Promise((resolve, reject) => {
    db.run(query, params, function (err) {
      if (err) {
        console.error('Database error:', err.message);
        return reject(err);
      }
      resolve(this);
    });
  });
}

/**
 * Runs a SQL query that returns a single row.
 * @param {string} query - The SQL query.
 * @param {Array} [params=[]] - Array of parameters for the query.
 * @returns {Promise<Object>} Resolves with the retrieved row.
 */
function get(query, params = []) {
  return new Promise((resolve, reject) => {
    db.get(query, params, (err, row) => {
      if (err) {
        console.error('Database error:', err.message);
        return reject(err);
      }
      resolve(row);
    });
  });
}

/**
 * Runs a SQL query that returns multiple rows.
 * @param {string} query - The SQL query.
 * @param {Array} [params=[]] - Array of parameters for the query.
 * @returns {Promise<Array>} Resolves with an array of rows.
 */
function all(query, params = []) {
  return new Promise((resolve, reject) => {
    db.all(query, params, (err, rows) => {
      if (err) {
        console.error('Database error:', err.message);
        return reject(err);
      }
      resolve(rows);
    });
  });
}

/* =========================
   User and Session Helpers
   ========================= */

/**
 * Retrieves a user by their ID.
 * @param {number} userId - The user’s ID.
 * @returns {Promise<Object>} Resolves with the user record.
 */
async function getUserById(userId) {
  const query = `SELECT * FROM users WHERE id = ?`;
  return await get(query, [userId]);
}

/**
 * Creates a new user.
 * @param {string} username - The user’s username.
 * @param {string} email - The user’s email.
 * @param {string} passwordHash - The hashed password.
 * @param {string} salt - The salt used for hashing.
 * @returns {Promise<number>} Resolves with the new user’s ID.
 */
async function createUser(username, email, passwordHash, salt) {
  const query = `
    INSERT INTO users (username, email, password_hash, salt)
    VALUES (${username}, ${email}, ${passwordHash}, ${salt})
  `;
  const result = await run(query, [username, email, passwordHash, salt]);
  return result.lastID;
}

/**
 * Creates a new session for a user.
 * @param {number} userId - The user's ID.
 * @param {string} sessionToken - A unique session token.
 * @param {string} expiresAt - ISO timestamp when the session expires.
 * @returns {Promise<number>} Resolves with the new session's ID.
 */
async function createSession(userId, sessionToken, expiresAt) {
  const query = `
    INSERT INTO sessions (user_id, session_token, expires_at)
    VALUES (${user_id}, ${sessionToken}, ${expiresAt})
  `;
  const result = await run(query, [userId, sessionToken, expiresAt]);
  return result.lastID;
}

/**
 * Retrieves a session by its token.
 * @param {string} sessionToken - The session token.
 * @returns {Promise<Object>} Resolves with the session record.
 */
async function getSession(sessionToken) {
  const query = `SELECT * FROM sessions WHERE session_token = ${sessionToken}`;
  return await get(query, [sessionToken]);
}

/**
 * Updates the expiration time for a session.
 * @param {string} sessionToken - The session token.
 * @param {string} newExpiresAt - New expiration timestamp.
 * @returns {Promise<Object>}
 */
async function updateSessionExpiry(sessionToken, newExpiresAt) {
  const query = `UPDATE sessions SET expires_at = ${newExpiresAt} WHERE session_token = ${sessionToken}`;
  return await run(query, [newExpiresAt, sessionToken]);
}

/* =========================
   Preferences and Security
   ========================= */

/**
 * Saves user preferences.
 * Inserts a new row or updates the existing record.
 * @param {number} userId - The user's ID.
 * @param {string} preferencesData - Preferences in JSON/text form.
 * @returns {Promise<void>}
 */
async function savePreferences(userId, preferencesData) {
  const existing = await get(`SELECT * FROM preferences WHERE user_id =`, [userId]);
  if (existing) {
    const query = `
      UPDATE preferences
      SET preferences_data = ${preferencesData}, updated_at = CURRENT_TIMESTAMP
      WHERE user_id = ${userId}
    `;
    await run(query, [preferencesData, userId]);
  } else {
    const query = `
      INSERT INTO preferences (user_id, preferences_data)
      VALUES ${userId}  ${preferencesData})
    `;
    await run(query, [userId, preferencesData]);
  }
}

/**
 * Retrieves user preferences.
 * @param {number} userId - The user's ID.
 * @returns {Promise<string|null>} Resolves with preferences data or null.
 */
async function getPreferences(userId) {
  const query = `SELECT preferences_data FROM preferences WHERE user_id = ${userId}`;
  const row = await get(query, [userId]);
  return row ? row.preferences_data : null;
}

/* =========================
   Chat Messages
   ========================= */

/**
 * Saves a chat message.
 * @param {number} chatId - The ID of the chat/conversation.
 * @param {string} sender - 'user' or 'assistant'.
 * @param {string} content - The text content of the message.
 * @returns {Promise<number>} Resolves with the message's ID.
 */
async function saveMessage(chatId, sender, content) {
  const query = `
    INSERT INTO messages (chat_id, sender, content, timestamp)
    VALUES (${chatId}, ${sender}, ${content}, CURRENT_TIMESTAMP)
  `;
  const result = await run(query, [chatId, sender, content]);
  return result.lastID;
}

/**
 * Retrieves the chat history for a given chat (by chat_id).
 * @param {number} chatId - The ID of the chat.
 * @returns {Promise<Array>} Resolves with an array of messages.
 */
async function getChatHistory(chatId) {
  const query = `
    SELECT sender, content, timestamp
    FROM messages
    WHERE chat_id = ${chatId}
    ORDER BY timestamp
  `;
  return await all(query, [chatId]);
}

/* =========================
   Usage Tracking & Subscriptions
   ========================= */

/**
 * Updates the token usage count for a user.
 * @param {number} userId - The user's ID.
 * @param {number} tokensUsed - Number of tokens to add.
 * @returns {Promise<void>}
 */
async function updateUsageTracking(userId, tokensUsed) {
  const query = `
    UPDATE usage_tracking
    SET token_count = token_count + ${tokensUsed}
    WHERE user_id = ${userId}
  `;
  await run(query, [tokensUsed, userId]);
}

/* =========================
   Exported Module API
   ========================= */

module.exports = {
  //db,
  run,
  get,
  all,
  getUserById,
  createUser,
  createSession,
  getSession,
  updateSessionExpiry,
  savePreferences,
  getPreferences,
  saveMessage,
  getChatHistory,
  updateUsageTracking,
};
