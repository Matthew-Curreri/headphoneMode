// sessionHandler.js - Manages session operations

const sqlite3 = require('sqlite3').verbose();
const config = require('../config');

const db = new sqlite3.Database(config.dbPath);

const SessionHandler = {
    createSession: (userId, sessionToken, expiresAt, callback) => {
        const query = `INSERT INTO sessions (user_id, session_token, expires_at) VALUES (?, ?, ?)`;
        db.run(query, [userId, sessionToken, expiresAt], function (err) {
            callback(err, this.lastID);
        });
    },

    getSession: (sessionToken, callback) => {
        const query = `SELECT * FROM sessions WHERE session_token = ? AND expires_at > CURRENT_TIMESTAMP`;
        db.get(query, [sessionToken], (err, row) => {
            callback(err, row);
        });
    },

    deleteSession: (sessionToken, callback) => {
        const query = `DELETE FROM sessions WHERE session_token = ?`;
        db.run(query, [sessionToken], function (err) {
            callback(err, this.changes);
        });
    },
};

module.exports = SessionHandler;


// Test the session handler functions
const testSessionHandler = () => {
    const testUserId = 1;
    const testToken = 'test-session-token';
    const testExpiry = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();
    
    // Test createSession
    SessionHandler.createSession(testUserId, testToken, testExpiry, (err, id) => {
        if (err) console.error('Error creating session:', err);
        else console.log('Session created with ID:', id);
    });

    // Test getSession
    SessionHandler.getSession(testToken, (err, session) => {
        if (err) console.error('Error retrieving session:', err);
        else console.log('Retrieved session:', session);
    });

    // Test deleteSession
    SessionHandler.deleteSession(testToken, (err, changes) => {
        if (err) console.error('Error deleting session:', err);
        else console.log('Deleted session, changes:', changes);
    });
};

// Uncomment to run tests
 testSessionHandler();

