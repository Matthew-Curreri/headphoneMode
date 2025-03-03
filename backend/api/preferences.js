// preferences.js - Handles user preferences storage and retrieval

const sqlite3 = require('sqlite3').verbose()
const config = require(path.join(process.cwd(), 'backend/config.js'))
const express = require('express')
const router = express.Router()

const db = new sqlite3.Database(config.dbPath)

router.get('/', (req, res) => {
  const { userId } = req.query
  if (!userId) {
    return res.status(400).json({ error: 'Missing userId parameter' })
  }

  const query = `SELECT preferences_data FROM preferences WHERE user_id = ?`
  db.get(query, [userId], (err, row) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' })
    }
    res
      .status(200)
      .json({ preferences: row ? JSON.parse(row.preferences_data) : {} })
  })
})

router.post('/', (req, res) => {
  const { userId, preferences } = req.body
  if (!userId || !preferences) {
    return res.status(400).json({ error: 'Missing required fields' })
  }

  const query = `INSERT INTO preferences (user_id, preferences_data, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) 
                   ON CONFLICT(user_id) DO UPDATE SET preferences_data = excluded.preferences_data, updated_at = CURRENT_TIMESTAMP`

  db.run(query, [userId, JSON.stringify(preferences)], function (err) {
    if (err) {
      return res.status(500).json({ error: 'Database update error' })
    }
    res.status(200).json({ success: true })
  })
})

module.exports = router

// Test the Preferences API handler
const testPreferencesAPI = async () => {
  const testUserId = 1
  const testPreferences = {
    theme: 'light',
    language: 'fr',
    notifications: false
  }

  // Test updatePreferences
  await fetch(`${config.baseUri}:${config.backendPort}/api/preferences`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId: testUserId, preferences: testPreferences })
  })

  // Test getPreferences
  const response = await fetch(
    `${config.baseUri}:${config.backendPort}/api/preferences?userId=${testUserId}`
  )
  const data = await response.json()
  console.log('Retrieved Preferences:', data)
}

// Uncomment to run test
testPreferencesAPI()
