const express = require('express');
const db = require('../database');
const bcrypt = require('bcrypt');
const router = express.Router();
const config = require('..config/')
router.post('/', (req, res) => {
  const { username, password } = req.body;
  const saltRounds = config.saltRounds;
  const hashedPassword = bcrypt.hashSync(password, saltRounds);
  db.run(
    `INSERT INTO users (username, password) VALUES (?, ?)`,
    [username, hashedPassword],
    function (err) {
      if (err) return res.status(400).send({ error: 'Username already exists' });
      res.send({ success: true });
    }
  );
});

module.exports = router;
