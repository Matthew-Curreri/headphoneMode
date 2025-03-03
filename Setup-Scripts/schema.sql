-- ============================================================================
-- SQLite Database Schema for Headphone Mode (Production Ready)
-- When Adding a table update here and add 1 to the setupdb.sh script
-- ============================================================================

-- Attach to or create the main database file
-- PRAGMA database_list;
ATTACH DATABASE '../backend/headphoneMode.db' AS main;

-- ============================================================================
-- 1. Users Table
--    Stores basic user information (unique username and email).
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    salt            TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 2. Sessions Table
--    Tracks user login sessions with expiry.
-- ============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    session_token   TEXT NOT NULL UNIQUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at      TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ============================================================================
-- 3. Preferences Table
--    Holds user-specific preference data in JSON or text form.
-- ============================================================================
CREATE TABLE IF NOT EXISTS preferences (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id            INTEGER NOT NULL UNIQUE,
    preferences_data   TEXT NOT NULL,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ============================================================================
-- 4. Security Table
--    Manages password hashing algorithm and salt per user.
-- ============================================================================
CREATE TABLE IF NOT EXISTS security (
    user_id        INTEGER PRIMARY KEY,
    salt           TEXT NOT NULL,
    hash_algorithm TEXT NOT NULL DEFAULT 'argon2',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ============================================================================
-- 5. Usage Tracking Table
--    Tracks token usage for freemium/premium features.
-- ============================================================================
CREATE TABLE IF NOT EXISTS usage_tracking (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0 CHECK (token_count >= 0),
    last_reset  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ============================================================================
-- 6. Subscription Plans Table
--    Stores plan types and status for each user (e.g., free, premium).
-- ============================================================================
CREATE TABLE IF NOT EXISTS subscriptions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL UNIQUE,
    plan_type    TEXT NOT NULL CHECK (plan_type IN ('free', 'premium')),
    renewal_date TIMESTAMP NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('active', 'canceled', 'expired')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ============================================================================
-- 7. Roles Table
--    Defines possible roles (e.g., "admin", "moderator", "user").
-- ============================================================================
CREATE TABLE IF NOT EXISTS roles (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name  TEXT NOT NULL UNIQUE
);

-- ============================================================================
-- 8. Permissions Table
--    Defines distinct permissions (e.g., "edit_content", "delete_user").
-- ============================================================================
CREATE TABLE IF NOT EXISTS permissions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    permission_name  TEXT NOT NULL UNIQUE
);

-- ============================================================================
-- 9. Role-Permissions Mapping (Bridge Table)
--    Many-to-many relationship between roles and permissions.
-- ============================================================================
CREATE TABLE IF NOT EXISTS role_permissions (
    role_id        INTEGER NOT NULL,
    permission_id  INTEGER NOT NULL,
    PRIMARY KEY (role_id, permission_id),
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
    FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE
);

-- ============================================================================
-- 10. User-Roles Mapping (Bridge Table)
--     Many-to-many relationship between users and roles.
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_roles (
    user_id  INTEGER NOT NULL,
    role_id  INTEGER NOT NULL,
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
);

-- ============================================================================
-- 11. Prompts Table
--     (Optional) Stores available prompts for the LLM usage across users.
-- ============================================================================
CREATE TABLE IF NOT EXISTS prompts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE,
    content   TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 12. User-Prompts Mapping (Bridge Table)
--     Many-to-many mapping of which users have access to (or use) which prompts.
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_prompts (
    user_id   INTEGER NOT NULL,
    prompt_id INTEGER NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 0 CHECK (usage_count >= 0),
    PRIMARY KEY (user_id, prompt_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
);

-- ============================================================================
-- Indexes for Optimized Queries
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_users_email             ON users(email);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id        ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_preferences_user_id     ON preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_tracking_user_id  ON usage_tracking(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id   ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_user_id      ON user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_role_permissions_role_id ON role_permissions(role_id);
CREATE INDEX IF NOT EXISTS idx_prompts_name            ON prompts(name);
