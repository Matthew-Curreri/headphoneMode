Headphone Mode Database Design

The Headphone Mode project employs a SQLite database to store user data, session management, subscription details, usage tracking, prompts, and more. This document provides an overview of the schema and explains the rationale behind each table and its relationships.
Table of Contents

    Overview
    Schema Diagram (High-Level)
    Table Descriptions
        1. Users
        2. Sessions
        3. Preferences
        4. Security
        5. Usage Tracking
        6. Subscriptions
        7. Roles
        8. Permissions
        9. Role-Permissions Mapping
        10. User-Roles Mapping
        11. Prompts
        12. User-Prompts Mapping
    Cascading & Data Integrity
    Indexing Strategy
    Future Considerations

Overview

Headphone Mode’s database focuses on:

    User Management: Authentication, security, and personalized preferences.
    Subscription & Freemium/Premium Features: Tracking token usage and subscription tiers.
    Access Control (Roles & Permissions): Granular assignment of roles and capabilities.
    Prompts & LLM Integration: Storage and usage tracking of LLM prompt templates.

The schema is designed to be scalable, secure, and maintainable. It emphasizes referential integrity and strict constraints to prevent invalid data.
Schema Diagram (High-Level)

erDiagram

    USERS ||--|{ SESSIONS : "has"
    USERS ||--|{ PREFERENCES : "has"
    USERS ||--|{ USAGE_TRACKING : "has"
    USERS ||--|| SECURITY : "has"
    USERS ||--|| SUBSCRIPTIONS : "has"
    USERS ||--|{ USER_ROLES : "can have"
    
    ROLES ||--|{ ROLE_PERMISSIONS : "can have"
    PERMISSIONS ||--|{ ROLE_PERMISSIONS : "is assigned"
    
    ROLES ||--|{ USER_ROLES : "belongs to"
    
    PROMPTS ||--|{ USER_PROMPTS : "is used by"
    USERS ||--|{ USER_PROMPTS : "uses"

Table Descriptions
1. Users

Stores primary user data.
Columns:

    id: Unique identifier (PK).
    username: Unique username.
    email: Unique email address.
    password_hash: Hashed user password.
    salt: Salt used in hashing.
    created_at: Timestamp of account creation.

Why?
Keeps track of essential user credentials and identifies users throughout the system.
2. Sessions

Tracks active login sessions, including expiry times.
Columns:

    id: Session ID (PK).
    user_id: References users(id).
    session_token: Unique token for the active session.
    created_at: When the session was created.
    expires_at: When the session becomes invalid.

Why?
Allows session-based authentication, enabling secure logins with the ability to expire sessions.
3. Preferences

Stores user-specific settings (e.g., UI color schemes, notification preferences).
Columns:

    id: PK.
    user_id: References users(id).
    preferences_data: JSON or text data with user settings.
    updated_at: Timestamp for the last update.

Why?
Centralizes user preferences for easy retrieval and update.
4. Security

Tracks the hashing algorithm and salt per user, allowing future updates to hashing methods.
Columns:

    user_id: PK (maps 1:1 with users(id)).
    salt: Salt used in hashing.
    hash_algorithm: Defaults to argon2.

Why?
Supports evolving security needs (e.g., transitioning from argon2 to another algorithm without reworking the entire schema).
5. Usage Tracking

Monitors token usage for Freemium/Premium models.
Columns:

    id: PK.
    user_id: References users(id).
    token_count: Tracks total tokens consumed.
    last_reset: Timestamp of the last usage reset.

Why?
Enables usage-based billing or limitations, crucial for an LLM-powered service.
6. Subscriptions

Defines user plan types (e.g., free or premium) and subscription status.
Columns:

    id: PK.
    user_id: Unique reference to users(id).
    plan_type: Enforced by CHECK(plan_type IN ('free','premium')).
    renewal_date: Next billing or renewal date.
    status: active, canceled, or expired.

Why?
Manages billing cycles and subscription statuses for each user.
7. Roles

Defines roles (e.g., admin, moderator, user).
Columns:

    id: PK.
    role_name: Unique name for the role.

Why?
Allows the system to categorize users by roles for permission-based access control.
8. Permissions

Defines discrete permissions (e.g., edit_content, delete_user).
Columns:

    id: PK.
    permission_name: Unique permission name.

Why?
Enables fine-grained control over system capabilities.
9. Role-Permissions Mapping

Resolves the many-to-many relationship between roles and permissions.
Columns:

    role_id: References roles(id).
    permission_id: References permissions(id).
    Primary Key: (role_id, permission_id) (composite).

Why?
Each role can have multiple permissions, and permissions can be shared among multiple roles.
10. User-Roles Mapping

Defines which users have which roles.
Columns:

    user_id: References users(id).
    role_id: References roles(id).
    Primary Key: (user_id, role_id) (composite).

Why?
A user can hold multiple roles, and a role can be assigned to many users.
11. Prompts

Stores pre-defined or custom prompts for LLM interactions.
Columns:

    id: PK.
    name: Unique identifier for the prompt.
    content: The text of the prompt.
    created_at: Timestamp for creation.

Why?
Centralizes prompt templates, making them shareable and reusable across the application.
12. User-Prompts Mapping

Tracks which prompts each user has access to, or how many times they’ve used them.
Columns:

    user_id: References users(id).
    prompt_id: References prompts(id).
    usage_count: Number of times the user used this prompt.
    Primary Key: (user_id, prompt_id).

Why?
Enables advanced features like personalized usage tracking or prompt-based analytics.
Cascading & Data Integrity

    ON DELETE CASCADE: Most foreign keys use this to automatically remove related data when a user or parent record is deleted.
    Check Constraints:
        plan_type IN ('free', 'premium')
        status IN ('active', 'canceled', 'expired')
        token_count >= 0

These measures ensure that references remain valid, prevent orphaned records, and disallow invalid data insertion.
Indexing Strategy

    User Email – idx_users_email for quick lookups during login or updates.
    User ID-Based Indexes – On tables like sessions, preferences, usage_tracking, etc., to expedite frequent JOINs.
    Name-Based Indexes – On prompts(name) for prompt searches.

Indexes accelerate read performance but can slow down inserts/updates. Each index is chosen to optimize common queries without excessive overhead.
Future Considerations

    Database Migration: When new features or tables are added, consider a migration strategy to preserve existing data (e.g., scripts, versioned migrations).
    Sharding or Splitting: If usage grows significantly, you might look at higher-scale solutions (e.g., a full SQL server or cloud-based DB).
    Logging & Auditing: Consider adding audit logs to track user changes, especially if more stringent compliance or debugging is required.
    Encryption: Evaluate the need for encrypting sensitive data at rest in the database.

Questions or Feedback?
Feel free to open a GitHub issue or contact the team if you have questions about this design or encounter challenges while using the database.