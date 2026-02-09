-- ============================================================================
-- Wellio Users Table Schema for Supabase
-- ============================================================================
-- Run this SQL in your Supabase SQL Editor to create the users table
-- 
-- Steps:
-- 1. Go to https://supabase.com/dashboard/project/omeemjdrzokykheegbnj
-- 2. Click "SQL Editor" in the left sidebar
-- 3. Click "New Query"
-- 4. Copy and paste this entire file
-- 5. Click "Run" or press Ctrl+Enter
-- ============================================================================

-- Create users table
CREATE TABLE IF NOT EXISTS users (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name TEXT NOT NULL,
  language VARCHAR(10) DEFAULT 'en',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  last_login TIMESTAMP WITH TIME ZONE,
  
  -- Email format validation
  CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
  -- Language validation
  CONSTRAINT language_check CHECK (language IN ('en', 'hi', 'te'))
);

-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Add comment to table
COMMENT ON TABLE users IS 'User authentication data for Wellio health monitoring application';

-- Add comments to columns
COMMENT ON COLUMN users.id IS 'Unique user identifier (UUID)';
COMMENT ON COLUMN users.email IS 'User email address (unique, lowercase)';
COMMENT ON COLUMN users.password_hash IS 'bcrypt hashed password with salt';
COMMENT ON COLUMN users.name IS 'User full name';
COMMENT ON COLUMN users.language IS 'Preferred language (en, hi, te)';
COMMENT ON COLUMN users.created_at IS 'Account creation timestamp';
COMMENT ON COLUMN users.last_login IS 'Last login timestamp';

-- ============================================================================
-- Row Level Security (RLS) - Optional but recommended
-- ============================================================================
-- Uncomment the following if you want to enable RLS
-- Note: For this simple app, we're using the anon key directly,
-- so RLS policies may need adjustment based on your auth strategy

-- Enable Row Level Security
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Policy: Anyone can insert (for registration)
-- CREATE POLICY "Anyone can register"
--   ON users
--   FOR INSERT
--   WITH CHECK (true);

-- Policy: Anyone can select (for login verification)
-- CREATE POLICY "Anyone can login"
--   ON users
--   FOR SELECT
--   USING (true);

-- Policy: Users can update their own last_login
-- CREATE POLICY "Users can update own login time"
--   ON users
--   FOR UPDATE
--   USING (true)
--   WITH CHECK (true);

-- ============================================================================
-- Verification Query
-- ============================================================================
-- Run this after creating the table to verify it was created successfully

SELECT 
  table_name,
  column_name,
  data_type,
  is_nullable
FROM information_schema.columns
WHERE table_name = 'users'
ORDER BY ordinal_position;

-- ============================================================================
-- Test Data (Optional - for testing only)
-- ============================================================================
-- Uncomment to insert a test user
-- Password: Test1234 (hashed with bcrypt)

-- INSERT INTO users (email, password_hash, name, created_at)
-- VALUES (
--   'test@example.com',
--   '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYIeWCrm.Ry',
--   'Test User',
--   NOW()
-- );

-- ============================================================================
-- Success!
-- ============================================================================
-- If you see the table structure above, the table was created successfully!
-- You can now use the Wellio application with Supabase authentication.
