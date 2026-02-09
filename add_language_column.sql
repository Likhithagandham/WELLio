-- ============================================================================
-- Add Language Column to Existing Users Table
-- ============================================================================
-- Run this SQL in your Supabase SQL Editor
-- 
-- Steps:
-- 1. Go to https://supabase.com/dashboard/project/omeemjdrzokykheegbnj
-- 2. Click "SQL Editor" in the left sidebar
-- 3. Click "New Query"
-- 4. Copy and paste the commands below
-- 5. Click "Run" or press Ctrl+Enter
-- ============================================================================

-- Add language column to existing users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS language VARCHAR(10) DEFAULT 'en';

-- Add language validation constraint
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'language_check'
    ) THEN
        ALTER TABLE users ADD CONSTRAINT language_check CHECK (language IN ('en', 'hi', 'te'));
    END IF;
END $$;

-- Add comment
COMMENT ON COLUMN users.language IS 'Preferred language (en, hi, te)';

-- Verify the column was added
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'users'
ORDER BY ordinal_position;

-- ============================================================================
-- Success!
-- ============================================================================
-- If you see the 'language' column in the output above, you're all set!
-- You can now create accounts with language preferences.
-- ============================================================================
