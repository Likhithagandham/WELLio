-- Create a table to track user files uploaded to S3
create table if not exists public.user_files (
  id uuid default gen_random_uuid() primary key,
  user_id uuid not null, -- Supabase Auth user ID (from auth.users)
  file_name text not null,
  s3_bucket text not null,
  s3_key text not null,
  file_url text, -- Presigned URL or public URL if bucket is public
  file_size_bytes bigint,
  content_type text,
  uploaded_at timestamptz default now()
);

-- Enable Row Level Security (RLS)
alter table public.user_files enable row level security;

-- Policy: Users can see their own files
create policy "Users can view their own files"
on public.user_files for select
using ( auth.uid() = user_id );

-- Policy: Users can insert their own files
create policy "Users can upload their own files"
on public.user_files for insert
with check ( auth.uid() = user_id );

-- Policy: Users can delete their own files
create policy "Users can delete their own files"
on public.user_files for delete
using ( auth.uid() = user_id );

-- Add index for faster lookups by user
create index idx_user_files_user_id on public.user_files(user_id);
