-- MindKey: wellbeing check-ins (user-reported context for the investigation).
-- Run once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
-- Safe to re-run.

create table if not exists public.checkins (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null,
  date        date not null,
  factor      text not null check (factor in (
                'feeling_well', 'tired', 'stressed', 'poor_sleep',
                'unwell', 'distracted', 'other')),
  note        text check (note is null or char_length(note) <= 280),
  created_at  timestamptz not null default now()
);

create index if not exists checkins_user_date_idx
  on public.checkins (user_id, date);

-- Only the backend (service-role key) reads and writes check-ins.
-- RLS on with no policies blocks the public anon key entirely.
alter table public.checkins enable row level security;
