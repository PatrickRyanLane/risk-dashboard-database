-- Optional: enable RLS and restrict writes to service role or specific users.
-- Adjust for your auth strategy before enabling in production.

-- Example: enable RLS but allow full access to service_role via bypass
-- (Supabase service role bypasses RLS by default).

alter table entities enable row level security;
alter table items enable row level security;
alter table item_overrides enable row level security;

-- Placeholder policies (allow read for authenticated users)
create policy if not exists entities_read on entities
  for select
  to authenticated
  using (true);

create policy if not exists items_read on items
  for select
  to authenticated
  using (true);

create policy if not exists overrides_read on item_overrides
  for select
  to authenticated
  using (true);

-- Allow inserts/updates only for service_role (supabase bypass) or a custom role.
-- You can add explicit write policies for your internal editors here.
