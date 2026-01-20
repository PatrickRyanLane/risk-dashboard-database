-- Optional helper RPC for inserting overrides with basic validation.

create or replace function apply_item_override(
  p_url_hash text,
  p_risk_override text default null,
  p_controlled_override boolean default null,
  p_reason text default null,
  p_user_id text default null
) returns table (
  id uuid,
  url_hash text,
  risk_override text,
  controlled_override boolean,
  reason text,
  user_id text,
  created_at timestamptz
)
language plpgsql
as $$
begin
  if p_url_hash is null or length(trim(p_url_hash)) = 0 then
    raise exception 'url_hash is required';
  end if;

  if p_risk_override is null and p_controlled_override is null then
    raise exception 'Provide risk_override or controlled_override';
  end if;

  insert into item_overrides (
    url_hash,
    risk_override,
    controlled_override,
    reason,
    user_id
  ) values (
    p_url_hash,
    p_risk_override,
    p_controlled_override,
    p_reason,
    p_user_id
  )
  returning item_overrides.id, item_overrides.url_hash,
    item_overrides.risk_override, item_overrides.controlled_override,
    item_overrides.reason, item_overrides.user_id,
    item_overrides.created_at
  into id, url_hash, risk_override, controlled_override, reason, user_id, created_at;

  return next;
end;
$$;
