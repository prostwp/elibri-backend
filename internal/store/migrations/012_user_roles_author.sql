-- 012_user_roles_author.sql
-- Adds 'author' (blogger) role to the existing users.role enum.
-- Existing CHECK constraint allowed ('user', 'admin'); blogger persona
-- needs its own role for the author dashboard and content publishing.
--
-- Author dashboard (per Ilya plan #7): blogger sees aggregated activity
-- of their subscribers. Admin keeps full platform access.
--
-- Idempotent: re-running drops + recreates the constraint with the
-- expanded set. Drops by name (constraint name fixed in migration 003).

ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;

ALTER TABLE users
  ADD CONSTRAINT users_role_check
  CHECK (role IN ('user', 'admin', 'author'));

-- For convenience: index for filtering authors in admin dashboards.
CREATE INDEX IF NOT EXISTS idx_users_role_author
  ON users(role) WHERE role = 'author';
