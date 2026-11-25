"""enable row level security on tenant-owned tables

Revision ID: a3c1f9e27b10
Revises: 564e710830c9
Create Date: 2026-10-08 10:00:00.000000

Policy design
-------------
Every policy is: row's tenant matches `app.current_tenant`, OR the session
is in the explicit system context (`app.bypass_rls = 'on'`, used only by
signup/login). USING governs which rows are visible/updatable/deletable;
WITH CHECK governs what may be inserted/updated *into*, which is what stops
a tenant from writing rows that belong to someone else.

FORCE ROW LEVEL SECURITY makes the policies apply to the table owner too.
Superusers and BYPASSRLS roles still skip RLS in Postgres no matter what,
which is why the API must connect as the restricted `documind_app` role
(see scripts/init-db.sql) and only migrations use the owner/superuser role.

The (SELECT fn()) form lets the planner evaluate the function once per
statement instead of once per row.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "a3c1f9e27b10"
down_revision: Union[str, None] = "564e710830c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# table -> column holding the tenant id
TENANT_TABLES = {
    "tenants": "id",
    "users": "tenant_id",
    "documents": "tenant_id",
    "chunks": "tenant_id",
}


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return  # RLS is a Postgres feature; SQLite test DBs skip it.

    op.execute(
        """
        CREATE OR REPLACE FUNCTION app_current_tenant() RETURNS uuid
        LANGUAGE sql STABLE AS
        $$ SELECT NULLIF(current_setting('app.current_tenant', true), '')::uuid $$
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app_bypass_rls() RETURNS boolean
        LANGUAGE sql STABLE AS
        $$ SELECT COALESCE(current_setting('app.bypass_rls', true), '') = 'on' $$
        """
    )

    for table, column in TENANT_TABLES.items():
        predicate = f"({column} = (SELECT app_current_tenant()) OR (SELECT app_bypass_rls()))"
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING {predicate} WITH CHECK {predicate}"
        )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    op.execute("DROP FUNCTION IF EXISTS app_bypass_rls()")
    op.execute("DROP FUNCTION IF EXISTS app_current_tenant()")
