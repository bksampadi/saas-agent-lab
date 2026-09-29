"""licence agent policy

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-29 09:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Named and written as SQLAlchemy writes the Enum's CHECK, so the schema
# matches create_all() exactly.
AGENT_POLICY_CHECK_NAME = "ck_licences_policy_decision"
AGENT_POLICY_CHECK = "agent_policy IN ('allow', 'require_approval', 'deny')"


def upgrade() -> None:
    # One ALTER TABLE, the same on SQLite and Postgres, instead of batch mode.
    # On SQLite, batch mode rebuilds the table (copy, drop, rename), and it
    # cannot drop licences while assignments and agent runs reference it:
    # both, and the runs' trace rows in turn, would have to be set aside as
    # 0005 does for agent_runs. op.add_column would not rebuild, but on
    # SQLite it leaves out the Enum's CHECK. A column constraint needs
    # neither. The default gives every existing licence "allow", which is how
    # agent runs have treated every licence so far.
    op.execute(
        "ALTER TABLE licences ADD COLUMN agent_policy VARCHAR(32) "
        "DEFAULT 'allow' NOT NULL "
        f"CONSTRAINT {AGENT_POLICY_CHECK_NAME} CHECK ({AGENT_POLICY_CHECK})"
    )


def downgrade() -> None:
    # Revision 0005 has no agent policy, so dropping a policy other than
    # "allow" would silently lift it. Refuse rather than lose it.
    remaining = (
        op.get_bind()
        .execute(sa.text("SELECT COUNT(*) FROM licences WHERE agent_policy != 'allow'"))
        .scalar_one()
    )
    if remaining:
        raise RuntimeError(
            f"{remaining} licence(s) have an agent policy other than 'allow'; "
            "revision 0005 cannot store it. Set them to 'allow' first."
        )

    # SQLite drops the column's own CHECK with it (3.35 and later).
    op.execute("ALTER TABLE licences DROP COLUMN agent_policy")
