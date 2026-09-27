"""tool call observations

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-27 23:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OBSERVATION_CHECK = "ck_tool_calls_observation_needs_outcome"


def upgrade() -> None:
    # SQLite rebuilds tool_calls here. No table references it, so its rows
    # can stay where they are while it is copied, unlike agent_runs in 0003.
    with op.batch_alter_table("tool_calls", schema=None) as batch_op:
        batch_op.add_column(sa.Column("observation", sa.Text(), nullable=True))
        batch_op.create_check_constraint(
            op.f(OBSERVATION_CHECK), "observation IS NULL OR status != 'started'"
        )


def downgrade() -> None:
    # Revision 0003 cannot store what a model was shown. Refuse rather than
    # delete it.
    remaining = (
        op.get_bind()
        .execute(
            sa.text("SELECT COUNT(*) FROM tool_calls WHERE observation IS NOT NULL")
        )
        .scalar_one()
    )
    if remaining:
        raise RuntimeError(
            f"{remaining} tool call(s) hold a model-visible observation; "
            "revision 0003 cannot store them. Remove them first."
        )

    with op.batch_alter_table("tool_calls", schema=None) as batch_op:
        batch_op.drop_constraint(op.f(OBSERVATION_CHECK), type_="check")
        batch_op.drop_column("observation")
