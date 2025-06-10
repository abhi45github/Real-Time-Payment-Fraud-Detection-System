"""
Database Migrations for Fraud Detection System
Handles schema migrations and data migrations
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
import asyncpg
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Migration:
    """Base migration class"""

    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.executed_at = None

    async def up(self, conn):
        """Apply migration"""
        raise NotImplementedError

    async def down(self, conn):
        """Rollback migration"""
        raise NotImplementedError


class MigrationRunner:
    """Manages database migrations"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.migrations = []

    async def init_migrations_table(self):
        """Create migrations tracking table"""

        conn = await asyncpg.connect(self.database_url)

        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    version VARCHAR(20) PRIMARY KEY,
                    description TEXT,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("Migrations table initialized")
        finally:
            await conn.close()

    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations"""

        conn = await asyncpg.connect(self.database_url)

        try:
            rows = await conn.fetch("SELECT version FROM migrations ORDER BY version")
            return [row['version'] for row in rows]
        finally:
            await conn.close()

    async def apply_migration(self, migration: Migration):
        """Apply a single migration"""

        conn = await asyncpg.connect(self.database_url)

        try:
            async with conn.transaction():
                # Apply migration
                await migration.up(conn)

                # Record migration
                await conn.execute("""
                    INSERT INTO migrations (version, description)
                    VALUES ($1, $2)
                """, migration.version, migration.description)

                logger.info(f"Applied migration {migration.version}: {migration.description}")

        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            raise

        finally:
            await conn.close()

    async def rollback_migration(self, migration: Migration):
        """Rollback a single migration"""

        conn = await asyncpg.connect(self.database_url)

        try:
            async with conn.transaction():
                # Rollback migration
                await migration.down(conn)

                # Remove migration record
                await conn.execute("""
                    DELETE FROM migrations WHERE version = $1
                """, migration.version)

                logger.info(f"Rolled back migration {migration.version}")

        except Exception as e:
            logger.error(f"Failed to rollback migration {migration.version}: {e}")
            raise

        finally:
            await conn.close()

    async def run_migrations(self):
        """Run all pending migrations"""

        await self.init_migrations_table()

        applied = await self.get_applied_migrations()
        pending = [m for m in self.migrations if m.version not in applied]

        if not pending:
            logger.info("No pending migrations")
            return

        for migration in pending:
            await self.apply_migration(migration)

        logger.info(f"Applied {len(pending)} migrations")


# Define migrations

class CreateIndexesMigration(Migration):
    """Add performance indexes"""

    def __init__(self):
        super().__init__("001", "Create performance indexes")

    async def up(self, conn):
        """Create indexes"""

        indexes = [
            # Transaction indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_created_at ON transactions(created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_risk_score_high ON transactions(risk_score) WHERE risk_score > 0.7",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_fraud ON transactions(is_fraud) WHERE is_fraud = true",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_review ON transactions(manual_review) WHERE manual_review = true",

            # User indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_fraud_count ON users(fraud_count) WHERE fraud_count > 0",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_last_transaction ON users(last_transaction_time DESC)",

            # Merchant indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_merchants_fraud_rate ON merchants(fraud_rate DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_merchants_blacklisted ON merchants(is_blacklisted) WHERE is_blacklisted = true",

            # Composite indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_user_time ON transactions(user_id, transaction_time DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_merchant_time ON transactions(merchant_id, transaction_time DESC)",
        ]

        for index_sql in indexes:
            await conn.execute(index_sql)
            logger.info(f"Created index: {index_sql[:50]}...")

    async def down(self, conn):
        """Drop indexes"""

        index_names = [
            "idx_transactions_created_at",
            "idx_transactions_risk_score_high",
            "idx_transactions_fraud",
            "idx_transactions_review",
            "idx_users_fraud_count",
            "idx_users_last_transaction",
            "idx_merchants_fraud_rate",
            "idx_merchants_blacklisted",
            "idx_transactions_user_time",
            "idx_transactions_merchant_time",
        ]

        for index_name in index_names:
            await conn.execute(f"DROP INDEX IF EXISTS {index_name}")


class CreatePartitionsMigration(Migration):
    """Partition transactions table by date"""

    def __init__(self):
        super().__init__("002", "Create table partitions for transactions")

    async def up(self, conn):
        """Create partitioned table"""

        # Create partitioned table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions_partitioned (
                LIKE transactions INCLUDING ALL
            ) PARTITION BY RANGE (transaction_time);
        """)

        # Create partitions for next 12 months
        from datetime import datetime, timedelta

        start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        for i in range(12):
            partition_date = start_date + timedelta(days=30 * i)
            next_date = partition_date + timedelta(days=30)

            partition_name = f"transactions_{partition_date.strftime('%Y_%m')}"

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF transactions_partitioned
                FOR VALUES FROM ('{partition_date.strftime('%Y-%m-%d')}')
                TO ('{next_date.strftime('%Y-%m-%d')}');
            """)

            logger.info(f"Created partition: {partition_name}")

    async def down(self, conn):
        """Drop partitioned table"""

        await conn.execute("DROP TABLE IF EXISTS transactions_partitioned CASCADE")


class CreateMaterializedViewsMigration(Migration):
    """Create materialized views for analytics"""

    def __init__(self):
        super().__init__("003", "Create materialized views")

    async def up(self, conn):
        """Create materialized views"""

        # User statistics view
        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS user_statistics AS
            SELECT
                u.id as user_id,
                u.risk_level,
                COUNT(t.id) as total_transactions,
                SUM(t.amount) as total_amount,
                AVG(t.amount) as avg_amount,
                SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) as fraud_count,
                AVG(t.risk_score) as avg_risk_score,
                MAX(t.transaction_time) as last_transaction
            FROM users u
            LEFT JOIN transactions t ON u.id = t.user_id
            GROUP BY u.id, u.risk_level;
        """)

        # Merchant statistics view
        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS merchant_statistics AS
            SELECT
                m.id as merchant_id,
                m.category,
                COUNT(t.id) as transaction_count,
                SUM(t.amount) as total_volume,
                AVG(t.amount) as avg_transaction,
                SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) as fraud_count,
                AVG(t.risk_score) as avg_risk_score,
                CASE
                    WHEN COUNT(t.id) > 0
                    THEN SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)::FLOAT / COUNT(t.id)
                    ELSE 0
                END as fraud_rate
            FROM merchants m
            LEFT JOIN transactions t ON m.id = t.merchant_id
            GROUP BY m.id, m.category;
        """)

        # Create indexes on views
        await conn.execute("""
            CREATE INDEX ON user_statistics(user_id);
            CREATE INDEX ON user_statistics(fraud_count DESC);
            CREATE INDEX ON merchant_statistics(merchant_id);
            CREATE INDEX ON merchant_statistics(fraud_rate DESC);
        """)

        logger.info("Created materialized views")

    async def down(self, conn):
        """Drop materialized views"""

        await conn.execute("DROP MATERIALIZED VIEW IF EXISTS user_statistics CASCADE")
        await conn.execute("DROP MATERIALIZED VIEW IF EXISTS merchant_statistics CASCADE")


class CreateStoredProceduresMigration(Migration):
    """Create stored procedures for common operations"""

    def __init__(self):
        super().__init__("004", "Create stored procedures")

    async def up(self, conn):
        """Create stored procedures"""

        # Procedure to update user statistics
        await conn.execute("""
            CREATE OR REPLACE FUNCTION update_user_statistics()
            RETURNS TRIGGER AS $$
            BEGIN
                UPDATE users
                SET
                    total_transactions = total_transactions + 1,
                    total_amount = total_amount + NEW.amount,
                    last_transaction_time = NEW.transaction_time,
                    fraud_count = fraud_count + CASE WHEN NEW.is_fraud THEN 1 ELSE 0 END
                WHERE id = NEW.user_id;

                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)

        # Trigger for auto-updating user stats
        await conn.execute("""
            CREATE TRIGGER update_user_stats_trigger
            AFTER INSERT ON transactions
            FOR EACH ROW
            EXECUTE FUNCTION update_user_statistics();
        """)

        # Procedure to detect fraud rings
        await conn.execute("""
            CREATE OR REPLACE FUNCTION detect_fraud_rings(
                min_shared_attributes INT DEFAULT 3,
                min_fraud_count INT DEFAULT 2
            )
            RETURNS TABLE(
                ring_id TEXT,
                user_ids TEXT[],
                shared_count INT,
                total_frauds INT
            ) AS $$
            BEGIN
                RETURN QUERY
                WITH shared_attributes AS (
                    SELECT
                        t1.user_id as user1,
                        t2.user_id as user2,
                        COUNT(DISTINCT CASE
                            WHEN t1.ip_address = t2.ip_address THEN t1.ip_address
                        END) as shared_ips,
                        COUNT(DISTINCT CASE
                            WHEN t1.device_id = t2.device_id THEN t1.device_id
                        END) as shared_devices
                    FROM transactions t1
                    JOIN transactions t2 ON t1.user_id < t2.user_id
                    WHERE t1.is_fraud = true OR t2.is_fraud = true
                    GROUP BY t1.user_id, t2.user_id
                    HAVING COUNT(*) >= min_shared_attributes
                )
                SELECT
                    MD5(array_to_string(array_agg(DISTINCT user_id), ',')) as ring_id,
                    array_agg(DISTINCT user_id) as user_ids,
                    COUNT(DISTINCT user_id) as shared_count,
                    SUM(fraud_count) as total_frauds
                FROM (
                    SELECT user1 as user_id, shared_ips + shared_devices as fraud_count
                    FROM shared_attributes
                    UNION ALL
                    SELECT user2 as user_id, shared_ips + shared_devices as fraud_count
                    FROM shared_attributes
                ) rings
                GROUP BY ring_id
                HAVING SUM(fraud_count) >= min_fraud_count;
            END;
            $$ LANGUAGE plpgsql;
        """)

        logger.info("Created stored procedures")

    async def down(self, conn):
        """Drop stored procedures"""

        await conn.execute("DROP TRIGGER IF EXISTS update_user_stats_trigger ON transactions")
        await conn.execute("DROP FUNCTION IF EXISTS update_user_statistics()")
        await conn.execute("DROP FUNCTION IF EXISTS detect_fraud_rings(INT, INT)")


async def run_all_migrations(database_url: str):
    """Run all defined migrations"""

    runner = MigrationRunner(database_url)

    # Add all migrations
    runner.migrations = [
        CreateIndexesMigration(),
        CreatePartitionsMigration(),
        CreateMaterializedViewsMigration(),
        CreateStoredProceduresMigration(),
    ]

    await runner.run_migrations()


if __name__ == "__main__":
    # Example: Run migrations
    database_url = "postgresql://fraud_user:fraud_pass@localhost/fraud_detection"
    asyncio.run(run_all_migrations(database_url))
# Updated: 2025-01-01 14:47:02 - Enhancement #5287

# Updated: 2025-01-03 20:46:54 - Enhancement #6806

# Updated: 2025-01-12 22:12:15 - Enhancement #8260

# Updated: 2025-01-14 18:29:11 - Enhancement #6208

# Updated: 2025-01-17 21:54:54 - Enhancement #9548

# Updated: 2025-01-21 21:57:06 - Enhancement #9827

# Updated: 2025-01-28 08:38:35 - Enhancement #1642

# Updated: 2025-02-08 20:52:22 - Enhancement #6659

# Updated: 2025-02-20 15:55:17 - Enhancement #8946

# Updated: 2025-02-24 08:50:26 - Enhancement #1528

# Updated: 2025-03-07 15:15:37 - Enhancement #3742

# Updated: 2025-03-26 12:04:17 - Enhancement #1407

# Updated: 2025-03-26 17:01:57 - Enhancement #1614

# Updated: 2025-04-03 19:43:30 - Enhancement #7265

# Updated: 2025-04-16 21:32:54 - Enhancement #1160

# Updated: 2025-04-17 17:15:48 - Enhancement #7334

# Updated: 2025-04-19 08:31:17 - Enhancement #6251

# Updated: 2025-04-20 13:10:02 - Enhancement #9970

# Updated: 2025-05-04 11:04:03 - Enhancement #1363

# Updated: 2025-05-15 11:24:49 - Enhancement #2048

# Updated: 2025-01-19 18:40:40 - Enhancement #2943

# Updated: 2025-02-12 19:44:28 - Enhancement #1050

# Updated: 2025-02-23 09:00:33 - Enhancement #9546

# Updated: 2025-03-05 22:40:22 - Enhancement #7413

# Updated: 2025-03-16 10:40:23 - Enhancement #5130

# Updated: 2025-03-19 14:25:19 - Enhancement #1890

# Updated: 2025-04-06 22:44:28 - Enhancement #9151

# Updated: 2025-05-11 14:23:59 - Enhancement #8713

# Updated: 2025-05-17 19:25:40 - Enhancement #1026

# Auto-generated update: 2024-12-18 15:29:40 - Task #435

# Auto-generated update: 2025-01-10 13:32:35 - Task #731

# Auto-generated update: 2025-01-23 12:52:20 - Task #103

# Auto-generated update: 2025-02-13 15:37:37 - Task #166

# Auto-generated update: 2025-02-13 14:13:02 - Task #449

# Auto-generated update: 2025-02-21 13:20:27 - Task #765

# Auto-generated update: 2025-02-21 18:04:46 - Task #866

# Auto-generated update: 2025-03-26 19:14:00 - Task #840

# Auto-generated update: 2025-04-20 13:37:56 - Task #911

# Auto-generated update: 2025-04-20 13:17:08 - Task #350

# Auto-generated update: 2025-05-04 19:11:08 - Task #694

# Auto-generated update: 2025-05-12 10:05:09 - Task #424

# Auto-generated update: 2025-05-26 11:47:34 - Task #205

# Auto-generated update: 2025-06-10 18:07:39 - Task #769
