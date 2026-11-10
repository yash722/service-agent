-- Synthetic demo-data correction for the CASE-4821 history record.
--
-- Run this script with a Snowflake role that owns the synthetic HISTORY data
-- (or has UPDATE on CASE_HISTORY). Do not grant write access to
-- SERVICING_HISTORY_READER; the application remains read-only.
--
-- The MERGE is deliberately scoped to the known synthetic record, case, and
-- legacy customer ID. Re-running it is safe: after the first correction the
-- MATCHED predicate is false and no row changes.

USE DATABASE SERVICING_AGENT_DEMO;
USE SCHEMA HISTORY;

BEGIN TRANSACTION;

MERGE INTO CASE_HISTORY AS target
USING (
    SELECT
        'HIST-001' AS HISTORY_ID,
        'CASE-4821' AS CASE_ID,
        'CUST-42' AS LEGACY_CUSTOMER_ID,
        'CUST-1001' AS AUTHORIZED_CUSTOMER_ID
) AS source
ON target.HISTORY_ID = source.HISTORY_ID
WHEN MATCHED
    AND target.CASE_ID = source.CASE_ID
    AND target.CUSTOMER_ID = source.LEGACY_CUSTOMER_ID
THEN UPDATE SET
    CUSTOMER_ID = source.AUTHORIZED_CUSTOMER_ID;

COMMIT;

-- Expected after correction: exactly one row for this synthetic history ID.
SELECT HISTORY_ID, CASE_ID, CUSTOMER_ID, STATUS
FROM CASE_HISTORY
WHERE HISTORY_ID = 'HIST-001'
  AND CASE_ID = 'CASE-4821'
  AND CUSTOMER_ID = 'CUST-1001';
