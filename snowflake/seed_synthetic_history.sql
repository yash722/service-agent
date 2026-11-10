-- Canonical synthetic history seed for the Servicing Agent local demo.
--
-- Execute this script only in a session using the role that owns the synthetic
-- SERVICING_AGENT_DEMO.HISTORY tables (or an equivalently scoped data-owner
-- role). Never grant INSERT, UPDATE, DELETE, MERGE, or OWNERSHIP to the
-- application role SERVICING_HISTORY_READER. The application remains read-only.
--
-- Every statement is an ID-keyed MERGE. Re-running the script converges the
-- named synthetic records to this canonical state without deleting other rows.

USE DATABASE SERVICING_AGENT_DEMO;
USE SCHEMA HISTORY;

BEGIN TRANSACTION;

MERGE INTO CUSTOMER_ACTIVITY AS target
USING (
    SELECT
        column1::VARCHAR AS ACTIVITY_ID,
        column2::VARCHAR AS CUSTOMER_ID,
        column3::VARCHAR AS PRODUCT_ID,
        column4::VARCHAR AS EVENT_TYPE,
        column5::VARCHAR AS EVENT_DESCRIPTION,
        column6::VARCHAR AS EVENT_TIME_TEXT
    FROM VALUES
        (
            'ACT-001', 'CUST-1001', 'PRD-TELEMETRY',
            'TELEMETRY_DELAY_OBSERVED',
            'CASE-4821 monitoring observed three synchronization windows with forty-to-sixty-minute telemetry delays; the reviewed batches later arrived.',
            '2026-11-05T09:12:00Z'
        ),
        (
            'ACT-002', 'CUST-1001', 'PRD-TELEMETRY',
            'CONNECTOR_RETRIES_OBSERVED',
            'CASE-4821 connector telemetry recorded repeated retries during the delayed windows; the record does not establish a definitive root cause.',
            '2026-11-05T09:18:00Z'
        ),
        (
            'ACT-4822-001', 'CUST-1002', 'PRD-BILLING',
            'BILLING_CONTACT_CONFIRMED',
            'For CASE-4822, the customer confirmed the corrected billing contact after administrator verification and the billing-system update.',
            '2026-11-05T14:12:00Z'
        ),
        (
            'ACT-4823-001', 'CUST-1003', 'PRD-DASHBOARD',
            'DASHBOARD_DIAGNOSTICS_REQUESTED',
            'For CASE-4823, support requested the mobile application version and an affected event timestamp; receipt is not recorded.',
            '2026-11-04T16:40:00Z'
        ),
        (
            'ACT-4824-001', 'CUST-1002', 'PRD-TELEMETRY',
            'TELEMETRY_RETRY_RECOVERED',
            'For CASE-4824, one export window failed and its next retry completed normally; the Standard-tier escalation threshold was not reached.',
            '2026-11-02T17:50:00Z'
        ),
        (
            'ACT-4825-001', 'CUST-1001', 'PRD-TELEMETRY',
            'TELEMETRY_BACKLOG_CLEARED',
            'For CASE-4825, monitoring confirmed that the queued telemetry export backlog cleared after certificate rotation and replay completed.',
            '2026-11-01T12:30:00Z'
        )
) AS source
ON target.ACTIVITY_ID = source.ACTIVITY_ID
WHEN MATCHED THEN UPDATE SET
    CUSTOMER_ID = source.CUSTOMER_ID,
    PRODUCT_ID = source.PRODUCT_ID,
    EVENT_TYPE = source.EVENT_TYPE,
    EVENT_DESCRIPTION = source.EVENT_DESCRIPTION,
    EVENT_TIME = TO_TIMESTAMP_NTZ(
        source.EVENT_TIME_TEXT,
        'YYYY-MM-DD"T"HH24:MI:SS"Z"'
    )
WHEN NOT MATCHED THEN INSERT (
    ACTIVITY_ID, CUSTOMER_ID, PRODUCT_ID, EVENT_TYPE,
    EVENT_DESCRIPTION, EVENT_TIME
) VALUES (
    source.ACTIVITY_ID, source.CUSTOMER_ID, source.PRODUCT_ID,
    source.EVENT_TYPE, source.EVENT_DESCRIPTION,
    TO_TIMESTAMP_NTZ(
        source.EVENT_TIME_TEXT,
        'YYYY-MM-DD"T"HH24:MI:SS"Z"'
    )
);

MERGE INTO CASE_HISTORY AS target
USING (
    SELECT
        column1::VARCHAR AS HISTORY_ID,
        column2::VARCHAR AS CASE_ID,
        column3::VARCHAR AS CUSTOMER_ID,
        column4::VARCHAR AS STATUS,
        column5::VARCHAR AS EVENT_TIME_TEXT,
        column6::VARCHAR AS DESCRIPTION
    FROM VALUES
        (
            'HIST-001', 'CASE-4821', 'CUST-1001', 'UNDER_INVESTIGATION',
            '2026-11-05T09:30:00Z',
            'Tier 2 investigation opened after three degraded synchronization windows. Connector retries were observed, but no definitive root cause or resolution date was confirmed.'
        ),
        (
            'HIST-002', 'CASE-4821', 'CUST-1001', 'AWAITING_DIAGNOSTICS',
            '2026-11-06T16:10:00Z',
            'Connector diagnostic logs were requested. Receipt is not recorded. An index integrity check and possible rebuild were scheduled for the November 8 maintenance window, with no completion result recorded.'
        ),
        (
            'HIST-4822-001', 'CASE-4822', 'CUST-1002', 'RESOLVED',
            '2026-11-05T14:15:00Z',
            'Customer Operations verified an authorized administrator, completed the billing-contact update at 14:08 UTC, received customer confirmation at 14:12 UTC, and marked the case Resolved.'
        ),
        (
            'HIST-4823-001', 'CASE-4823', 'CUST-1003', 'AWAITING_CUSTOMER',
            '2026-11-04T16:45:00Z',
            'The documented support attempt did not reproduce the dashboard refresh issue. The application version and affected timestamp were requested and are not recorded as received.'
        ),
        (
            'HIST-4824-001', 'CASE-4824', 'CUST-1002', 'MONITORING',
            '2026-11-02T18:20:00Z',
            'One telemetry export window failed and the following retry completed. The case remains in Monitoring because the Standard-tier escalation threshold was not reached.'
        ),
        (
            'HIST-4825-001', 'CASE-4825', 'CUST-1001', 'RESOLVED',
            '2026-11-01T12:40:00Z',
            'Tier 2 confirmed an expired connector certificate, completed certificate rotation at 11:35 UTC, completed queued export replay at 12:10 UTC, and confirmed backlog clearance at 12:30 UTC.'
        )
) AS source
ON target.HISTORY_ID = source.HISTORY_ID
WHEN MATCHED THEN UPDATE SET
    CASE_ID = source.CASE_ID,
    CUSTOMER_ID = source.CUSTOMER_ID,
    STATUS = source.STATUS,
    EVENT_TIME = TO_TIMESTAMP_NTZ(
        source.EVENT_TIME_TEXT,
        'YYYY-MM-DD"T"HH24:MI:SS"Z"'
    ),
    DESCRIPTION = source.DESCRIPTION
WHEN NOT MATCHED THEN INSERT (
    HISTORY_ID, CASE_ID, CUSTOMER_ID, STATUS, EVENT_TIME, DESCRIPTION
) VALUES (
    source.HISTORY_ID, source.CASE_ID, source.CUSTOMER_ID, source.STATUS,
    TO_TIMESTAMP_NTZ(
        source.EVENT_TIME_TEXT,
        'YYYY-MM-DD"T"HH24:MI:SS"Z"'
    ),
    source.DESCRIPTION
);

MERGE INTO SUPPORT_INTERACTIONS AS target
USING (
    SELECT
        column1::VARCHAR AS INTERACTION_ID,
        column2::VARCHAR AS CASE_ID,
        column3::VARCHAR AS CUSTOMER_ID,
        column4::VARCHAR AS TRANSCRIPT_ID,
        column5::VARCHAR AS INTERACTION_TYPE,
        column6::VARCHAR AS SUMMARY,
        column7::VARCHAR AS EVENT_TIME_TEXT
    FROM VALUES
        (
            'INT-001', 'CASE-4821', 'CUST-1001',
            'CALL-4821-20261105', 'PHONE',
            'Maya reported telemetry arriving forty to sixty minutes late during three evening windows and said the reviewed delayed readings eventually appeared.',
            '2026-11-05T11:00:00Z'
        ),
        (
            'INT-002', 'CASE-4821', 'CUST-1001',
            NULL, 'EMAIL',
            'Support requested connector diagnostic logs for the affected windows. This interaction records the request, not receipt of the logs.',
            '2026-11-06T16:15:00Z'
        ),
        (
            'INT-4822-001', 'CASE-4822', 'CUST-1002',
            NULL, 'EMAIL',
            'Luis confirmed that the corrected billing contact was visible after the authorized administrator verification and completed update.',
            '2026-11-05T14:12:00Z'
        ),
        (
            'INT-4823-001', 'CASE-4823', 'CUST-1003',
            NULL, 'EMAIL',
            'Support asked Priya for the mobile application version and an affected timestamp. No response containing those details is present in the seeded history.',
            '2026-11-04T16:42:00Z'
        ),
        (
            'INT-4824-001', 'CASE-4824', 'CUST-1002',
            NULL, 'PHONE',
            'Luis reported one delayed export window; support advised monitoring after the next retry completed normally.',
            '2026-11-02T18:05:00Z'
        ),
        (
            'INT-4825-001', 'CASE-4825', 'CUST-1001',
            NULL, 'EMAIL',
            'Maya confirmed that telemetry exports were current after certificate rotation and queued replay completed.',
            '2026-11-01T12:35:00Z'
        )
) AS source
ON target.INTERACTION_ID = source.INTERACTION_ID
WHEN MATCHED THEN UPDATE SET
    CASE_ID = source.CASE_ID,
    CUSTOMER_ID = source.CUSTOMER_ID,
    TRANSCRIPT_ID = COALESCE(source.TRANSCRIPT_ID, target.TRANSCRIPT_ID),
    INTERACTION_TYPE = source.INTERACTION_TYPE,
    SUMMARY = source.SUMMARY,
    EVENT_TIME = TO_TIMESTAMP_NTZ(
        source.EVENT_TIME_TEXT,
        'YYYY-MM-DD"T"HH24:MI:SS"Z"'
    )
WHEN NOT MATCHED THEN INSERT (
    INTERACTION_ID, CASE_ID, CUSTOMER_ID, TRANSCRIPT_ID,
    INTERACTION_TYPE, SUMMARY, EVENT_TIME
) VALUES (
    source.INTERACTION_ID, source.CASE_ID, source.CUSTOMER_ID,
    source.TRANSCRIPT_ID, source.INTERACTION_TYPE, source.SUMMARY,
    TO_TIMESTAMP_NTZ(
        source.EVENT_TIME_TEXT,
        'YYYY-MM-DD"T"HH24:MI:SS"Z"'
    )
);

COMMIT;

-- Safe verification queries. Expected seeded row counts: 6 per table.
SELECT COUNT(*) AS SEEDED_CUSTOMER_ACTIVITY_ROWS
FROM CUSTOMER_ACTIVITY
WHERE ACTIVITY_ID IN (
    'ACT-001', 'ACT-002', 'ACT-4822-001', 'ACT-4823-001',
    'ACT-4824-001', 'ACT-4825-001'
);

SELECT COUNT(*) AS SEEDED_CASE_HISTORY_ROWS
FROM CASE_HISTORY
WHERE HISTORY_ID IN (
    'HIST-001', 'HIST-002', 'HIST-4822-001', 'HIST-4823-001',
    'HIST-4824-001', 'HIST-4825-001'
);

SELECT COUNT(*) AS SEEDED_SUPPORT_INTERACTION_ROWS
FROM SUPPORT_INTERACTIONS
WHERE INTERACTION_ID IN (
    'INT-001', 'INT-002', 'INT-4822-001', 'INT-4823-001',
    'INT-4824-001', 'INT-4825-001'
);

-- Detailed validation preserves stable identifiers and exposes every mapped
-- target column without returning any secrets.
SELECT
    ACTIVITY_ID,
    CUSTOMER_ID,
    PRODUCT_ID,
    EVENT_TYPE,
    EVENT_DESCRIPTION,
    EVENT_TIME
FROM CUSTOMER_ACTIVITY
WHERE ACTIVITY_ID IN (
    'ACT-001', 'ACT-002', 'ACT-4822-001', 'ACT-4823-001',
    'ACT-4824-001', 'ACT-4825-001'
)
ORDER BY ACTIVITY_ID;

SELECT
    HISTORY_ID,
    CASE_ID,
    CUSTOMER_ID,
    STATUS,
    DESCRIPTION,
    EVENT_TIME
FROM CASE_HISTORY
WHERE HISTORY_ID IN (
    'HIST-001', 'HIST-002', 'HIST-4822-001', 'HIST-4823-001',
    'HIST-4824-001', 'HIST-4825-001'
)
ORDER BY HISTORY_ID;

SELECT
    INTERACTION_ID,
    CASE_ID,
    CUSTOMER_ID,
    TRANSCRIPT_ID,
    INTERACTION_TYPE,
    SUMMARY,
    EVENT_TIME
FROM SUPPORT_INTERACTIONS
WHERE INTERACTION_ID IN (
    'INT-001', 'INT-002', 'INT-4822-001', 'INT-4823-001',
    'INT-4824-001', 'INT-4825-001'
)
ORDER BY INTERACTION_ID;
