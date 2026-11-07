-- Banking77 weight-space-programming (WSP) model, callable from SQL via AI_COMPLETE bring-your-own-model.
-- The service ML_DEMO.BANKING77_WSP.BANKING77_WSP_SVC_Q4 serves LLAMA_3_1_8B_BANKING77_WSP / V_OPUS55_R1_Q4BASE:
-- Llama 3.1 8B Instruct with a closed-form edit to 188 layer-31 MLP neurons (no gradient training), on the exact
-- base the editor fit it on (mlx-community 4-bit, dequantized to fp16).

USE ROLE SYSADMIN;
USE WAREHOUSE BANKING77_WSP_WH;
USE SCHEMA ML_DEMO.BANKING77_WSP;

-- 1. A SQL function: one banking message in, one Banking77 intent label out.
--    Same prompt as the SFT benchmark; AI_COMPLETE returns the reply as a quoted string, so strip the quotes.
CREATE OR REPLACE FUNCTION CLASSIFY_BANKING_INTENT(MESSAGE VARCHAR)
RETURNS VARCHAR
COMMENT = 'Banking77 intent via AI_COMPLETE on the weight-edited Llama 3.1 8B service BANKING77_WSP_SVC_Q4'
AS
$$
    TRIM(AI_COMPLETE(
        'ML_DEMO.BANKING77_WSP.BANKING77_WSP_SVC_Q4',
        'Classify this banking customer message into the correct intent.\n\nMessage: ' || MESSAGE || '\n\nIntent:',
        {'temperature': 0, 'max_tokens': 16}
    ), '" \n')
$$;

-- 2. A simple streaming pipeline: new messages land in INCOMING_MESSAGES, a task classifies them.
CREATE TABLE IF NOT EXISTS INCOMING_MESSAGES (
    MESSAGE_ID   NUMBER AUTOINCREMENT,
    MESSAGE_TEXT VARCHAR,
    RECEIVED_AT  TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);
CREATE TABLE IF NOT EXISTS CLASSIFIED_MESSAGES (
    MESSAGE_ID    NUMBER,
    MESSAGE_TEXT  VARCHAR,
    INTENT        VARCHAR,
    RECEIVED_AT   TIMESTAMP_LTZ,
    CLASSIFIED_AT TIMESTAMP_LTZ
);
CREATE STREAM IF NOT EXISTS INCOMING_MESSAGES_STREAM ON TABLE INCOMING_MESSAGES APPEND_ONLY = TRUE;

CREATE OR REPLACE TASK CLASSIFY_INCOMING_TASK
    WAREHOUSE = BANKING77_WSP_WH
    SCHEDULE = '5 MINUTE'
    COMMENT = 'Classify new banking messages with the weight-edited Llama (AI_COMPLETE BYOM)'
    WHEN SYSTEM$STREAM_HAS_DATA('ML_DEMO.BANKING77_WSP.INCOMING_MESSAGES_STREAM')
AS
    INSERT INTO CLASSIFIED_MESSAGES
    SELECT MESSAGE_ID, MESSAGE_TEXT, CLASSIFY_BANKING_INTENT(MESSAGE_TEXT), RECEIVED_AT, CURRENT_TIMESTAMP()
    FROM INCOMING_MESSAGES_STREAM;
-- Tasks are created suspended. ALTER TASK CLASSIFY_INCOMING_TASK RESUME; to run on schedule
-- (only while BANKING77_WSP_SVC_Q4 is up), or EXECUTE TASK CLASSIFY_INCOMING_TASK; for a one-off run.
