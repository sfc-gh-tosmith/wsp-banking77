"""Shared constants and helpers for the Banking77 weight-space-programming project.

Prompt and scoring are byte-for-byte the SFT benchmark's
(SFT-benchmark-banking77/sql/benchmark.sql and monitor_eval.py).
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

# Name of a connection in your ~/.snowflake/connections.toml (or config.toml).
SF_CONNECTION = os.environ.get("WSP_SF_CONNECTION", "default")
SF_DB, SF_SCHEMA, SF_WH = "ML_DEMO", "BANKING77_WSP", "BANKING77_WSP_WH"

MODEL_4BIT = MODELS / "llama31-8b-instruct-4bit"
MODEL_BF16 = MODELS / "llama31-8b-instruct-bf16"

LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working", "country_support",
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card",
    "get_physical_card", "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal",
]
assert len(LABELS) == 77

# SFT prompt (the SQL literal '\n' in SNOWFLAKE.CORTEX.COMPLETE is a real newline).
SFT_PROMPT = "Classify this banking customer message into the correct intent.\n\nMessage: {text}\n\nIntent:"

# The SFT benchmark's zero-shot frontier prompt (labels joined with '; ' in alphabetical order, as LISTAGG ... ORDER BY LABEL).
ZERO_SHOT_PROMPT = (
    "You are a banking customer-support intent classifier. Classify the customer message into EXACTLY ONE of "
    "these intents. Respond with only the exact intent label, nothing else.\n\nIntents: {labels}\n\n"
    "Message: {text}\n\nIntent:"
)


def norm(s: str) -> str:
    """SFT scoring normalisation: REPLACE(LOWER(TRIM(x)), ' ', '')."""
    return (s or "").strip().lower().replace(" ", "")


def sf_connect():
    """Connector to the Snowflake account that holds ML_DEMO.BANKING77_WSP.

    authenticator='snowflake' overrides a stale authenticator in a local config.toml; change it to match yours."""
    import snowflake.connector
    return snowflake.connector.connect(
        connection_name=SF_CONNECTION, authenticator="snowflake",
        database=SF_DB, schema=SF_SCHEMA, warehouse=SF_WH,
    )
