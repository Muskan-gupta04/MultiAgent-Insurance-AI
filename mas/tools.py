import sqlite3
from typing import Any, Dict, List

from . import resources


DB_PATH = "insurance_support.db"


def set_db_path(path: str):
    global DB_PATH
    DB_PATH = path


def ask_user(question: str, missing_info: str = "") -> Dict[str, Any]:
    """
    DEPRECATED: This function uses blocking terminal input(). 
    The system now uses asynchronous state-based clarification.
    """
    resources.logger.warning(f"Blocking ask_user called with: {question}. Returning empty response.")
    return {"context": "PENDING_ASYNCHRONOUS_CLARIFICATION", "source": "System"}


def get_policy_details(policy_number: str) -> Dict[str, Any]:
    resources.logger.info(f"Fetching policy details for: {policy_number}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.*, c.first_name, c.last_name
        FROM policies p
        JOIN customers c ON p.customer_id = c.customer_id
        WHERE p.policy_number = ?
        """,
        (policy_number,),
    )
    result = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    conn.close()
    if result:
        resources.logger.info(f"Policy found: {policy_number}")
        return dict(zip(columns, result))
    resources.logger.warning(f"Policy not found: {policy_number}")
    return {"error": "Policy not found"}


def get_claim_status(claim_id: str = None, policy_number: str = None) -> Dict[str, Any]:
    resources.logger.info(f"Fetching claim status - Claim ID: {claim_id}, Policy: {policy_number}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if claim_id:
        cursor.execute(
            """
            SELECT c.*, p.policy_type
            FROM claims c
            JOIN policies p ON c.policy_number = p.policy_number
            WHERE c.claim_id = ?
            """,
            (claim_id,),
        )
        result = cursor.fetchall()
    elif policy_number:
        cursor.execute(
            """
            SELECT c.*, p.policy_type
            FROM claims c
            JOIN policies p ON c.policy_number = p.policy_number
            WHERE c.policy_number = ?
            ORDER BY c.claim_date DESC LIMIT 3
            """,
            (policy_number,),
        )
        result = cursor.fetchall()
    else:
        result = []

    columns = [desc[0] for desc in cursor.description] if result else []
    conn.close()
    if result:
        resources.logger.info(f"Found {len(result)} claim(s)")
        return [dict(zip(columns, row)) for row in result]
    resources.logger.warning("No claims found")
    return {"error": "Claim not found"}


def get_billing_info(policy_number: str = None, customer_id: str = None) -> Dict[str, Any]:
    resources.logger.info(f"Fetching billing info - Policy: {policy_number}, Customer: {customer_id}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if policy_number:
        cursor.execute(
            """
            SELECT b.*, p.premium_amount, p.billing_frequency
            FROM billing b
            JOIN policies p ON b.policy_number = p.policy_number
            WHERE b.policy_number = ? AND b.status = 'pending'
            ORDER BY b.due_date DESC LIMIT 1
            """,
            (policy_number,),
        )
        result = cursor.fetchone()
    elif customer_id:
        cursor.execute(
            """
            SELECT b.*, p.premium_amount, p.billing_frequency
            FROM billing b
            JOIN policies p ON b.policy_number = p.policy_number
            WHERE p.customer_id = ? AND b.status = 'pending'
            ORDER BY b.due_date DESC LIMIT 1
            """,
            (customer_id,),
        )
        result = cursor.fetchone()
    else:
        result = None

    columns = [desc[0] for desc in cursor.description] if result else []
    conn.close()
    if result:
        resources.logger.info("Billing info found")
        return dict(zip(columns, result))
    resources.logger.warning("Billing info not found")
    return {"error": "Billing information not found"}


def get_payment_history(policy_number: str) -> List[Dict[str, Any]]:
    resources.logger.info(f"Fetching payment history for policy: {policy_number}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.payment_date, p.amount, p.status, p.payment_method
        FROM payments p
        JOIN billing b ON p.bill_id = b.bill_id
        WHERE b.policy_number = ?
        ORDER BY p.payment_date DESC LIMIT 10
        """,
        (policy_number,),
    )
    results = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description] if results else []
    conn.close()
    if results:
        resources.logger.info(f"Found {len(results)} payment records")
        return [dict(zip(columns, row)) for row in results]
    resources.logger.warning("No payment history found")
    return []


def get_auto_policy_details(policy_number: str) -> Dict[str, Any]:
    resources.logger.info(f"Fetching auto policy details for: {policy_number}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT apd.*, p.policy_type, p.premium_amount
        FROM auto_policy_details apd
        JOIN policies p ON apd.policy_number = p.policy_number
        WHERE apd.policy_number = ?
        """,
        (policy_number,),
    )
    result = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description] if result else []
    conn.close()
    if result:
        resources.logger.info("Auto policy details found")
        return dict(zip(columns, result))
    resources.logger.warning("Auto policy details not found")
    return {"error": "Auto policy details not found"}
