import os
import sys

# Add parent directory to path so mas can be resolved
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mas.tools import (
    get_policy_details,
    get_claim_status,
    get_billing_info,
    get_payment_history,
    get_auto_policy_details,
    set_db_path
)

print("Starting Direct Tool Tests...\n")
set_db_path("mas/insurance_support.db")

print("1. get_policy_details:")
try:
    print(get_policy_details("POL-123456"))
except Exception as e:
    print("Error:", e)

print("\n2. get_claim_status:")
try:
    print(get_claim_status(policy_number="POL-123456"))
except Exception as e:
    print("Error:", e)

print("\n3. get_billing_info:")
try:
    print(get_billing_info(policy_number="POL-123456"))
except Exception as e:
    print("Error:", e)

print("\n4. get_payment_history:")
try:
    print(get_payment_history("POL-123456"))
except Exception as e:
    print("Error:", e)

print("\n5. get_auto_policy_details:")
try:
    print(get_auto_policy_details("POL-123456"))
except Exception as e:
    print("Error:", e)

print("\nTests completed.")
