# Error analysis: lora vs baseline (3080 test messages)

| | |
|---|---|
| Accuracy lora | 90.8% |
| Accuracy baseline | 91.0% |
| Same prediction | 89.2% |
| Accuracy when they agree | 97.3% |
| Both wrong | 4.9% |
| Oracle (either right) | 95.1% |

## Intents where baseline is better
| Intent | F1 lora | F1 baseline |
|---|---|---|
| topping_up_by_card | 0.70 | 0.85 |
| reverted_card_payment? | 0.78 | 0.93 |
| card_payment_not_recognised | 0.74 | 0.86 |
| transfer_not_received_by_recipient | 0.74 | 0.84 |
| beneficiary_not_allowed | 0.86 | 0.96 |

## Intents where lora is better
| Intent | F1 lora | F1 baseline |
|---|---|---|
| top_up_failed | 0.90 | 0.76 |
| exchange_charge | 0.96 | 0.85 |
| unable_to_verify_identity | 0.95 | 0.85 |
| extra_charge_on_statement | 0.95 | 0.85 |
| wrong_exchange_rate_for_cash_withdrawal | 0.91 | 0.82 |
