| Model | Accuracy | Macro-F1 | Invalid outputs | ms / example |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct zero-shot | 15.5% | 14.6% | 35.8% | 88.8 |
| Qwen2.5-0.5B-Instruct + LoRA | 90.8% | 90.9% | 0.1% | 17.2 |
| TF-IDF + logistic regression | 91.0% | 91.0% | 0.0% | 0.1 |

**Qwen2.5-0.5B-Instruct zero-shot**, most frequent errors: exchange_rate → <invalid> (35); receiving_money → <invalid> (33); card_delivery_estimate → <invalid> (29); country_support → <invalid> (29); card_acceptance → <invalid> (28)

**Qwen2.5-0.5B-Instruct + LoRA**, most frequent errors: card_payment_not_recognised → compromised_card (6); why_verify_identity → verify_my_identity (6); balance_not_updated_after_bank_transfer → transfer_not_received_by_recipient (6); card_arrival → card_delivery_estimate (4); pin_blocked → get_physical_card (4)

**TF-IDF + logistic regression**, most frequent errors: top_up_failed → top_up_reverted (5); balance_not_updated_after_bank_transfer → transfer_not_received_by_recipient (5); verify_my_identity → why_verify_identity (4); card_arrival → card_delivery_estimate (3); card_payment_wrong_exchange_rate → wrong_exchange_rate_for_cash_withdrawal (3)
