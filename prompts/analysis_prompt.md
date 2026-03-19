# Analysis Prompt Memory

Use the following durable product and research memory in every analysis or classification task:

1. The target users are domestic workers in India.
2. Anyone without a smartphone or without a bank account is non-target.
3. Trust barriers must be considered explicitly.
4. WhatsApp usage should be considered as a relevant behavioral and operational channel.
5. Informal lending is the baseline comparison for user behavior and product value.
6. The response must be structured JSON only.
7. Separate income evidence into per-household earnings, participant personal monthly income, and total household monthly income.
8. Do not treat a single-household wage as total monthly income.
9. If multiple income numbers appear, identify what each number refers to and require direct quote evidence for each captured income figure.
10. Prefer the participant's explicitly stated total monthly income when available, and mark inferred income values as inferred_or_uncertain.

When uncertain, preserve these rules instead of introducing assumptions that conflict with them.
