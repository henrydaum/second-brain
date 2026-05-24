"""Thin wrapper around the Stripe SDK so frontend_web.py stays readable.

Stripe keys are read from frontend config; the pack amount/price comes from
the core web_credits policy.

If the stripe SDK isn't installed or keys are missing, these functions raise
RuntimeError — the frontend converts that into a user-visible error.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("stripe_client")


def _stripe(secret_key: str):
    if not secret_key:
        raise RuntimeError("Stripe is not configured (missing stripe_secret_key).")
    try:
        import stripe  # type: ignore
    except ImportError as e:
        raise RuntimeError("stripe package not installed (pip install stripe).") from e
    stripe.api_key = secret_key
    return stripe


def create_checkout_session(
    secret_key: str,
    price_id: str,
    price_cents: int,
    success_url: str,
    cancel_url: str,
    email_hint: str | None = None,
    metadata: dict | None = None,
) -> dict:
    stripe = _stripe(secret_key)
    item = {"price": price_id, "quantity": 1} if price_id else {
        "price_data": {"currency": "usd", "unit_amount": int(price_cents), "product_data": {"name": "Second Brain Credits"}},
        "quantity": 1,
    }
    kwargs = dict(
        mode="payment",
        line_items=[item],
        success_url=success_url,
        cancel_url=cancel_url,
        allow_promotion_codes=False,
        metadata=metadata or {},
    )
    if email_hint:
        kwargs["customer_email"] = email_hint
    session = stripe.checkout.Session.create(**kwargs)
    return {"id": session.id, "url": session.url}


def verify_webhook(secret_key: str, webhook_secret: str, payload: bytes, sig_header: str) -> dict:
    """Verify a Stripe webhook signature and return the event dict.
    Raises RuntimeError on invalid signature or missing config."""
    stripe = _stripe(secret_key)
    if not webhook_secret:
        raise RuntimeError("Stripe webhook secret is not configured.")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except Exception as e:
        raise RuntimeError(f"Invalid Stripe webhook signature: {e}") from e
    return event
