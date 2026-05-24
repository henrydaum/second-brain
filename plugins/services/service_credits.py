"""Service-loader bridge for public-web credits."""

from billing.credits import CreditsService


def build_services(config: dict) -> dict:
    return {"credits": CreditsService(config or {})}
