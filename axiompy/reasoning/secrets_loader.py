"""Resolve secrets at the composition root (explicit injection, not factory-hidden)."""

from __future__ import annotations

from typing import Optional

from axiompy.secrets import LocalSettings, SecretsClientFactory, SecretsClientType


def resolve_secret(
    key: str,
    *,
    fallback: Optional[str] = None,
    env_file: str = ".env",
) -> Optional[str]:
    """
    Load a secret value by key from the local secrets backend.

    Args:
        key: Secret name (e.g. ``OPENAI_API_KEY``)
        fallback: Value when the secret cannot be resolved
        env_file: Path to ``.env`` for :class:`LocalSettings`

    Returns:
        Secret string, or ``fallback`` when unavailable
    """
    settings = LocalSettings(env_file=env_file)
    client_result = SecretsClientFactory.create(SecretsClientType.LOCAL, settings)
    if client_result.is_err():
        return fallback

    secret_result = client_result.unwrap().get_secret(key)
    if secret_result.is_err():
        return fallback

    value = secret_result.unwrap()
    return value if value else fallback
