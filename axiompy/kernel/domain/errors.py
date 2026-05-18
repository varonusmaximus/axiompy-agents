"""Kernel error hierarchy."""


class KernelError(Exception):
    """Base exception for kernel errors."""

    pass


class KernelConfigurationError(KernelError):
    """Invalid kernel configuration."""

    pass


class KernelRuntimeError(KernelError):
    """Error during agent execution."""

    pass


class KernelToolError(KernelError):
    """Tool invocation failed."""

    pass
