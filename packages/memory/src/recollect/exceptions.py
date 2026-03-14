"""Exception hierarchy for Memory SDK."""


class MemorySDKError(Exception):
    """Base exception for Memory SDK."""


class StorageError(MemorySDKError):
    """Raised when storage operations fail."""


class EmbeddingError(MemorySDKError):
    """Raised when embedding generation fails."""


class ExtractionError(MemorySDKError):
    """Raised when pattern extraction fails."""


class ConsolidationError(MemorySDKError):
    """Raised when memory consolidation fails."""


class TraceNotFoundError(MemorySDKError):
    """Raised when a memory trace is not found."""


class SessionNotFoundError(MemorySDKError):
    """Raised when a session is not found."""
