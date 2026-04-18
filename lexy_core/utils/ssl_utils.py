"""
Lexy AI - Self-signed certificate generator.

When ``config.server.ssl_enabled`` is true and the configured certfile/
keyfile don't exist yet, ``ensure_cert()`` creates a self-signed pair so
browsers can treat the backend as a secure context. This is specifically
needed for ``getUserMedia`` (mic access) from Firefox/Chrome on a
non-localhost device — plain HTTP is blocked for microphone access
everywhere except the loopback address.

The generated cert is valid for:

* ``localhost`` + ``127.0.0.1`` + ``::1`` (always)
* every IPv4 address detected on the local host (so LAN access works)
* 5 years (plenty for a local dev cert)

``cryptography`` is a standard dependency via FastAPI/chromadb's graph,
so no extra install is usually needed.
"""

from __future__ import annotations

import datetime
import ipaddress
import socket
from pathlib import Path

from lexy_core.utils.logging import get_logger

log = get_logger(module="ssl_utils")


def _local_ip_addresses() -> list[str]:
    """Best-effort discovery of local IPs to bake into the cert SAN."""
    addresses: set[str] = {"127.0.0.1"}
    try:
        hostname = socket.gethostname()
        addresses.add(socket.gethostbyname(hostname))
        for info in socket.getaddrinfo(hostname, None):
            family, _type, _proto, _canon, sockaddr = info
            if family == socket.AF_INET:
                addresses.add(str(sockaddr[0]))
    except OSError as exc:
        log.debug("ssl.local_ip_probe_failed", error=str(exc))
    # Also try the "unreachable sendto" trick for the default route
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            addresses.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(addresses)


def ensure_cert(certfile: str | Path, keyfile: str | Path) -> tuple[Path, Path]:
    """
    Make sure ``certfile`` + ``keyfile`` exist. If either is missing,
    generate a new self-signed pair on the fly. Returns absolute Paths
    to the files as they exist on disk after the call.
    """
    cert_path = Path(certfile).resolve()
    key_path = Path(keyfile).resolve()
    if cert_path.exists() and key_path.exists():
        log.info("ssl.cert_exists", cert=str(cert_path))
        return cert_path, key_path

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:  # pragma: no cover - cryptography is a hard dep via FastAPI graph
        raise RuntimeError(
            "cryptography library required for self-signed cert generation. "
            "Install with: pip install cryptography"
        ) from exc

    log.info("ssl.generating_cert", cert=str(cert_path))

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "DE"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Lexy AI"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Lexy AI local"),
        ]
    )

    san_entries: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
    ]
    for addr in _local_ip_addresses():
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(addr)))
        except ValueError:
            san_entries.append(x509.DNSName(addr))
    try:
        san_entries.append(x509.IPAddress(ipaddress.ip_address("::1")))
    except ValueError:
        pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365 * 5))
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(private_key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    log.info(
        "ssl.cert_written",
        cert=str(cert_path),
        key=str(key_path),
        san=[str(s) for s in san_entries],
    )
    return cert_path, key_path
