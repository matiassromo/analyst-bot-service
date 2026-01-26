"""
Security middleware for IP-based access control.
Validates incoming requests against a whitelist of allowed IPs.
"""

import ipaddress
import logging
from typing import List
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    Middleware to restrict API access to whitelisted IP addresses.

    Supports:
    - Individual IPs (e.g., "192.168.1.100")
    - CIDR notation (e.g., "192.168.1.0/24")
    - X-Forwarded-For header for proxied requests

    If the request IP is not in the whitelist, returns 403 Forbidden.
    """

    def __init__(self, app, allowed_ips: List[str]):
        super().__init__(app)
        self.allowed_networks = self._parse_allowed_ips(allowed_ips)
        logger.info(f"IP whitelist initialized with {len(self.allowed_networks)} entries")

    def _parse_allowed_ips(self, allowed_ips: List[str]) -> List[ipaddress.IPv4Network | ipaddress.IPv6Network]:
        """
        Parse allowed IPs/CIDR notation into network objects.

        Args:
            allowed_ips: List of IP addresses or CIDR ranges

        Returns:
            List of IPv4Network/IPv6Network objects
        """
        networks = []
        for ip_str in allowed_ips:
            try:
                # Try parsing as network (CIDR notation or single IP)
                network = ipaddress.ip_network(ip_str, strict=False)
                networks.append(network)
            except ValueError as e:
                logger.warning(f"Invalid IP/CIDR notation '{ip_str}': {e}")
        return networks

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request.
        Checks X-Forwarded-For header first (for proxied requests),
        then falls back to direct client IP.

        Args:
            request: FastAPI request object

        Returns:
            Client IP address as string
        """
        # Check X-Forwarded-For header (for proxied requests)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs (client, proxy1, proxy2...)
            # The first IP is the original client
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            # Direct connection
            client_ip = request.client.host

        return client_ip

    def _is_ip_allowed(self, ip_str: str) -> bool:
        """
        Check if an IP address is in the whitelist.

        Args:
            ip_str: IP address to check

        Returns:
            True if allowed, False otherwise
        """
        try:
            ip_address = ipaddress.ip_address(ip_str)
            for network in self.allowed_networks:
                if ip_address in network:
                    return True
            return False
        except ValueError:
            logger.error(f"Invalid IP address format: {ip_str}")
            return False

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process each request and validate IP address.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response

        Raises:
            HTTPException: 403 if IP not whitelisted
        """
        # Skip IP check for health/docs endpoints
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        if not self._is_ip_allowed(client_ip):
            logger.warning(
                f"Access denied for IP {client_ip} to {request.url.path}"
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": f"Access forbidden: IP {client_ip} is not whitelisted"}
            )

        logger.debug(f"Allowed IP {client_ip} accessing {request.url.path}")
        response = await call_next(request)
        return response


def get_client_ip(request: Request) -> str:
    """
    Utility function to get client IP from request.
    Can be used as a FastAPI dependency.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host
