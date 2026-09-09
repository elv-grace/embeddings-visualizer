#!/usr/bin/env python3
"""Decode a fabric auth token's claims, locally and offline.

A token is `<6-char prefix><base58(signature || deflateRaw(json))>`, per the
worked example in elv-client-js's `ElvClient.CreateFabricToken` docstring. The
prefix encodes the scheme (`acspjc` = account/space, `aessjc` = signed,
`aplsjc` = plain; `jc` = json-compressed).

Only the claims are read -- the signature is skipped, not verified. Useful for
answering "which account signed this?" when a 403 could be either the wrong
account or the wrong content space.

    python3 tools/decode_token.py <token>
    python3 tools/decode_token.py --file /path/to/token.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# secp256k1 recoverable signature: r(32) + s(32) + v(1).
SIGNATURE_BYTES = 65


def b58decode(text: str) -> bytes:
    value = 0
    for char in text:
        index = B58.find(char)
        if index < 0:
            raise ValueError(f"{char!r} is not a base58 character")
        value = value * 58 + index
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    # Leading '1's are leading zero bytes, which the integer form drops.
    return b"\x00" * (len(text) - len(text.lstrip("1"))) + raw


def decode(token: str) -> tuple[str, dict]:
    token = token.strip()
    prefix, body = token[:6], token[6:]
    raw = b58decode(body)
    payload = raw[SIGNATURE_BYTES:]
    try:
        # -15 = raw deflate, no zlib header (Pako.deflateRaw).
        claims = zlib.decompressobj(-15).decompress(payload)
    except zlib.error:
        claims = payload  # some schemes carry plain json
    return prefix, json.loads(claims)


def address_of(claims: dict) -> str:
    """The signing account as 0x…, from the base64 `adr` claim."""
    import base64

    adr = claims.get("adr")
    if not adr:
        return "(no adr claim)"
    return "0x" + base64.b64decode(adr).hex()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token", nargs="?", help="the token string")
    parser.add_argument("--file", help="read the token from a file instead")
    args = parser.parse_args()

    text = open(args.file).read() if args.file else args.token
    if not text:
        parser.error("give a token or --file")

    prefix, claims = decode(text)
    print(f"prefix  : {prefix}")
    print(f"address : {address_of(claims)}")
    print(f"space   : {claims.get('spc')}")
    print(f"subject : {claims.get('sub')}")
    for key in ("qid", "lib", "gra", "iat", "exp"):
        if key in claims:
            print(f"{key:<8}: {claims[key]}")
    print("\nall claims:")
    print(json.dumps(claims, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
