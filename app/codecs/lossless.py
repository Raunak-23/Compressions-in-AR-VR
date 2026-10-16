"""Lossless codecs: RLE, Huffman, and the stdlib dictionary coders
(zlib/DEFLATE, LZMA) as reference points.

All operate on raw byte strings -- in the experiments these are the raw RGB
bytes of an XR frame or PCM audio samples."""

import heapq
import json
import lzma
import zlib
from collections import Counter
from itertools import groupby


def compression_ratio(original_len: int, compressed_len: int) -> float:
    """ratio > 1 means the payload shrank."""
    return original_len / compressed_len if compressed_len else float("inf")


# --- Run-Length Encoding: (count, value) byte pairs --------------------------

def rle_encode(data: bytes) -> bytes:
    out = bytearray()
    for value, group in groupby(data):
        run = len(list(group))
        while run > 255:
            out += bytes((255, value))
            run -= 255
        out += bytes((run, value))
    return bytes(out)


def rle_decode(blob: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(blob) - 1, 2):
        out += bytes((blob[i + 1],)) * blob[i]
    return bytes(out)


# --- Huffman coding: table header + bit-packed payload -----------------------

def _build_codes(freq: Counter) -> dict[int, str]:
    heap: list = []
    tiebreak = 0
    for sym, f in freq.items():
        heap.append((f, tiebreak, sym))
        tiebreak += 1
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, tiebreak, (n1, n2)))
        tiebreak += 1

    codes: dict[int, str] = {}

    def walk(node, prefix):
        if isinstance(node, int):
            codes[node] = prefix or "0"
            return
        walk(node[0], prefix + "0")
        walk(node[1], prefix + "1")

    walk(heap[0][2], "")
    return codes


def huffman_encode(data: bytes) -> bytes:
    if not data:
        return b""
    codes = _build_codes(Counter(data))
    header = json.dumps({str(s): c for s, c in codes.items()}).encode("ascii")
    bits = "".join(map(codes.__getitem__, data))
    pad = (-len(bits)) % 8
    body = int(bits + "0" * pad, 2).to_bytes((len(bits) + pad) // 8, "big")
    return len(header).to_bytes(4, "big") + header + bytes((pad,)) + body


def huffman_decode(blob: bytes) -> bytes:
    if not blob:
        return b""
    hlen = int.from_bytes(blob[:4], "big")
    decode_map = {code: int(sym) for sym, code in json.loads(blob[4 : 4 + hlen]).items()}
    pad = blob[4 + hlen]
    body = blob[5 + hlen :]
    nbits = len(body) * 8 - pad
    bits = bin(int.from_bytes(body, "big"))[2:].zfill(len(body) * 8)[:nbits]
    out = bytearray()
    code = ""
    for bit in bits:
        code += bit
        sym = decode_map.get(code)
        if sym is not None:
            out.append(sym)
            code = ""
    return bytes(out)


# --- Stdlib dictionary coders as the reference points ------------------------

def zlib_compress(data: bytes) -> bytes:
    return zlib.compress(data, 9)


def lzma_compress(data: bytes) -> bytes:
    return lzma.compress(data, preset=6)
