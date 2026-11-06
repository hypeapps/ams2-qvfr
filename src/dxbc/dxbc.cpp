#include "dxbc.h"

#include <cstring>

namespace qvfr::dxbc {
namespace {

uint32_t Rotl(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }

// Standard MD5 block transform (RFC 1321).
void Md5Block(uint32_t s[4], const uint8_t block[64]) {
    static const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
    static const int R[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9,  14, 20, 5, 9,
                              14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                              4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
    uint32_t m[16];
    memcpy(m, block, 64);
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3];
    for (int i = 0; i < 64; ++i) {
        uint32_t f;
        int g;
        if (i < 16) {
            f = (b & c) | (~b & d);
            g = i;
        } else if (i < 32) {
            f = (d & b) | (~d & c);
            g = (5 * i + 1) % 16;
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (3 * i + 5) % 16;
        } else {
            f = c ^ (b | ~d);
            g = (7 * i) % 16;
        }
        const uint32_t t = d;
        d = c;
        c = b;
        b = b + Rotl(a + f + K[i] + m[g], R[i]);
        a = t;
    }
    s[0] += a;
    s[1] += b;
    s[2] += c;
    s[3] += d;
}

void Put32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }
uint32_t Get32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

void SetErr(std::string* err, const char* msg) {
    if (err) *err = msg;
}

}  // namespace

// Same block structure as MD5, but the tail differs: the bit count goes first in the last block (or in a block of
// its own), and the last word is (bits >> 2) | 1. The result is the raw MD5 state.
void Checksum(const void* data, size_t size, uint32_t out[4]) {
    const uint8_t* p = static_cast<const uint8_t*>(data) + 20;
    const uint32_t n = static_cast<uint32_t>(size - 20);
    uint32_t s[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    const uint32_t full = n & ~63u;
    for (uint32_t i = 0; i < full; i += 64) Md5Block(s, p + i);
    const uint32_t last = n - full;
    const uint32_t bits = n * 8;
    uint8_t block[64] = {};
    if (last >= 56) {
        memcpy(block, p + full, last);
        block[last] = 0x80;
        Md5Block(s, block);
        memset(block, 0, sizeof(block));
        Put32(block, bits);
        Put32(block + 60, (bits >> 2) | 1);
        Md5Block(s, block);
    } else {
        Put32(block, bits);
        memcpy(block + 4, p + full, last);
        block[4 + last] = 0x80;
        Put32(block + 60, (bits >> 2) | 1);
        Md5Block(s, block);
    }
    memcpy(out, s, 16);
}

bool ChecksumMatches(const void* data, size_t size) {
    if (size < 32) return false;
    uint32_t h[4];
    Checksum(data, size, h);
    return memcmp(h, static_cast<const uint8_t*>(data) + 4, 16) == 0;
}

bool Parse(const void* data, size_t size, std::vector<Chunk>& chunks, std::string* err) {
    const auto* b = static_cast<const uint8_t*>(data);
    chunks.clear();
    if (size < 32 || Get32(b) != FourCC('D', 'X', 'B', 'C')) return SetErr(err, "not a DXBC container"), false;
    const uint32_t total = Get32(b + 24), count = Get32(b + 28);
    if (total != size || 32 + 4ull * count > size) return SetErr(err, "bad container header"), false;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t off = Get32(b + 32 + 4 * i);
        if (off + 8ull > size) return SetErr(err, "chunk offset out of range"), false;
        const uint32_t len = Get32(b + off + 4);
        if (off + 8ull + len > size) return SetErr(err, "chunk size out of range"), false;
        Chunk c;
        c.fourcc = Get32(b + off);
        c.data.assign(b + off + 8, b + off + 8 + len);
        chunks.push_back(std::move(c));
    }
    return true;
}

std::vector<uint8_t> Build(const std::vector<Chunk>& chunks) {
    size_t total = 32 + 4 * chunks.size();
    for (const auto& c : chunks) total += 8 + c.data.size();
    std::vector<uint8_t> out(total, 0);
    Put32(out.data(), FourCC('D', 'X', 'B', 'C'));
    Put32(out.data() + 20, 1);  // container version
    Put32(out.data() + 24, static_cast<uint32_t>(total));
    Put32(out.data() + 28, static_cast<uint32_t>(chunks.size()));
    size_t off = 32 + 4 * chunks.size();
    for (size_t i = 0; i < chunks.size(); ++i) {
        Put32(out.data() + 32 + 4 * i, static_cast<uint32_t>(off));
        Put32(out.data() + off, chunks[i].fourcc);
        Put32(out.data() + off + 4, static_cast<uint32_t>(chunks[i].data.size()));
        if (!chunks[i].data.empty()) memcpy(out.data() + off + 8, chunks[i].data.data(), chunks[i].data.size());
        off += 8 + chunks[i].data.size();
    }
    uint32_t h[4];
    Checksum(out.data(), out.size(), h);
    memcpy(out.data() + 4, h, 16);
    return out;
}

Chunk* Find(std::vector<Chunk>& chunks, uint32_t fourcc) {
    for (auto& c : chunks)
        if (c.fourcc == fourcc) return &c;
    return nullptr;
}

bool ReadSignature(const Chunk& chunk, std::vector<SigElement>& out, std::string* err) {
    const auto& d = chunk.data;
    out.clear();
    if (d.size() < 8) return SetErr(err, "signature chunk too small"), false;
    const uint32_t count = Get32(d.data()), first = Get32(d.data() + 4);
    if (first + 24ull * count > d.size()) return SetErr(err, "signature elements out of range"), false;
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* e = d.data() + first + 24 * i;
        SigElement s;
        const uint32_t nameOff = Get32(e);
        if (nameOff >= d.size()) return SetErr(err, "signature name out of range"), false;
        const auto* name = reinterpret_cast<const char*>(d.data() + nameOff);
        s.name.assign(name, strnlen(name, d.size() - nameOff));
        s.semanticIndex = Get32(e + 4);
        s.systemValue = Get32(e + 8);
        s.componentType = Get32(e + 12);
        s.reg = Get32(e + 16);
        s.mask = e[20];
        s.rwMask = e[21];
        out.push_back(std::move(s));
    }
    return true;
}

Chunk WriteSignature(uint32_t fourcc, const std::vector<SigElement>& elements) {
    Chunk c;
    c.fourcc = fourcc;
    const size_t strBase = 8 + 24 * elements.size();
    std::vector<uint8_t> strings;
    std::vector<uint32_t> nameOff;
    for (const auto& e : elements) {
        // Elements with the same name share one string, as the compiler does.
        uint32_t off = 0;
        bool found = false;
        for (size_t j = 0; j < nameOff.size(); ++j)
            if (elements[j].name == e.name) {
                off = nameOff[j];
                found = true;
                break;
            }
        if (!found) {
            off = static_cast<uint32_t>(strBase + strings.size());
            strings.insert(strings.end(), e.name.begin(), e.name.end());
            strings.push_back(0);
        }
        nameOff.push_back(off);
    }
    while (strings.size() % 4) strings.push_back(0xAB);  // the compiler pads with 0xAB
    c.data.assign(strBase + strings.size(), 0);
    Put32(c.data.data(), static_cast<uint32_t>(elements.size()));
    Put32(c.data.data() + 4, 8);
    for (size_t i = 0; i < elements.size(); ++i) {
        uint8_t* e = c.data.data() + 8 + 24 * i;
        Put32(e, nameOff[i]);
        Put32(e + 4, elements[i].semanticIndex);
        Put32(e + 8, elements[i].systemValue);
        Put32(e + 12, elements[i].componentType);
        Put32(e + 16, elements[i].reg);
        e[20] = elements[i].mask;
        e[21] = elements[i].rwMask;
    }
    if (!strings.empty()) memcpy(c.data.data() + strBase, strings.data(), strings.size());
    return c;
}

}  // namespace qvfr::dxbc
