// DXBC container helpers: chunk parsing and rebuilding, the container checksum and signature chunks.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qvfr::dxbc {

constexpr uint32_t FourCC(char a, char b, char c, char d) {
    return static_cast<uint32_t>(static_cast<uint8_t>(a)) | (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) | (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

// The container checksum: a variant of MD5 over everything after the checksum field (offset 20).
void Checksum(const void* data, size_t size, uint32_t out[4]);
bool ChecksumMatches(const void* data, size_t size);

struct Chunk {
    uint32_t fourcc = 0;
    std::vector<uint8_t> data;
};

bool Parse(const void* data, size_t size, std::vector<Chunk>& chunks, std::string* err);
// Builds a container from chunks in the given order and signs it.
std::vector<uint8_t> Build(const std::vector<Chunk>& chunks);

Chunk* Find(std::vector<Chunk>& chunks, uint32_t fourcc);

// One element of an ISGN/OSGN chunk (24-byte form).
struct SigElement {
    std::string name;
    uint32_t semanticIndex = 0;
    uint32_t systemValue = 0;    // D3D_NAME
    uint32_t componentType = 0;  // D3D_REGISTER_COMPONENT_TYPE
    uint32_t reg = 0;
    uint8_t mask = 0;
    uint8_t rwMask = 0;  // inputs: components used; outputs: components never written
};

bool ReadSignature(const Chunk& chunk, std::vector<SigElement>& out, std::string* err);
Chunk WriteSignature(uint32_t fourcc, const std::vector<SigElement>& elements);

}  // namespace qvfr::dxbc
