#include "mvr_patch.h"

#include <algorithm>
#include <cstring>

#include "dxbc.h"

namespace qvfr::dxbc {
namespace {

// Shader model 4/5 token format (d3d11TokenizedProgramFormat.hpp).
enum : uint32_t {
    kOpDp4 = 0x11,
    kOpCustomData = 0x35,
    kOpMov = 0x36,
    kOpRet = 0x3E,
    kOpRetc = 0x3F,
    kOpDclConstantBuffer = 0x59,
    kOpDclOutput = 0x65,
    kOpDclOutputSgv = 0x66,
    kOpDclOutputSiv = 0x67,
    kOpDclTemps = 0x68,
    kOpDclGlobalFlags = 0x6A,
    kOpInterfaceCall = 0x89,
};
enum : uint32_t { kOperandTemp = 0, kOperandOutput = 2, kOperandImm32 = 4, kOperandImm64 = 5 };
constexpr uint32_t kVertexShader = 1;
constexpr uint32_t kNameUndefined = 0, kNamePosition = 1, kTypeFloat32 = 3;

// Operand tokens used by the inserted code.
constexpr uint32_t kDstOutputXyzw = 0x001020F2;  // o#.xyzw, 1D immediate index
constexpr uint32_t kSrcTempXyzw = 0x00100E46;    // r#.xyzw
constexpr uint32_t kSrcCbXyzw = 0x00208E46;      // cb#[#].xyzw, 2D immediate index
constexpr uint32_t DstOutputComp(uint32_t c) { return 0x00102002 | ((1u << c) << 4); }  // o#.x / .y / .z / .w

uint32_t Opcode(uint32_t tok) { return tok & 0x7FF; }

bool IsDeclaration(uint32_t op) { return (op >= 0x58 && op <= 0x6A) || (op >= 0x8F && op <= 0xA2) || op == 0xCE; }

uint32_t InstrLength(const uint32_t* t, size_t avail) {
    if (Opcode(t[0]) == kOpCustomData) return avail >= 2 ? t[1] : 0;
    uint32_t len = (t[0] >> 24) & 0x7F;
    if (len == 0 && avail >= 2) len = t[1];  // long declarations keep the length in the next token
    return len;
}

struct Walker {
    uint32_t posReg = 0, tempReg = 0, rewritten = 0;
    bool error = false;

    // Returns the operand length in tokens; rewrites writes to the position output to the temp register.
    size_t Operand(uint32_t* t, size_t avail) {
        if (avail == 0) return Fail();
        const uint32_t tok = t[0];
        size_t n = 1;
        for (bool ext = (tok >> 31) != 0; ext; ++n) {
            if (n >= avail) return Fail();
            ext = (t[n] >> 31) != 0;
        }
        const uint32_t type = (tok >> 12) & 0xFF;
        const uint32_t comps = tok & 3;
        const uint32_t dim = (tok >> 20) & 3;
        if (type == kOperandOutput && dim == 1) {
            const uint32_t rep = (tok >> 22) & 7;
            if (rep != 0) return Fail();  // relatively indexed output: not handled
            if (n >= avail) return Fail();
            if (t[n] == posReg) {
                t[0] = (tok & ~(0xFFu << 12)) | (kOperandTemp << 12);
                t[n] = tempReg;
                ++rewritten;
            }
            return n + 1;
        }
        if (type == kOperandImm32) n += comps == 2 ? 4 : 1;
        if (type == kOperandImm64) n += comps == 2 ? 8 : 2;
        for (uint32_t d = 0; d < dim; ++d) {
            const uint32_t rep = (tok >> (22 + 3 * d)) & 7;
            switch (rep) {
                case 0: n += 1; break;
                case 1: n += 2; break;
                case 2: n += Sub(t, n, avail); break;
                case 3: n += 1; n += Sub(t, n, avail); break;
                case 4: n += 2; n += Sub(t, n, avail); break;
                default: return Fail();
            }
        }
        return n <= avail ? n : Fail();
    }

    size_t Sub(uint32_t* t, size_t at, size_t avail) { return at < avail ? Operand(t + at, avail - at) : Fail(); }
    size_t Fail() {
        error = true;
        return 1u << 30;
    }
};

void SetErr(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

}  // namespace

std::string MvrPositionSemantic(uint32_t view) { return "NV_POSITION_VIEW_" + std::to_string(view) + "_SEMANTIC"; }

bool PatchVsForMvr(const void* code, size_t size, const MvrVsPatchOptions& opt, std::vector<uint8_t>& out,
                   MvrVsPatchInfo* info, std::string* err) {
    if (opt.extraViews < 1 || opt.extraViews > 3) return SetErr(err, "extraViews must be 1..3"), false;
    if (opt.cbSlot > 13) return SetErr(err, "cbSlot must be 0..13"), false;
    std::vector<Chunk> chunks;
    if (!Parse(code, size, chunks, err)) return false;
    Chunk* shex = Find(chunks, FourCC('S', 'H', 'E', 'X'));
    if (!shex) shex = Find(chunks, FourCC('S', 'H', 'D', 'R'));
    Chunk* osgn = Find(chunks, FourCC('O', 'S', 'G', 'N'));
    if (!shex) return SetErr(err, "no SHEX/SHDR chunk"), false;
    if (!osgn) return SetErr(err, "no OSGN chunk (OSG1/OSG5 not supported)"), false;

    // Output signature: find the position register and append the extra views after the last output register.
    std::vector<SigElement> sig;
    if (!ReadSignature(*osgn, sig, err)) return false;
    MvrVsPatchInfo inf;
    bool havePos = false;
    uint32_t maxReg = 0;
    for (const auto& e : sig) {
        if (e.systemValue == kNamePosition) {
            if (havePos) return SetErr(err, "more than one SV_Position output"), false;
            havePos = true;
            inf.positionReg = e.reg;
        }
        if (e.name.rfind("NV_", 0) == 0) return SetErr(err, "shader already has NV outputs"), false;
        maxReg = std::max(maxReg, e.reg);
    }
    if (!havePos) return SetErr(err, "no SV_Position output"), false;
    inf.firstNewOutputReg = maxReg + 1;
    if (inf.firstNewOutputReg + opt.extraViews > 32) return SetErr(err, "no free output registers"), false;
    for (uint32_t v = 1; v <= opt.extraViews; ++v) {
        SigElement e;
        e.name = MvrPositionSemantic(v);
        e.systemValue = kNameUndefined;
        e.componentType = kTypeFloat32;
        e.reg = inf.firstNewOutputReg + v - 1;
        e.mask = 0xF;
        e.rwMask = 0;
        sig.push_back(e);
    }

    // Program tokens.
    if (shex->data.size() % 4 || shex->data.size() < 8) return SetErr(err, "bad program chunk size"), false;
    std::vector<uint32_t> in(shex->data.size() / 4);
    memcpy(in.data(), shex->data.data(), shex->data.size());
    if ((in[0] >> 16) != kVertexShader) return SetErr(err, "not a vertex shader"), false;
    if (in[1] != in.size()) return SetErr(err, "program length does not match the chunk"), false;

    // Pass 1: declarations (insertion points, temp count, constant buffer slot in use).
    size_t lastCb = 0, lastOutput = 0, globalFlagsEnd = 2, tempsAt = 0, firstInstr = 0;
    bool haveTemps = false;
    for (size_t pc = 2; pc < in.size();) {
        const uint32_t op = Opcode(in[pc]);
        const uint32_t len = InstrLength(&in[pc], in.size() - pc);
        if (len == 0 || pc + len > in.size()) return SetErr(err, "bad instruction length"), false;
        if (!IsDeclaration(op) && op != kOpCustomData) {
            firstInstr = pc;
            break;
        }
        if (op == kOpDclConstantBuffer) {
            if (len < 4) return SetErr(err, "bad constant buffer declaration"), false;
            if (in[pc + 2] == opt.cbSlot) return SetErr(err, "constant buffer slot already used"), false;
            lastCb = pc + len;
        } else if (op == kOpDclOutput || op == kOpDclOutputSgv || op == kOpDclOutputSiv) {
            lastOutput = pc + len;
        } else if (op == kOpDclTemps) {
            haveTemps = true;
            tempsAt = pc;
            inf.tempReg = in[pc + 1];
        } else if (op == kOpDclGlobalFlags) {
            globalFlagsEnd = pc + len;
        }
        pc += len;
    }
    if (!firstInstr) return SetErr(err, "no instructions"), false;
    if (!lastOutput) return SetErr(err, "no output declarations"), false;
    if (!haveTemps) inf.tempReg = 0;
    const size_t cbInsert = lastCb ? lastCb : globalFlagsEnd;

    const uint32_t cbDecl[4] = {kOpDclConstantBuffer | (4u << 24), kSrcCbXyzw, opt.cbSlot, 4 * opt.extraViews};
    std::vector<uint32_t> outDecls;
    for (uint32_t v = 0; v < opt.extraViews; ++v)
        outDecls.insert(outDecls.end(), {kOpDclOutput | (3u << 24), kDstOutputXyzw, inf.firstNewOutputReg + v});
    std::vector<uint32_t> epilogue = {kOpMov | (5u << 24), kDstOutputXyzw, inf.positionReg, kSrcTempXyzw,
                                      inf.tempReg};
    for (uint32_t v = 0; v < opt.extraViews; ++v)
        for (uint32_t c = 0; c < 4; ++c)
            epilogue.insert(epilogue.end(), {kOpDp4 | (8u << 24), DstOutputComp(c), inf.firstNewOutputReg + v,
                                             kSrcCbXyzw, opt.cbSlot, 4 * v + c, kSrcTempXyzw, inf.tempReg});

    // Pass 2: copy with insertions and operand rewrites.
    Walker w;
    w.posReg = inf.positionReg;
    w.tempReg = inf.tempReg;
    std::vector<uint32_t> res(in.begin(), in.begin() + 2);
    for (size_t pc = 2; pc < in.size();) {
        if (pc == cbInsert) res.insert(res.end(), cbDecl, cbDecl + 4);
        if (pc == lastOutput) {
            res.insert(res.end(), outDecls.begin(), outDecls.end());
            if (!haveTemps) res.insert(res.end(), {kOpDclTemps | (2u << 24), 1u});
        }
        const uint32_t op = Opcode(in[pc]);
        const uint32_t len = InstrLength(&in[pc], in.size() - pc);
        if (len == 0 || pc + len > in.size()) return SetErr(err, "bad instruction length"), false;
        std::vector<uint32_t> ins(in.begin() + pc, in.begin() + pc + len);
        if (pc == tempsAt && haveTemps) ins[1] += 1;
        if (pc >= firstInstr && op != kOpCustomData && !IsDeclaration(op)) {
            if (op == kOpInterfaceCall) return SetErr(err, "interface calls are not supported"), false;
            size_t k = 1;
            for (bool ext = (ins[0] >> 31) != 0; ext && k < ins.size(); ++k) ext = (ins[k] >> 31) != 0;
            while (k < ins.size() && !w.error) k += w.Operand(&ins[k], ins.size() - k);
            if (w.error || k != ins.size())
                return SetErr(err, "cannot parse the operands of opcode " + std::to_string(op) + " at token " +
                                       std::to_string(pc)),
                       false;
            if (op == kOpRet || op == kOpRetc) {
                res.insert(res.end(), epilogue.begin(), epilogue.end());
                ++inf.returns;
            }
        }
        res.insert(res.end(), ins.begin(), ins.end());
        pc += len;
    }
    if (inf.returns == 0) return SetErr(err, "no ret instruction"), false;
    inf.rewrittenOperands = w.rewritten;
    if (inf.rewrittenOperands == 0) return SetErr(err, "position is never written"), false;
    res[1] = static_cast<uint32_t>(res.size());

    shex->data.resize(res.size() * 4);
    memcpy(shex->data.data(), res.data(), shex->data.size());
    *osgn = WriteSignature(osgn->fourcc, sig);
    out = Build(chunks);
    if (info) *info = inf;
    return true;
}

}  // namespace qvfr::dxbc
