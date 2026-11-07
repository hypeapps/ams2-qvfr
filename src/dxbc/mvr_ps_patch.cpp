#include "mvr_ps_patch.h"

#include <algorithm>
#include <cstring>

#include "dxbc.h"

namespace qvfr::dxbc {
namespace {

// Shader model 4/5 token format (d3d11TokenizedProgramFormat.hpp).
enum : uint32_t {
    kOpAnd = 0x01,
    kOpDiv = 0x0E,
    kOpElse = 0x12,
    kOpEndIf = 0x15,
    kOpIf = 0x1F,
    kOpIeq = 0x20,
    kOpLd = 0x2D,
    kOpLdMs = 0x2E,
    kOpMad = 0x32,
    kOpCustomData = 0x35,
    kOpMov = 0x36,
    kOpMovc = 0x37,
    kOpMul = 0x38,
    kOpResinfo = 0x3D,
    kOpSample = 0x45,
    kOpSampleD = 0x49,
    kOpSampleB = 0x4A,
    kOpDclResource = 0x58,
    kOpDclConstantBuffer = 0x59,
    kOpDclInputPs = 0x62,
    kOpDclInputPsSgv = 0x63,
    kOpDclInputPsSiv = 0x64,
    kOpDclOutput = 0x65,
    kOpDclOutputSiv = 0x67,
    kOpDclTemps = 0x68,
    kOpDclGlobalFlags = 0x6A,
    kOpLod = 0x6C,
    kOpGather4 = 0x6D,
    kOpSamplePos = 0x6E,
    kOpSampleInfo = 0x6F,
    kOpInterfaceCall = 0x78,
    kOpBufInfo = 0x79,
    kOpGather4C = 0x7E,
    kOpGather4Po = 0x7F,
    kOpGather4PoC = 0x80,
    kOpDclResourceRaw = 0xA1,
    kOpDclResourceStructured = 0xA2,
    kOpEvalSnapped = 0xCB,
    kOpEvalCentroid = 0xCD,
};
enum : uint32_t {
    kOperandTemp = 0,
    kOperandInput = 1,
    kOperandImm32 = 4,
    kOperandImm64 = 5,
    kOperandResource = 7,
};
constexpr uint32_t kPixelShader = 0;
constexpr uint32_t kNameRenderTargetArrayIndex = 4;  // D3D_NAME_RENDER_TARGET_ARRAY_INDEX
constexpr uint32_t kTypeUint32 = 1;
constexpr uint32_t kInterpolationConstant = 1;
constexpr uint32_t kTestNonZero = 1u << 18;
constexpr uint32_t kFloatOne = 0x3F800000;
constexpr uint32_t kSitTexture = 2;  // D3D_SIT_TEXTURE

constexpr uint32_t Swz(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return a | (b << 2) | (c << 4) | (d << 6); }
constexpr uint32_t kXyzw = Swz(0, 1, 2, 3);
// A 4-component operand with a 1D immediate index: mask (destinations) or swizzle (sources).
constexpr uint32_t Dst(uint32_t type, uint32_t mask) { return 2u | (mask << 4) | (type << 12) | (1u << 20); }
constexpr uint32_t Src(uint32_t type, uint32_t swz) { return 2u | (1u << 2) | (swz << 4) | (type << 12) | (1u << 20); }
constexpr uint32_t Select1(uint32_t type, uint32_t comp) { return 2u | (2u << 2) | (comp << 4) | (type << 12) | (1u << 20); }
constexpr uint32_t kSrcCbXyzw = 0x00208E46;  // cb#[#].xyzw, 2D immediate index
constexpr uint32_t kSrcCbX = 2u | (2u << 2) | (8u << 12) | (2u << 20);  // cb#[#].x
constexpr uint32_t kSrcImm1 = 0x00004001;                                // l(a), followed by 1 value
constexpr uint32_t kSrcImm4 = 0x00004002;    // l(a, b, c, d), followed by 4 values

uint32_t Opcode(uint32_t tok) { return tok & 0x7FF; }
bool IsDeclaration(uint32_t op) { return (op >= 0x58 && op <= 0x6A) || (op >= 0x8F && op <= 0xA2) || op == 0xCE; }

uint32_t InstrLength(const uint32_t* t, size_t avail) {
    if (Opcode(t[0]) == kOpCustomData) return avail >= 2 ? t[1] : 0;
    uint32_t len = (t[0] >> 24) & 0x7F;
    if (len == 0 && avail >= 2) len = t[1];  // long declarations keep the length in the next token
    return len;
}

uint32_t Get32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

void SetErr(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

// One top-level operand of an instruction.
struct Operand {
    size_t at = 0, len = 0;  // in tokens, from the instruction start
    uint32_t type = 0;
    bool immIndex = false;  // 1D immediate index
    uint32_t index = 0;
    uint32_t token = 0;
    size_t indexAt = 0;  // the index token, from the instruction start (immIndex)
};

// Parses operands and finds every register operand with a 1D immediate index, at any depth (relative indices hold
// operands of their own).
struct OperandParser {
    bool error = false;
    bool relativeInput = false;  // an input register with a relative index
    // Every input register operand with a 1D immediate index: positions of its token and index token.
    std::vector<std::pair<size_t, size_t>> inputs;

    size_t Parse(const uint32_t* t, size_t base, size_t avail, Operand* top) {
        if (avail == 0) return Fail();
        const uint32_t tok = t[base];
        size_t n = 1;
        for (bool ext = (tok >> 31) != 0; ext; ++n) {
            if (n >= avail) return Fail();
            ext = (t[base + n] >> 31) != 0;
        }
        const uint32_t type = (tok >> 12) & 0xFF;
        const uint32_t comps = tok & 3;
        const uint32_t dim = (tok >> 20) & 3;
        if (top) {
            top->type = type;
            top->token = tok;
        }
        if (type == kOperandImm32) n += comps == 2 ? 4 : 1;
        if (type == kOperandImm64) n += comps == 2 ? 8 : 2;
        for (uint32_t d = 0; d < dim; ++d) {
            const uint32_t rep = (tok >> (22 + 3 * d)) & 7;
            if (d == 0 && dim == 1 && rep == 0) {
                if (n >= avail) return Fail();
                if (top) {
                    top->immIndex = true;
                    top->index = t[base + n];
                    top->indexAt = base + n;
                }
                if (type == kOperandInput) inputs.emplace_back(base, base + n);
            } else if (type == kOperandInput && rep != 0) {
                relativeInput = true;
            }
            switch (rep) {
                case 0: n += 1; break;
                case 1: n += 2; break;
                case 2: n += Sub(t, base + n, avail - n); break;
                case 3: n += 1; n += n <= avail ? Sub(t, base + n, avail - n) : Fail(); break;
                case 4: n += 2; n += n <= avail ? Sub(t, base + n, avail - n) : Fail(); break;
                default: return Fail();
            }
            if (error) return Fail();
        }
        return n <= avail ? n : Fail();
    }
    size_t Sub(const uint32_t* t, size_t base, size_t avail) { return Parse(t, base, avail, nullptr); }
    size_t Fail() {
        error = true;
        return 1u << 30;
    }
};

// The operands of one instruction (not a declaration). False: cannot parse.
bool ParseInstruction(const std::vector<uint32_t>& ins, std::vector<Operand>& ops, OperandParser& p) {
    ops.clear();
    size_t k = 1;
    for (bool ext = (ins[0] >> 31) != 0; ext && k < ins.size(); ++k) ext = (ins[k] >> 31) != 0;
    while (k < ins.size()) {
        Operand o;
        o.at = k;
        o.len = p.Parse(ins.data(), k, ins.size() - k, &o);
        if (p.error) return false;
        ops.push_back(o);
        k += o.len;
    }
    return k == ins.size();
}

enum class TexUse { None, Keep, Texel, Uv, Unsupported };

TexUse UseOf(uint32_t op) {
    switch (op) {
        case kOpLd:
        case kOpLdMs: return TexUse::Texel;
        case kOpResinfo:
        case kOpSampleInfo:
        case kOpSamplePos:
        case kOpBufInfo: return TexUse::Keep;
        case kOpSampleD: return TexUse::Unsupported;  // the gradients would need the scale too
        case kOpLod:
        case kOpGather4:
        case kOpGather4C:
        case kOpGather4Po:
        case kOpGather4PoC: return TexUse::Uv;
        default: return op >= kOpSample && op <= kOpSampleB ? TexUse::Uv : TexUse::Unsupported;
    }
}

// Texture registers of the screen textures, by name from the reflection chunk: reg -> index into the option list.
bool ScreenTextureRegs(std::vector<Chunk>& chunks, const MvrPsPatchOptions& opt, std::vector<std::pair<uint32_t, uint32_t>>& out,
                       std::string* err) {
    out.clear();
    const Chunk* rdef = Find(chunks, FourCC('R', 'D', 'E', 'F'));
    if (!rdef) return true;  // stripped: screen inputs only
    const auto& d = rdef->data;
    if (d.size() < 28) return SetErr(err, "reflection chunk too small"), false;
    const uint32_t bindCount = Get32(d.data() + 8), bindOffset = Get32(d.data() + 12), target = Get32(d.data() + 16);
    const uint32_t major = (target >> 4) & 0xF, minor = target & 0xF;
    const size_t stride = major == 5 && minor >= 1 ? 40 : 32;
    if (bindOffset + stride * bindCount > d.size()) return SetErr(err, "reflection bindings out of range"), false;
    for (uint32_t i = 0; i < bindCount; ++i) {
        const uint8_t* b = d.data() + bindOffset + stride * i;
        const uint32_t nameOff = Get32(b), type = Get32(b + 4), point = Get32(b + 20), count = Get32(b + 24);
        if (type != kSitTexture || nameOff >= d.size()) continue;
        const auto* s = reinterpret_cast<const char*>(d.data() + nameOff);
        const std::string name(s, strnlen(s, d.size() - nameOff));
        for (uint32_t k = 0; k < opt.screenTextures.size(); ++k) {
            if (name != opt.screenTextures[k]) continue;
            if (count != 1) return SetErr(err, "screen texture " + name + " is an array"), false;
            out.emplace_back(point, k);
        }
    }
    return true;
}

}  // namespace

MvrPsPatchResult PatchPsForMvr(const void* code, size_t size, const MvrPsPatchOptions& opt, std::vector<uint8_t>& out,
                               MvrPsPatchInfo* info, std::string* err) {
    using R = MvrPsPatchResult;
    auto refuse = [&](const std::string& msg) {
        SetErr(err, msg);
        return R::Refused;
    };
    if (opt.cbSlot > 13) return refuse("cbSlot must be 0..13");
    if (opt.screenTextures.size() > 32 || opt.texBase + opt.screenTextures.size() > 128)
        return refuse("bad screen texture slots");
    std::vector<Chunk> chunks;
    if (!Parse(code, size, chunks, err)) return R::Refused;
    Chunk* shex = Find(chunks, FourCC('S', 'H', 'E', 'X'));
    if (!shex) shex = Find(chunks, FourCC('S', 'H', 'D', 'R'));
    Chunk* isgn = Find(chunks, FourCC('I', 'S', 'G', 'N'));
    if (!shex) return refuse("no SHEX/SHDR chunk");
    if (shex->data.size() % 4 || shex->data.size() < 8) return refuse("bad program chunk size");
    std::vector<uint32_t> in(shex->data.size() / 4);
    memcpy(in.data(), shex->data.data(), shex->data.size());
    if ((in[0] >> 16) != kPixelShader) return refuse("not a pixel shader");
    if (((in[0] >> 4) & 0xF) == 5 && (in[0] & 0xF) >= 1) return refuse("shader model 5.1 is not supported");
    if (in[1] != in.size()) return refuse("program length does not match the chunk");

    std::vector<std::pair<uint32_t, uint32_t>> texRegs;  // screen texture register -> option index
    if (!ScreenTextureRegs(chunks, opt, texRegs, err)) return R::Refused;
    auto texIndex = [&](uint32_t reg) -> int {
        for (const auto& [r, k] : texRegs)
            if (r == reg) return static_cast<int>(k);
        return -1;
    };

    // Pass 1: declarations.
    struct InputDecl {
        uint32_t reg, mask;
    };
    std::vector<InputDecl> inputDecls;
    std::vector<std::pair<size_t, uint32_t>> resourceDecls;  // position, register
    size_t lastCb = 0, lastInput = 0, firstOutput = 0, lastResource = 0, globalFlagsEnd = 2, tempsAt = 0, firstInstr = 0;
    uint32_t temps = 0;
    bool haveTemps = false;
    for (size_t pc = 2; pc < in.size();) {
        const uint32_t op = Opcode(in[pc]);
        const uint32_t len = InstrLength(&in[pc], in.size() - pc);
        if (len == 0 || pc + len > in.size()) return refuse("bad instruction length");
        if (!IsDeclaration(op) && op != kOpCustomData) {
            firstInstr = pc;
            break;
        }
        if (op == kOpDclConstantBuffer) {
            if (len < 4) return refuse("bad constant buffer declaration");
            if (in[pc + 2] == opt.cbSlot) return refuse("constant buffer slot already used");
            lastCb = pc + len;
        } else if (op == kOpDclInputPs || op == kOpDclInputPsSgv || op == kOpDclInputPsSiv) {
            if (len < 3 || ((in[pc + 1] >> 20) & 3) != 1) return refuse("unsupported input declaration");
            inputDecls.push_back({in[pc + 2], (in[pc + 1] >> 4) & 0xF});
            if (op == kOpDclInputPsSiv && len >= 4 && in[pc + 3] == kNameRenderTargetArrayIndex)
                return refuse("shader already reads SV_RenderTargetArrayIndex");
            lastInput = pc + len;
        } else if (op == kOpDclOutput || op == kOpDclOutputSiv || op == kOpDclOutput + 1) {
            if (!firstOutput) firstOutput = pc;
        } else if (op == kOpDclResource || op == kOpDclResourceRaw || op == kOpDclResourceStructured) {
            if (len < 3 || ((in[pc + 1] >> 20) & 3) != 1) return refuse("unsupported resource declaration");
            const uint32_t reg = in[pc + 2];
            if (reg >= opt.texBase && reg < opt.texBase + opt.screenTextures.size())
                return refuse("texture slot for the extra view already used");
            resourceDecls.emplace_back(pc, reg);
            lastResource = pc + len;
        } else if (op == kOpDclTemps) {
            haveTemps = true;
            tempsAt = pc;
            temps = in[pc + 1];
        } else if (op == kOpDclGlobalFlags) {
            globalFlagsEnd = pc + len;
        }
        pc += len;
    }
    if (!firstInstr) return refuse("no instructions");

    // Pass 2: screen inputs (divided by their own w) and screen texture reads.
    uint32_t screenMask = 0, texMask = 0, wraps = 0;
    bool relativeInput = false;
    std::vector<uint32_t> evalInputs;
    std::vector<Operand> ops;
    for (size_t pc = firstInstr; pc < in.size();) {
        const uint32_t op = Opcode(in[pc]);
        const uint32_t len = InstrLength(&in[pc], in.size() - pc);
        if (len == 0 || pc + len > in.size()) return refuse("bad instruction length");
        if (op == kOpCustomData || IsDeclaration(op)) {
            pc += len;
            continue;
        }
        if (op == kOpInterfaceCall) return refuse("interface calls are not supported");
        const std::vector<uint32_t> ins(in.begin() + pc, in.begin() + pc + len);
        OperandParser p;
        if (!ParseInstruction(ins, ops, p))
            return refuse("cannot parse the operands of opcode " + std::to_string(op) + " at token " + std::to_string(pc));
        relativeInput = relativeInput || p.relativeInput;
        if (op >= kOpEvalSnapped && op <= kOpEvalCentroid)
            for (const auto& o : ops)
                if (o.type == kOperandInput && o.immIndex) evalInputs.push_back(o.index);
        if (op == kOpDiv && ops.size() == 3) {
            const Operand &a = ops[1], &b = ops[2];
            const bool swzA = (a.token & 3) == 2 && ((a.token >> 2) & 3) == 1;
            const bool swzB = (b.token & 3) == 2 && ((b.token >> 2) & 3) == 1;
            const uint32_t selA = (a.token >> 4) & 0xFF, selB = (b.token >> 4) & 0xFF;
            bool aXy = swzA;
            for (int c = 0; c < 4; ++c) aXy = aXy && ((selA >> (2 * c)) & 3) <= 1;
            if (a.type == kOperandInput && b.type == kOperandInput && a.immIndex && b.immIndex && a.index == b.index &&
                a.index < 32 && aXy && swzB && selB == 0xFF && (a.token >> 31) == 0 && (b.token >> 31) == 0)
                screenMask |= 1u << a.index;
        }
        for (const auto& o : ops) {
            if (o.type != kOperandResource || !o.immIndex) continue;
            const int k = texIndex(o.index);
            if (k < 0) continue;
            const TexUse use = UseOf(op);
            if (use == TexUse::Unsupported)
                return refuse("opcode " + std::to_string(op) + " reads a screen texture");
            if (use != TexUse::Keep) {
                texMask |= 1u << k;
                ++wraps;
            }
        }
        pc += len;
    }
    // Screen inputs need x, y and w declared.
    for (uint32_t r = 0; r < 32; ++r) {
        if (!(screenMask & (1u << r))) continue;
        bool ok = false;
        for (const auto& d : inputDecls)
            if (d.reg == r && (d.mask & 0xB) == 0xB) ok = true;
        if (!ok) screenMask &= ~(1u << r);
    }
    if (!screenMask && !texMask) return R::NotNeeded;
    if (screenMask && relativeInput) return refuse("relatively indexed inputs");
    for (uint32_t r : evalInputs)
        if (r < 32 && (screenMask & (1u << r))) return refuse("eval_* on a screen input");
    if (!isgn) return refuse("no ISGN chunk (ISG1 not supported)");

    // Input signature: SV_RenderTargetArrayIndex after the last input register.
    std::vector<SigElement> sig;
    if (!ReadSignature(*isgn, sig, err)) return R::Refused;
    uint32_t maxReg = 0;
    bool anyInput = false;
    for (const auto& e : sig) {
        maxReg = std::max(maxReg, e.reg);
        anyInput = true;
    }
    for (const auto& d : inputDecls) {
        maxReg = std::max(maxReg, d.reg);
        anyInput = true;
    }
    MvrPsPatchInfo inf;
    inf.viewReg = opt.viewReg ? opt.viewReg : (anyInput ? maxReg + 1 : 0);
    if (inf.viewReg >= 32) return refuse("no free input register");
    for (const auto& e : sig)
        if (e.reg == inf.viewReg) return refuse("input register for the view number already used");
    {
        SigElement e;
        e.name = "SV_RenderTargetArrayIndex";
        e.systemValue = kNameRenderTargetArrayIndex;
        e.componentType = kTypeUint32;
        e.reg = inf.viewReg;
        e.mask = 0x1;
        e.rwMask = 0x1;
        sig.push_back(e);
    }

    // Temps: S and O, the scaled coordinate, the gated view number, one per screen input.
    inf.firstTemp = temps;
    const uint32_t tSo = temps, tUv = temps + 1, tView = temps + 2;
    uint32_t screenTemp[32] = {};
    uint32_t next = temps + 3;
    for (uint32_t r = 0; r < 32; ++r)
        if (screenMask & (1u << r)) {
            screenTemp[r] = next++;
            ++inf.screenInputs;
        }
    inf.temps = next - temps;
    inf.screenInputMask = screenMask;
    inf.textureMask = texMask;

    // New declarations.
    const uint32_t cbDecl[4] = {kOpDclConstantBuffer | (4u << 24), kSrcCbXyzw, opt.cbSlot, kMvrPsRows};
    const uint32_t viewDecl[4] = {kOpDclInputPsSiv | (kInterpolationConstant << 11) | (4u << 24),
                                  Dst(kOperandInput, 0x1), inf.viewReg, kNameRenderTargetArrayIndex};
    std::vector<uint32_t> texDecls;
    for (uint32_t k = 0; k < opt.screenTextures.size(); ++k) {
        if (!(texMask & (1u << k))) continue;
        for (const auto& [r, idx] : texRegs) {
            if (idx != k) continue;
            for (const auto& [at, reg] : resourceDecls) {
                if (reg != r) continue;
                const uint32_t len = InstrLength(&in[at], in.size() - at);
                std::vector<uint32_t> d(in.begin() + at, in.begin() + at + len);
                d[2] = opt.texBase + k;
                texDecls.insert(texDecls.end(), d.begin(), d.end());
                break;
            }
            break;
        }
    }
    const size_t viewInsert = lastInput ? lastInput : firstOutput ? firstOutput : firstInstr;
    const size_t cbInsert = lastCb ? lastCb : globalFlagsEnd;
    const size_t texInsert = lastResource ? lastResource : cbInsert;

    // Prologue: the gated view number (ieq on row 2.x, and with SV_RenderTargetArrayIndex), movc S/O by it, then
    // each screen input into its temp.
    std::vector<uint32_t> prologue = {kOpIeq | (8u << 24), Dst(kOperandTemp, 0x1), tView,
                                      kSrcCbX, opt.cbSlot, 2,
                                      kSrcImm1, kMvrPsMagic,
                                      kOpAnd | (7u << 24), Dst(kOperandTemp, 0x1), tView,
                                      Select1(kOperandTemp, 0), tView,
                                      Select1(kOperandInput, 0), inf.viewReg,
                                      kOpMovc | (13u << 24), Dst(kOperandTemp, 0xF), tSo,
                                      Src(kOperandTemp, Swz(0, 0, 0, 0)), tView,
                                      kSrcCbXyzw, opt.cbSlot, 0,
                                      kSrcImm4, kFloatOne, kFloatOne, 0, 0};
    for (uint32_t r = 0; r < 32; ++r) {
        if (!(screenMask & (1u << r))) continue;
        const uint32_t t = screenTemp[r];
        // mul t.xy, v.wwww, so.zwzz; mad t.xy, v.xyxx, so.xyxx, t.xyxx
        prologue.insert(prologue.end(), {kOpMul | (7u << 24), Dst(kOperandTemp, 0x3), t, Src(kOperandInput, 0xFF), r,
                                         Src(kOperandTemp, Swz(2, 3, 2, 2)), tSo});
        prologue.insert(prologue.end(), {kOpMad | (9u << 24), Dst(kOperandTemp, 0x3), t,
                                         Src(kOperandInput, Swz(0, 1, 0, 0)), r, Src(kOperandTemp, Swz(0, 1, 0, 0)), tSo,
                                         Src(kOperandTemp, Swz(0, 1, 0, 0)), t});
        uint32_t declMask = 0;
        for (const auto& d : inputDecls)
            if (d.reg == r) declMask |= d.mask;
        if (declMask & 0xC)
            prologue.insert(prologue.end(),
                            {kOpMov | (5u << 24), Dst(kOperandTemp, declMask & 0xC), t, Src(kOperandInput, kXyzw), r});
    }

    // Pass 3: copy with the insertions and rewrites.
    std::vector<uint32_t> res(in.begin(), in.begin() + 2);
    for (size_t pc = 2; pc < in.size();) {
        if (pc == cbInsert) res.insert(res.end(), cbDecl, cbDecl + 4);
        if (pc == texInsert) res.insert(res.end(), texDecls.begin(), texDecls.end());
        if (pc == viewInsert) res.insert(res.end(), viewDecl, viewDecl + 4);
        if (pc == firstInstr) {
            if (!haveTemps) res.insert(res.end(), {kOpDclTemps | (2u << 24), inf.temps});
            res.insert(res.end(), prologue.begin(), prologue.end());
        }
        const uint32_t op = Opcode(in[pc]);
        const uint32_t len = InstrLength(&in[pc], in.size() - pc);
        std::vector<uint32_t> ins(in.begin() + pc, in.begin() + pc + len);
        pc += len;
        if (op == kOpCustomData || IsDeclaration(op)) {
            if (op == kOpDclTemps && haveTemps) ins[1] += inf.temps;
            res.insert(res.end(), ins.begin(), ins.end());
            continue;
        }
        OperandParser p;
        if (!ParseInstruction(ins, ops, p)) return refuse("cannot parse an instruction");
        for (const auto& [tokAt, idxAt] : p.inputs) {
            const uint32_t r = ins[idxAt];
            if (r >= 32 || !(screenMask & (1u << r))) continue;
            ins[tokAt] = (ins[tokAt] & ~(0xFFu << 12)) | (kOperandTemp << 12);
            ins[idxAt] = screenTemp[r];
        }
        int tex = -1;
        size_t texOp = 0;
        for (size_t i = 0; i < ops.size(); ++i)
            if (ops[i].type == kOperandResource && ops[i].immIndex && texIndex(ops[i].index) >= 0) {
                tex = texIndex(ops[i].index);
                texOp = i;
            }
        const TexUse use = tex >= 0 ? UseOf(op) : TexUse::None;
        if (use != TexUse::Texel && use != TexUse::Uv) {
            res.insert(res.end(), ins.begin(), ins.end());
            continue;
        }
        // if_nz view; [mul uv, coord, cb[1]]; the instruction on the extra view's texture; else; as it was; endif
        std::vector<uint32_t> alt = ins;
        alt[ops[texOp].indexAt] = opt.texBase + static_cast<uint32_t>(tex);
        res.insert(res.end(), {kOpIf | kTestNonZero | (3u << 24), Select1(kOperandTemp, 0), tView});
        if (use == TexUse::Uv) {
            if (ops.size() < 2) return refuse("sample without coordinates");
            const Operand& c = ops[1];
            std::vector<uint32_t> mul = {kOpMul, Dst(kOperandTemp, 0xF), tUv};
            mul.insert(mul.end(), ins.begin() + c.at, ins.begin() + c.at + c.len);
            mul.insert(mul.end(), {kSrcCbXyzw, opt.cbSlot, 1});
            if (mul.size() > 127) return refuse("instruction too long");
            mul[0] |= static_cast<uint32_t>(mul.size()) << 24;
            res.insert(res.end(), mul.begin(), mul.end());
            std::vector<uint32_t> a(alt.begin(), alt.begin() + c.at);
            a.insert(a.end(), {Src(kOperandTemp, kXyzw), tUv});
            a.insert(a.end(), alt.begin() + c.at + c.len, alt.end());
            if (a.size() > 127) return refuse("instruction too long");
            a[0] = (a[0] & ~(0x7Fu << 24)) | (static_cast<uint32_t>(a.size()) << 24);
            alt = std::move(a);
            ++inf.scaled;
        }
        res.insert(res.end(), alt.begin(), alt.end());
        res.push_back(kOpElse | (1u << 24));
        res.insert(res.end(), ins.begin(), ins.end());
        res.push_back(kOpEndIf | (1u << 24));
        ++inf.wrapped;
    }
    res[1] = static_cast<uint32_t>(res.size());

    shex->data.resize(res.size() * 4);
    memcpy(shex->data.data(), res.data(), shex->data.size());
    *isgn = WriteSignature(isgn->fourcc, sig);
    out = Build(chunks);
    if (info) *info = inf;
    return R::Patched;
}

}  // namespace qvfr::dxbc
