import Foundation
import GaeaPluginKit
import GaeaProtocol
import ImageIO
import MLXKlein
import MLXToolKit

// gaea-plugin-flux2-klein: Black Forest Labs' FLUX.2 Klein 4B (Apache-2.0) as a Gaea plugin, on flux2-klein-swift
// through MLXServeEngine (Gaea AB-T-0367).
//
// - klein-4b: distilled to 4 steps at guidance 1 (no CFG). Fast generation, and Gaea's default image.edit:
//   multi-reference editing from up to four images is what this model is for.
// - klein-4b-base: the undistilled checkpoint, ~28 steps with two-pass CFG and negative prompts. The `best` tier.
// - klein-4b-rdm (0.4.0): EPFL VITA's one-step RDM fine-tune of Klein 4B's DiT on the same encoder and VAE, generate
//   only. 0.43× the 4-step model's time at 1024² at the same TierBench quality (AB-R-1056): the `draft` tier.
//
// Each has three lanes, the DiT's quant: bf16 (the default, as in mlx-image-runtime's catalog), int8 and int4, with
// the package's measured footprints; a Mac too small for bf16 falls back to a smaller lane (G-23). Weights come from
// the models root Gaea sends in configure (MLXEngine's store by default); the engine materializes a missing snapshot.
//
// Over 1024² of area (0.3.0, AB-T-0464) the DiT draws at a quarter or a sixteenth of the area and FlowUpscaler
// (flux2-klein-swift 0.11.0, `metaData["latentUpscale"]`) doubles its latents once or twice before the row-band decode:
// the picture the smaller draw made, with finer detail, up to 4096² (AB-R-0795, AB-R-0797). The package declares the
// line its activation follows over output pixels, and Gaea charges each job on it.
//
// Both models take adapters (AB-T-0381) through `KleinAdapters.support`: up to 4 at 0–2, plain on generate, plain and
// ic on edit. They ship v0.9.1's 7 FLUX.2 Klein 4B adapters, all edit adapters (ids flux2-klein:<id>, the same pinned
// revisions and sha256s); refcontrol-pose and tryon are ic, and their images lead the edit's references in slot order.

/// The largest draw: the area the package's footprints were measured at, and what a latent upscale starts from.
public let drawPixels = 1024 * 1024
/// The largest image: 4096², two FlowUpscaler passes from a 1024² draw.
public let maxPixels = 4096 * 4096
public let minSide = 256, maxSide = 4096, drawMaxSide = 2048, grid = 16, maxReferences = 4

/// FLUX.2 Klein 1-step RDM (epfl-vita/flux2-klein-1step-rdm, Apache-2.0): Klein 4B's DiT fine-tuned for one step by
/// Representation Distribution Matching, on Klein's own encoder and VAE. Its weights are BFL-layout; this repo holds the
/// diffusers-layout snapshot (jeveration-oracle `make_rdm_snapshot.py`, AB-R-1054).
public let rdmRepo = "mlx-community/FLUX.2-klein-4B-1step-RDM-bf16"
/// The published revision, pinned: its DiT is byte for byte the one TierBench measured (sha256 297718962320…).
public let rdmRevision = "2037b5c8da31053d2c3fb9a9520681d95c28080d"

public enum Tier: String, Sendable {
    case distilled, base, rdm

    public var modelID: String {
        switch self {
        case .distilled: "klein-4b"
        case .base: "klein-4b-base"
        case .rdm: "klein-4b-rdm"
        }
    }
    public var defaultSteps: Int {
        switch self {
        case .distilled: 4
        case .base: 28
        case .rdm: 1
        }
    }
    /// Only the base model runs two-pass CFG; the distilled models run at guidance 1.
    public var usesCFG: Bool { self == .base }
    public var defaultGuidance: Double { usesCFG ? 4 : 1 }

    /// The package configuration for one lane; klein-inproc builds the same one.
    public func configuration(quant: Quant) -> KleinConfiguration {
        switch self {
        case .distilled: .turbo(quant: quant)
        case .base: .base(quant: quant)
        case .rdm: KleinConfiguration(repo: rdmRepo, revision: rdmRevision, quant: quant, defaultSteps: 1, guidanceScale: 1.0)
        }
    }
}

/// The FLUX.2 Klein 4B adapters both models ship: v0.9.1's registry (`Resources/flux2-klein-4b.json`), as the engine
/// registers them.
public let kleinAdapters: [ShippedAdapter] = {
    do {
        let decoded = try AdapterRegistry.decode(Data(PackageResources.flux2_klein_4b_json))
        precondition(decoded.skipped.isEmpty, "flux2-klein-4b.json: \(decoded.skipped)")
        return decoded.adapters
    } catch {
        preconditionFailure("flux2-klein-4b.json doesn't decode: \(error)")
    }
}()

/// The MES size a job asks for (`size`, or `aspect` with `megapixels`, Image profile §2), else `fallback`.
func requestedSize(_ job: Job, fallback: (Int, Int)) throws -> (Int, Int) {
    if let size = job.param("size") {
        guard let w = size["width"]?.intValue, let h = size["height"]?.intValue, w > 0, h > 0 else {
            throw GPPError(.invalidRequest, "size is {width, height} in pixels.")
        }
        return (w, h)
    }
    if let aspect = job.param("aspect")?.stringValue {
        let parts = aspect.split(separator: ":").compactMap { Double($0) }
        guard parts.count == 2, parts[0] > 0, parts[1] > 0 else { throw GPPError(.invalidRequest, "aspect is W:H, e.g. 16:9.") }
        let pixels = (job.param("megapixels")?.doubleValue ?? 1.0) * 1_000_000
        let height = (pixels * parts[1] / parts[0]).squareRoot()
        return (Int(height * parts[0] / parts[1]), Int(height))
    }
    return fallback
}

/// How an image size is made: the draw, and the FlowUpscaler passes (0, 1 or 2) that double it on each side.
public struct SizePlan: Equatable, Sendable {
    public var width: Int, height: Int
    public var drawWidth: Int, drawHeight: Int
    public var passes: Int
    public var warning: Warning?

    public static func == (a: SizePlan, b: SizePlan) -> Bool {
        (a.width, a.height, a.drawWidth, a.drawHeight, a.passes) == (b.width, b.height, b.drawWidth, b.drawHeight, b.passes)
    }

    /// `latentUpscale` for the package, when there are passes.
    public var metaData: MetaData { passes > 0 ? ["latentUpscale": .int(passes)] : [:] }
}

/// Fits a size to Klein's envelope and plans it: scaled down to 4096² of area, keeping the aspect; drawn as it is up
/// to 1024²; past that drawn at a quarter (one pass) or a sixteenth (two) of the area and upscaled. Each side is
/// floored to a multiple of 16 × 2^passes, so the draw is a multiple of 16. A change is reported as `size_snapped`.
public func plan(_ size: (Int, Int)) throws -> SizePlan {
    var (w, h) = (Double(size.0), Double(size.1))
    if w * h > Double(maxPixels) {
        let scale = (Double(maxPixels) / (w * h)).squareRoot()
        w *= scale
        h *= scale
    }
    let passes = w * h <= Double(drawPixels) ? 0 : w * h <= 4 * Double(drawPixels) ? 1 : 2
    let unit = grid << passes
    let width = Int(w) / unit * unit, height = Int(h) / unit * unit
    let drawWidth = width >> passes, drawHeight = height >> passes
    guard min(drawWidth, drawHeight) >= minSide, max(drawWidth, drawHeight) <= drawMaxSide else {
        let sides = passes == 0 ? "\(minSide)–\(drawMaxSide) px a side" : "a draw of \(minSide)–\(drawMaxSide) px a side, ×\(1 << passes)"
        throw GPPError(.invalidRequest, "FLUX.2 Klein makes \(minSide)–\(maxSide) px a side, at most 4096² in area: past 1024² it draws "
            + "a quarter or a sixteenth of the area and upscales it. \(size.0)×\(size.1) needs \(sides), which doesn't fit.",
            detail: ["param": "size"])
    }
    let note = passes == 0 ? "multiples of 16, at most 1024² drawn" : "drawn at \(drawWidth)×\(drawHeight) and upscaled ×\(1 << passes), so multiples of \(unit)"
    let warning = (width, height) == size ? nil : Warning(
        code: "size_snapped", message: "Made \(width)×\(height) for \(size.0)×\(size.1): \(note), at most 4096² in area.", param: "size")
    return SizePlan(width: width, height: height, drawWidth: drawWidth, drawHeight: drawHeight, passes: passes, warning: warning)
}

/// The run's settings shared by generate and edit: steps, guidance, seed, and the warnings for what the tier ignores.
struct Shared {
    var steps: Int
    var guidance: Double
    var negative: String?
    var seed: UInt64
    var count: Int
    var warnings: [Warning] = []

    /// `counted`: image.generate takes MES `count`; image.edit has one output and no count.
    init(_ job: Job, tier: Tier, counted: Bool = true) throws {
        steps = job.param("x-klein.steps")?.intValue ?? tier.defaultSteps
        guard (1...100).contains(steps) else { throw GPPError(.invalidRequest, "x-klein.steps is 1–100.") }
        guidance = !tier.usesCFG ? 1 : job.param("x-klein.guidance")?.doubleValue ?? tier.defaultGuidance
        guard (1...20).contains(guidance) else { throw GPPError(.invalidRequest, "x-klein.guidance is 1–20.") }
        count = counted ? job.param("count")?.intValue ?? 1 : 1
        guard (1...8).contains(count) else { throw GPPError(.invalidRequest, "count is 1–8.") }
        let negative = job.param("negative_prompt")?.stringValue.flatMap { $0.isEmpty ? nil : $0 }
        if !tier.usesCFG, negative != nil {
            warnings.append(Warning(code: "negative_prompt_ignored",
                                    message: "This FLUX.2 Klein 4B model is distilled to run without CFG, so a negative prompt does nothing; "
                                        + "Klein 4B Base (best) honours it.",
                                    param: "negative_prompt"))
        }
        self.negative = tier.usesCFG ? negative : nil
        if job.param("prompt_enhance")?.stringValue == "on" {
            warnings.append(Warning(code: "prompt_enhance_ignored", message: "This plugin doesn't rewrite prompts; Gaea's enhancer does.",
                                    param: "prompt_enhance"))
        }
        // Image profile §2: a request without a seed still gets one, reported in effective.
        seed = job.seed ?? UInt64.random(in: 0..<(1 << 31))
    }

    func effective(_ size: SizePlan, counted: Bool = true) -> [String: JSONValue] {
        var values: [String: JSONValue] = [
            "size": ["width": .int(Int64(size.width)), "height": .int(Int64(size.height))],
            "seed": .int(Int64(seed)), "x-klein.steps": .int(Int64(steps)),
        ]
        if size.passes > 0 {
            values["x-klein.upscaled"] = ["from": ["width": .int(Int64(size.drawWidth)), "height": .int(Int64(size.drawHeight))],
                                          "passes": .int(Int64(size.passes)), "model": "FlowUpscaler"]
        }
        if counted { values["count"] = .int(Int64(count)) }
        if guidance > 1 { values["x-klein.guidance"] = .double(guidance) }
        if let negative { values["negative_prompt"] = .string(negative) }
        return values
    }
}

/// MES image.generate: one engine call per image, each with seed + i.
func generate(_ tier: Tier) -> EngineOperation {
    EngineOperation("image.generate", outputNames: ["image": "images"]) { job in
        guard let prompt = try job.text("prompt"), !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw GPPError(.invalidRequest, "image.generate needs a prompt.")
        }
        let size = try plan(try requestedSize(job, fallback: (1024, 1024)))
        let adapters = try job.adapterRefs()
        var shared = try Shared(job, tier: tier)
        if let warning = size.warning { shared.warnings.insert(warning, at: 0) }
        let effective = shared.effective(size)
        return (0..<shared.count).map { i in
            EngineCall(T2IRequest(prompt: prompt, negativePrompt: shared.negative, width: size.drawWidth, height: size.drawHeight,
                                  steps: shared.steps, guidanceScale: shared.guidance, seed: shared.seed + UInt64(i),
                                  adapters: adapters, metaData: size.metaData),
                       effective: effective, warnings: shared.warnings)
        }
    }
}

/// MES image.edit: Klein's multi-reference editing. Up to four images condition a new image drawn toward the
/// instruction ("the subject from image 1, on a beach"), at the size asked, else at the package's 1024²; the host brings
/// it back to the first image's size (Image profile §3). One output. An ic adapter's images (refcontrol-pose's skeleton
/// and subject) lead the references in slot order, so with one the edit may have no images of its own.
func edit(_ tier: Tier) -> EngineOperation {
    EngineOperation("image.edit", outputNames: ["image": "image"]) { job in
        guard let instruction = try job.text("instruction"), !instruction.isEmpty else {
            throw GPPError(.invalidRequest, "image.edit needs an instruction.")
        }
        let references = job.inputs["images"] ?? []
        let leading = job.adapters.filter { $0.kind == .referenceConditioned }.flatMap { adapter in
            let slots = kleinAdapters.first { $0.id == adapter.id }?.descriptor.conditioning.map(\.role)
                ?? adapter.inputs.keys.sorted()
            return slots.flatMap { adapter.inputs[$0] ?? [] }
        }
        guard (leading.isEmpty ? 1 : 0)...maxReferences ~= references.count else {
            throw GPPError(.invalidRequest, "image.edit takes 1–\(maxReferences) images; got \(references.count).",
                           detail: ["param": "images"])
        }
        guard let first = (leading + references).first,
              let source = CGImageSourceCreateWithURL(first as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let inputWidth = properties[kCGImagePropertyPixelWidth] as? Int,
              let inputHeight = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw GPPError(.invalidRequest, "image.edit needs readable PNG or JPEG images.")
        }
        _ = (inputWidth, inputHeight)
        // Without a size the package chooses (1024², whatever the inputs' aspect), as on pixel-agency-image-engine
        // v0.9.1, which sends none; the normalizer brings the result back to the first image's size (AB-T-0383).
        let asked = job.param("size") != nil || job.param("aspect") != nil
        let size = try plan(try requestedSize(job, fallback: (1024, 1024)))
        var shared = try Shared(job, tier: tier, counted: false)
        if let warning = size.warning { shared.warnings.insert(warning, at: 0) }
        if job.param("strength") != nil {
            shared.warnings.append(Warning(code: "param_ignored",
                                           message: "Klein edits by reference conditioning from noise, not image-to-image, so strength does nothing.",
                                           param: "strength"))
        }
        let images = try references.indices.map { try job.image("images", index: $0) }
        return [EngineCall(IEditRequest(images: images, prompt: instruction, negativePrompt: shared.negative,
                                        width: asked ? size.drawWidth : nil, height: asked ? size.drawHeight : nil, steps: shared.steps,
                                        guidanceScale: shared.guidance, seed: shared.seed, adapters: try job.adapterRefs(),
                                        metaData: asked ? size.metaData : [:]),
                           effective: shared.effective(size, counted: false),
                           warnings: shared.warnings)]
    }
}

public func lanes(_ tier: Tier) -> [LaneDefinition] {
    [Quant.bf16, .int8, .int4].map { quant in
        LaneDefinition(id: quant.rawValue, default: quant == .bf16, configuration: tier.configuration(quant: quant))
    }
}

let apache: JSONValue = ["spdx": "Apache-2.0", "commercial_use": true, "terms_url": "https://www.apache.org/licenses/LICENSE-2.0"]
let limits = PluginManifest.Limits(maxPixels: maxPixels, minSide: minSide, maxSide: maxSide, sizeMultiple: grid)
let editLimits = PluginManifest.Limits(maxPixels: maxPixels, minSide: minSide, maxSide: maxSide, sizeMultiple: grid,
                                       maxImages: maxReferences)

func stepsParam(_ tier: Tier) -> JSONValue {
    let description = switch tier {
    case .distilled: "Denoising steps. Klein 4B is distilled for 4."
    case .rdm: "Denoising steps. This model is trained for 1."
    case .base: "Denoising steps; more is slower and finer."
    }
    return ["type": "integer", "minimum": 1, "maximum": 100, "default": .int(Int64(tier.defaultSteps)),
            "description": .string(description),
            "x-mes-ui": ["widget": "slider", "label": "Steps", "advanced": true]]
}

let guidanceParam: JSONValue = [
    "type": "number", "minimum": 1, "maximum": 20, "default": 4,
    "description": "Classifier-free guidance: how strongly the prompt (and against the negative prompt) steers. 1 turns CFG off.",
    "x-mes-ui": ["widget": "slider", "label": "Guidance", "advanced": true],
]

func params(_ tier: Tier) -> JSONValue {
    var properties: [String: JSONValue] = ["x-klein.steps": stepsParam(tier)]
    if tier.usesCFG { properties["x-klein.guidance"] = guidanceParam }
    return ["type": "object", "properties": .object(properties), "additionalProperties": false]
}

func offerModel(_ tier: Tier, edit: Bool) -> JSONValue {
    let titles: [Tier: String] = [.distilled: "FLUX.2 Klein 4B", .base: "FLUX.2 Klein 4B Base", .rdm: "FLUX.2 Klein 4B 1-step (RDM)"]
    let qualityTiers: [Tier: JSONValue] = [.distilled: ["draft", "standard"], .base: ["best"], .rdm: ["draft"]]
    var model: [String: JSONValue] = [
        "id": .string(tier.modelID), "title": .string(titles[tier]!), "license": apache,
        "memory_class": "large", "speed_class": .string(tier == .base ? "slow" : "fast"),
        "quality_tiers": qualityTiers[tier]!,
    ]
    switch (tier, edit) {
    case (.rdm, _):
        model["notes"] = .string("Klein 4B's DiT distilled to one step (RDM, EPFL VITA, Apache-2.0) on Klein's encoder and VAE: about 2.5× "
            + "faster than the 4-step model at 512². Trained at 512²; negative prompts do nothing. Text to image only.")
    case (.distilled, false):
        model["notes"] = .string("Distilled 4 steps without CFG: negative prompts do nothing. Strong prompt adherence. Past 1024² it draws a "
            + "quarter or a sixteenth of the area and upscales it (FlowUpscaler), up to 4096²: the same picture, finer.")
    case (.base, false):
        model["notes"] = .string("The undistilled model: about 28 steps with CFG, honours negative prompts. Slower; tighter prompt adherence. "
            + "Past 1024² it draws a quarter or a sixteenth of the area and upscales it, up to 4096².")
    case (.distilled, true):
        model["default"] = true
        model["notes"] = .string("Multi-reference editing: up to 4 images, named in the instruction in order (\"the subject from image 1, on a beach\"). "
            + "Drawn from noise at 1024² unless a size is asked, and returned at the first image's size; strength does nothing. "
            + "Past 1024² it draws a quarter or a sixteenth of the area and upscales it (FlowUpscaler), up to 4096².")
    case (.base, true):
        model["notes"] = "Multi-reference editing with CFG, for tighter instruction adherence; about 28 steps."
    }
    return .object(model)
}

public let kleinPlugin = GaeaPluginDefinition(
    id: "com.xocialize.gaea.flux2-klein",
    version: "0.4.0",
    minGaea: "1.2.0",
    team: "CFZS6RXUCP",
    repo: "https://github.com/xocialize-code/gaea-plugin-flux2-klein",
    title: "FLUX.2 Klein",
    summary: "Black Forest Labs FLUX.2 [klein] 4B (Apache-2.0): a compact rectified-flow MMDiT with Qwen3-4B conditioning. "
        + "The distilled 4-step model and the 28-step base, for generation and multi-reference editing.",
    license: "Apache-2.0",
    executable: "gaea-flux2-klein-worker",
    models: [
        .engine(Klein4BT2IPackage.self, id: Tier.distilled.modelID, title: "FLUX.2 Klein 4B", lanes: lanes(.distilled),
                operations: [generate(.distilled), edit(.distilled)]).shipping(kleinAdapters),
        .engine(Klein4BBaseT2IPackage.self, id: Tier.base.modelID, title: "FLUX.2 Klein 4B Base", lanes: lanes(.base),
                operations: [generate(.base), edit(.base)]).shipping(kleinAdapters),
        .engine(Klein4BT2IPackage.self, id: Tier.rdm.modelID, title: "FLUX.2 Klein 4B 1-step (RDM)", lanes: lanes(.rdm),
                operations: [generate(.rdm)]),
    ],
    offers: [
        PluginManifest.Offer(operation: "image.generate", profile: "image", model: offerModel(.distilled, edit: false),
                             params: params(.distilled), defaults: ["count": 1], limits: limits),
        PluginManifest.Offer(operation: "image.generate", profile: "image", model: offerModel(.base, edit: false),
                             params: params(.base), defaults: ["count": 1], limits: limits),
        PluginManifest.Offer(operation: "image.generate", profile: "image", model: offerModel(.rdm, edit: false),
                             params: params(.rdm), defaults: ["count": 1], limits: limits),
        PluginManifest.Offer(operation: "image.edit", profile: "image", model: offerModel(.distilled, edit: true),
                             params: params(.distilled), limits: editLimits),
        PluginManifest.Offer(operation: "image.edit", profile: "image", model: offerModel(.base, edit: true),
                             params: params(.base), limits: editLimits),
    ])
