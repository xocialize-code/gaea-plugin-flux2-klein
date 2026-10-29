import Foundation
import KleinPlugin
import MLXKlein
import MLXServeCore
import MLXToolKit

// The in-process reference for the plugin: one Klein lane loaded by MLXServeEngine in this process, with the lane
// configuration the worker uses (KleinPlugin) and the request it would build. A cold run then a warm one; the first
// image is written to --out, and one JSON line of timings is printed. Not part of the plugin bundle.
//
//   klein-inproc [--model klein-4b | klein-4b-base] [--lane bf16 | int8 | int4] [--prompt <text>] [--negative <text>]
//                [--width 1024] [--height 1024] [--steps N] [--guidance G] [--seed 42] [--image <png> ...]
//                [--models-root <dir>] [--out <png>]
//
// With --image (repeatable, up to 4) it runs image edit with the prompt as the instruction.
// --adapter <id>=<strength> (repeatable) applies a shipped adapter (flux2-klein:<id>), registered as the worker
// registers it; --adapter-input <id>:<role>=<png> gives an ic adapter its images. With an ic adapter the edit may have
// no --image of its own.

var args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    defer { args.removeSubrange(i...(i + 1)) }
    return args[i + 1]
}
func options(_ name: String) -> [String] {
    var values: [String] = []
    while let value = option(name) { values.append(value) }
    return values
}
func seconds(since start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}

let started = ContinuousClock.now
let tier: Tier = option("--model") == "klein-4b-base" ? .base : .distilled
let quant = Quant(rawValue: option("--lane") ?? "bf16") ?? .bf16
let prompt = option("--prompt") ?? "a red fox sitting in a snowy forest at sunrise, photorealistic"
let negative = option("--negative")
let width = Int(option("--width") ?? "") ?? 1024
let height = Int(option("--height") ?? "") ?? 1024
let steps = Int(option("--steps") ?? "") ?? tier.defaultSteps
let guidance = Double(option("--guidance") ?? "") ?? tier.defaultGuidance
let seed = UInt64(option("--seed") ?? "") ?? 42
let imagePaths = options("--image")
var conditioning: [String: [ConditioningInput]] = [:]
for spec in options("--adapter-input") {
    let (head, path) = (spec.split(separator: "=", maxSplits: 1).map(String.init)[0], String(spec.split(separator: "=", maxSplits: 1)[1]))
    let parts = head.split(separator: ":").map(String.init)
    let (adapter, role) = (parts.dropLast().joined(separator: ":"), parts.last!)
    let image = Image(format: path.lowercased().hasSuffix(".png") ? .png : .jpeg, data: try Data(contentsOf: URL(fileURLWithPath: path)))
    conditioning[adapter, default: []].append(ConditioningInput(role: role, media: .image(image)))
}
let adapterRefs: [AdapterRef] = options("--adapter").map { spec in
    let parts = spec.split(separator: "=", maxSplits: 1).map(String.init)
    return AdapterRef(id: parts[0], strength: parts.count > 1 ? Double(parts[1]) : nil, conditioning: conditioning[parts[0]] ?? [])
}
let outPath = option("--out")
let modelsRoot = URL(fileURLWithPath: option("--models-root")
                     ?? NSHomeDirectory() + "/Library/Application Support/MLXEngine/Models", isDirectory: true)

var result: [String: Any] = ["model": tier.modelID, "lane": quant.rawValue, "width": width, "height": height,
                             "steps": steps, "guidance": guidance, "seed": seed, "references": imagePaths.count]

// The budget Gaea sends a worker on a large Mac (wiresWeights off, the automatic pool cap).
let engine = MLXServeEngine(workerBudget: WorkerBudget(ceilingBytes: 64_000_000_000, wiresWeights: false))
await engine.useModelStore(ModelStore(root: modelsRoot))
let id = PackageID(tier.modelID)
let registration = tier == .distilled ? Klein4BT2IPackage.registration : Klein4BBaseT2IPackage.registration
_ = try await engine.register(registration, configuration: tier.configuration(quant: quant), id: id)
for ref in adapterRefs {
    guard let shipped = kleinAdapters.first(where: { $0.id == ref.id }) else {
        FileHandle.standardError.write(Data("no shipped adapter \(ref.id)\n".utf8))
        exit(2)
    }
    try await engine.registerAdapter(shipped.descriptor)
}
if !adapterRefs.isEmpty { result["adapters"] = adapterRefs.map { "\($0.id)=\($0.strength.map { String($0) } ?? "default")" } }

let edit = !imagePaths.isEmpty || !conditioning.isEmpty
var clock = ContinuousClock.now
_ = try await engine.prepare(edit ? .imageEdit : .textToImage, package: id)
result["prepare_s"] = seconds(since: clock)

let images = try imagePaths.map { path in
    Image(format: path.lowercased().hasSuffix(".png") ? .png : .jpeg, data: try Data(contentsOf: URL(fileURLWithPath: path)))
}
let request: any CapabilityRequest = edit
    ? IEditRequest(images: images, prompt: prompt, negativePrompt: negative, width: width, height: height, steps: steps,
                   guidanceScale: guidance, seed: seed, adapters: adapterRefs)
    : T2IRequest(prompt: prompt, negativePrompt: negative, width: width, height: height, steps: steps,
                 guidanceScale: guidance, seed: seed, adapters: adapterRefs)

for run in ["run1", "run2"] {
    clock = ContinuousClock.now
    let response = try await engine.run(request, package: id)
    result["\(run)_s"] = seconds(since: clock)
    let image = (response as? T2IResponse)?.image ?? (response as? IEditResponse)?.image
    if run == "run1", let image, let outPath { try image.data.write(to: URL(fileURLWithPath: outPath)) }
}
result["cold_s"] = seconds(since: started) - (result["run2_s"] as! Double)

let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
