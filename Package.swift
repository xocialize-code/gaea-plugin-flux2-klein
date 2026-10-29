// swift-tools-version: 6.2
import PackageDescription

// gaea-plugin-flux2-klein: Black Forest Labs' FLUX.2 Klein 4B and Klein 4B Base as a Gaea plugin. One worker
// executable built on GaeaPluginKit over flux2-klein-swift, packaged as a signed .gaeaplugin by
// scripts/build-plugin.sh.
//
// klein-inproc is not part of the plugin: it runs the same package in-process through MLXServeEngine, with the
// same dependency resolution, as the reference a run through Gaea is compared against.
//
// GAEA_PLUGIN_KIT_PATH points the kit at a local checkout while you work on both.
let kitPath = Context.environment["GAEA_PLUGIN_KIT_PATH"]

let package = Package(
    name: "gaea-plugin-flux2-klein",
    platforms: [.macOS(.v26)], // mlx-engine-swift's MLXToolKit requires macOS 26
    dependencies: [
        kitPath.map { .package(path: $0) }
            ?? .package(url: "https://github.com/xocialize-code/gaea-plugin-kit.git", from: "0.2.0"),
        .package(url: "https://github.com/xocialize/flux2-klein-swift.git", exact: "0.10.1"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", from: "0.72.0"),
    ],
    targets: [
        .target(
            name: "KleinPlugin",
            dependencies: [
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
                .product(name: "MLXKlein", package: "flux2-klein-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ]
        ),
        .executableTarget(
            name: "gaea-flux2-klein-worker",
            dependencies: [
                "KleinPlugin",
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
            ]
        ),
        .executableTarget(
            name: "klein-inproc",
            dependencies: [
                "KleinPlugin",
                .product(name: "MLXKlein", package: "flux2-klein-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
