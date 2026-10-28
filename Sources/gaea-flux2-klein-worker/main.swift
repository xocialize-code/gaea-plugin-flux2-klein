import GaeaPluginKit
import KleinPlugin

// The plugin's worker: KleinPlugin's definition over GPP (GaeaPluginKit). `--manifest <path>` writes manifest.json.
await GaeaPluginMain.run(kleinPlugin)
