#!/bin/zsh
# Adapters over GPP against in-process MLXServeEngine (AB-T-0381): a seeded Klein 4B edit with one ic adapter
# (refcontrol-pose: a skeleton and a subject) and one plain one (background-remove), called on the signed bundle the
# way Gaea calls it (gaea-plugin-check: launch under the launch requirement, configure, prepare, run with run.adapters
# built from the manifest's adapters_meta), and through klein-inproc with the same AdapterRefs on this package's
# dependency resolution: byte for byte, and pixel for pixel when the bytes differ. A no-adapter control proves the
# adapters changed the image.
#
#   scripts/golden.sh <out-dir>     (after scripts/build-plugin.sh and `swift build -c release --product klein-inproc`)
#
# The adapters are fetched into the models root (adapters/<sha256>/) by the first run that names them.
set -uo pipefail
ROOT=${0:A:h:h}
cd $ROOT
OUT=${1:?out dir}; mkdir -p $OUT
B=$ROOT/dist/flux2-klein.gaeaplugin
MODELS=${CHECK_MODELS_ROOT:-$HOME/Library/Application Support/MLXEngine/Models}
BIN=$(xcrun swift build -c release --show-bin-path)
CHECK=$BIN/gaea-plugin-check; INPROC=$BIN/klein-inproc
FX=$ROOT/scripts/fixtures
POSE=$FX/pose-512.png; FOX=$FX/fox-256.png
PIXDIFF=$OUT/pixdiff; swiftc -O -o $PIXDIFF $ROOT/scripts/pixdiff.swift
LOG=$OUT/golden.log; : > $LOG
pass=0; fail=0
say() { print -r -- "$*" | tee -a $LOG }

PROMPT="refcontrol apply pose from image 1 with reference from image 2, on a plain white background"
POSE_ID=flux2-klein:refcontrol-pose; BG_ID=flux2-klein:background-remove
say "# Klein adapter golden check $(date -u +%FT%TZ): $(python3 -c "import json; m=json.load(open('$B/Contents/Resources/manifest.json')); print(m['id'], m['version'])")"

$CHECK $B --operation image.edit --model klein-4b --lane bf16 --params '{"size":{"width":512,"height":512}}' --seed 7 \
  --text "instruction=$PROMPT" --adapter "$POSE_ID=1.0" --adapter "$BG_ID=1.1" \
  --adapter-input "${POSE_ID}:pose=$POSE" --adapter-input "${POSE_ID}:reference=$FOX" \
  --models-root "$MODELS" --budget 64000000000 --timeout 900 --session $OUT/edit.gpp > $OUT/edit-gpp.log 2>&1
file=$(grep -o '"path":"[^"]*"' $OUT/edit-gpp.log | head -1 | sed 's/"path":"//; s/"$//; s/\\\//\//g')
[[ -n $file && -f $file ]] && cp $file $OUT/edit-gpp.png
$INPROC --model klein-4b --lane bf16 --prompt "$PROMPT" --width 512 --height 512 --seed 7 \
  --adapter "$POSE_ID=1.0" --adapter "$BG_ID=1.1" --adapter-input "${POSE_ID}:pose=$POSE" --adapter-input "${POSE_ID}:reference=$FOX" \
  --models-root "$MODELS" --out $OUT/edit-inproc.png > $OUT/edit-inproc.json 2>> $LOG
# The control: the same edit with the skeleton and subject as plain references and no adapters.
$INPROC --model klein-4b --lane bf16 --prompt "$PROMPT" --width 512 --height 512 --seed 7 --image $POSE --image $FOX \
  --models-root "$MODELS" --out $OUT/edit-control.png > $OUT/edit-control.json 2>> $LOG

if [[ ! -f $OUT/edit-gpp.png ]]; then fail=$((fail+1)); say "FAIL  edit: no output over GPP ($(grep -E 'FAIL|error' $OUT/edit-gpp.log | tail -1))"
elif [[ ! -f $OUT/edit-inproc.png ]]; then fail=$((fail+1)); say "FAIL  edit: no in-process output"
else
  a=$(shasum -a 256 $OUT/edit-gpp.png | cut -c1-16); b=$(shasum -a 256 $OUT/edit-inproc.png | cut -c1-16)
  ran=$(grep -o 'ran in [0-9.]* s' $OUT/edit-gpp.log | head -1)
  if [[ $a == $b ]]; then pass=$((pass+1)); say "PASS  edit (klein-4b bf16, $POSE_ID ic + $BG_ID plain): byte-identical, sha256 $a…; over GPP $ran; in-process $(cat $OUT/edit-inproc.json)"
  elif $PIXDIFF $OUT/edit-gpp.png $OUT/edit-inproc.png >> $LOG 2>&1; then pass=$((pass+1)); say "PASS  edit: pixel-identical (bytes $a… / $b…)"
  else fail=$((fail+1)); say "FAIL  edit: differs ($a… / $b…): $(tail -1 $LOG)"; fi
  c=$(shasum -a 256 $OUT/edit-control.png 2>/dev/null | cut -c1-16)
  if [[ -n $c && $c != $b ]]; then pass=$((pass+1)); say "PASS  control: without the adapters the edit differs (sha256 $c…)"
  else fail=$((fail+1)); say "FAIL  control: the no-adapter edit is the same image ($c…), so the adapters did nothing"; fi
fi
say "== $pass passed, $fail failed"
exit $(( fail > 0 ))
