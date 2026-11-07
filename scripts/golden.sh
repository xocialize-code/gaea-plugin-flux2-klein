#!/bin/zsh
# Sizes with FlowUpscaler (AB-T-0464), then adapters over GPP against in-process MLXServeEngine (AB-T-0381): a seeded Klein 4B edit with one ic adapter
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

# Sizes (0.3.0, AB-T-0464): past 1024² the worker draws a quarter or a sixteenth of the area and FlowUpscaler doubles
# it once or twice. Each case over GPP against klein-inproc (which plans the size the same way), byte for byte; the
# output is the asked size, and effective reports the draw and the passes.
size_case() { # name, model, operation, seed, width, height, passes, [extra gpp args…] (inproc args follow --)
  local name=$1 model=$2 op=$3 seed=$4 w=$5 h=$6 passes=$7; shift 7
  local gpp=() inp=(); while (( $# )) && [[ $1 != -- ]]; do gpp+=($1); shift; done; [[ ${1:-} == -- ]] && shift; inp=($@)
  local text=prompt; [[ $op == image.edit ]] && text=instruction
  $CHECK $B --operation $op --model $model --lane bf16 --params "{\"size\":{\"width\":$w,\"height\":$h}}" --seed $seed \
    --text "$text=$SIZEPROMPT" "${gpp[@]}" --models-root "$MODELS" --budget 64000000000 --timeout 1800 --session $OUT/$name.gpp > $OUT/$name-gpp.log 2>&1
  local file=$(grep -o '"path":"[^"]*"' $OUT/$name-gpp.log | head -1 | sed 's/"path":"//; s/"$//; s/\\\//\//g')
  [[ -n $file && -f $file ]] && cp $file $OUT/$name-gpp.png
  $INPROC --model $model --lane bf16 --prompt "$SIZEPROMPT" --width $w --height $h --seed $seed "${inp[@]}" \
    --models-root "$MODELS" --out $OUT/$name-inproc.png > $OUT/$name-inproc.json 2>> $LOG
  if [[ ! -f $OUT/$name-gpp.png ]]; then fail=$((fail+1)); say "FAIL  $name: no output over GPP ($(grep -E 'FAIL|error' $OUT/$name-gpp.log | tail -1))"; return; fi
  local dims=$(sips -g pixelWidth -g pixelHeight $OUT/$name-gpp.png | awk '/pixel/{print $2}' | tr '\n' x)
  if [[ $dims == ${w}x${h}x ]]; then pass=$((pass+1)); say "PASS  $name: ${w}×${h} out"; else fail=$((fail+1)); say "FAIL  $name: $dims, not ${w}×${h}"; fi
  if (( passes > 0 )); then
    if grep -q "\"passes\":$passes" $OUT/$name-gpp.log && grep -q 'x-klein.upscaled' $OUT/$name-gpp.log; then pass=$((pass+1)); say "PASS  $name: effective reports $passes upscale pass(es)"
    else fail=$((fail+1)); say "FAIL  $name: effective doesn't report the upscale"; fi
  fi
  local a=$(shasum -a 256 $OUT/$name-gpp.png | cut -c1-16) b=$(shasum -a 256 $OUT/$name-inproc.png 2>/dev/null | cut -c1-16)
  local ran=$(grep -o 'ran in [0-9.]* s' $OUT/$name-gpp.log | head -1)
  if [[ $a == $b ]]; then pass=$((pass+1)); say "PASS  $name: byte-identical to in-process, sha256 $a…; over GPP $ran; in-process $(cat $OUT/$name-inproc.json)"
  elif $PIXDIFF $OUT/$name-gpp.png $OUT/$name-inproc.png >> $LOG 2>&1; then pass=$((pass+1)); say "PASS  $name: pixel-identical (bytes $a… / $b…)"
  else fail=$((fail+1)); say "FAIL  $name: differs from in-process ($a… / $b…)"; fi
}
SIZEPROMPT="An enamel shop sign on a brick wall that reads 'BAKERY - FRESH BREAD DAILY', a tabby cat asleep on the windowsill beside it"
say "# sizes"
size_case gen-1024 klein-4b image.generate 44 1024 1024 0
size_case gen-2048 klein-4b image.generate 44 2048 2048 1
size_case gen-4096 klein-4b image.generate 44 4096 4096 2
size_case gen-1536x1024 klein-4b image.generate 5 1536 1024 1
size_case base-1792x1024 klein-4b-base image.generate 9 1792 1024 1
size_case edit-2048 klein-4b image.edit 7 2048 2048 1 --input "images=$FOX" -- --image $FOX

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
