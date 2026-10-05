#!/usr/bin/env bash
# Record the story game's voices again, after editing laserapp/story_words_*.py.
#
# The recordings in laserapp/assets/story/ ship with the app, so this is only
# needed when the story's text changes. The lines are spoken by text-to-speech
# models that run locally; each language has its own, set up on first use in
# its own venv under .build-cache/ (English ~0.5 GB, Ukrainian ~2.5 GB).
#
#   ./build_story_audio.sh                 # lines that changed, both languages
#   ./build_story_audio.sh uk              # one language
#   ./build_story_audio.sh uk --force      # all of it again
#   ./build_story_audio.sh uk --only c1i03,c2f
set -euo pipefail
cd "$(dirname "$0")"

langs="en uk"
case "${1:-}" in en|uk) langs=$1; shift ;; esac

venv() {    # venv DIR - this system's Python ships without pip (see README)
    if [ ! -x "$1/venv/bin/pip" ]; then
        mkdir -p "$1"
        python3 -m venv --without-pip "$1/venv"
        curl -sS https://bootstrap.pypa.io/get-pip.py | "$1/venv/bin/python" - -q
    fi
}

for lang in $langs; do
    if [ "$lang" = en ]; then
        cache=.build-cache/tts
        release=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
        venv "$cache"
        "$cache/venv/bin/python" -c 'import kokoro_onnx' 2>/dev/null \
            || "$cache/venv/bin/pip" install -q "kokoro-onnx>=0.4"
        for f in kokoro-v1.0.onnx voices-v1.0.bin; do
            [ -f "$cache/$f" ] || curl -fsSL -o "$cache/$f" "$release/$f"
        done
        "$cache/venv/bin/python" tools/story_audio.py       # the effects need only numpy
        "$cache/venv/bin/python" tools/story_audio.py --lang en \
            --model "$cache/kokoro-v1.0.onnx" --voices "$cache/voices-v1.0.bin" "$@"
    else
        cache=.build-cache/tts-uk
        space=https://huggingface.co/spaces/patriotyk/styletts2-ukrainian/resolve/main/voices
        venv "$cache"
        if ! "$cache/venv/bin/python" -c 'import styletts2_inference' 2>/dev/null; then
            "$cache/venv/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu \
                torch torchaudio
            "$cache/venv/bin/pip" install -q soundfile librosa scipy einops einops_exts munch \
                pyyaml transformers accelerate huggingface_hub \
                "git+https://github.com/patriotyk/ukrainian-word-stress.git" \
                "git+https://github.com/patriotyk/ipa-uk.git" \
                "git+https://github.com/patriotyk/styletts2-inference@209352700cf703a05ff94501290095dd9072fd5f"
        fi
        # The voices the cast in tools/story_audio.py is made of.
        mkdir -p "$cache/voices"
        "$cache/venv/bin/python" - "$cache/voices" "$space" <<'EOF'
import sys, urllib.parse, urllib.request
from pathlib import Path
sys.path.insert(0, "tools")
from story_audio import CAST
for voice, *_ in CAST["uk"].values():
    path = Path(sys.argv[1]) / f"{voice}.pt"
    if not path.exists():
        urllib.request.urlretrieve(f"{sys.argv[2]}/{urllib.parse.quote(voice)}.pt", path)
EOF
        "$cache/venv/bin/python" -W ignore tools/story_audio.py --lang uk \
            --voices "$cache/voices" "$@"
    fi
done
