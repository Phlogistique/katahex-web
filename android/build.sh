#!/bin/sh
# Builds the apk: the ui as one inlined page, then the app around it.
#
#   VASTAI_API_KEY=... android/build.sh      # -> android/app/build/outputs/apk/release/
#
# The key ends up inside the apk. Keep the apk to yourself.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
: "${VASTAI_API_KEY:?set VASTAI_API_KEY, the app rents its GPU with it}"
: "${ANDROID_HOME:=/opt/android-sdk}"
export ANDROID_HOME

npm --prefix "$HERE/../ui" ci
npm --prefix "$HERE/../ui" run build
cp "$HERE/../ui/dist/index.html" "$HERE/app/src/main/assets/index.html"

cd "$HERE"
gradle --no-daemon -q assembleRelease
ls -l app/build/outputs/apk/release/*.apk
