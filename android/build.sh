#!/bin/sh
# Builds the apk: the ui as one inlined page, then the app around it.
#
#   android/build.sh                         # asks for the vast.ai key on first start
#   VASTAI_API_KEY=... android/build.sh      # key built in: keep that apk to yourself
#
# -> android/app/build/outputs/apk/release/app-release.apk
#    android/worker/build/outputs/apk/release/worker-release.apk
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
: "${ANDROID_HOME:=/opt/android-sdk}"
export ANDROID_HOME

npm --prefix "$HERE/../ui" ci
npm --prefix "$HERE/../ui" run build
cp "$HERE/../ui/dist/index.html" "$HERE/app/src/main/assets/index.html"

cd "$HERE"
gradle --no-daemon -q assembleRelease
ls -l app/build/outputs/apk/release/*.apk worker/build/outputs/apk/release/*.apk
