# Third-Party Native Dependencies

The Zephr Agent native build is prepared by `uv run prepare_native_deps` and
bridged into each checkout by `uv run prepare_dev`.

## Public Runtime Dependencies

- LiteRT is built from `https://github.com/google-ai-edge/LiteRT.git` at the
  pinned `LITERT_SOURCE_REF` in
  `clis/support/native_dependencies/prepare_native_deps.py`.
- SentencePiece is built as a static native dependency with
  `SPM_USE_BUILTIN_PROTOBUF=ON`.
- Platform toolchains come from Xcode on Apple platforms and the Android SDK /
  NDK selected by `clis/support/platform_dependencies/android_config.py`.
