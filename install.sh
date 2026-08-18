#!/bin/sh
# ============================================================================
# Caramel language installer
# ----------------------------------------------------------------------------
#   curl -fsSL https://raw.githubusercontent.com/FabianoDicheti/caramel-lang-preview/master/install.sh | sh
#
# Downloads the prebuilt caramel-run / caramel-lint / crpk-gen binaries for this
# machine and drops them in a bin directory on PATH. No compiler, no CMake, no
# LLVM. When no prebuilt binary matches the platform, it falls back to building
# from source (needs cmake + a C++17 compiler + git).
#
# Environment overrides:
#   CARAMEL_VERSION=v0.1.0   install a specific release instead of the latest
#   CARAMEL_PREFIX=/usr/local install somewhere else (default: ~/.local)
#   CARAMEL_FROM_SOURCE=1     skip the download and build from source
# ============================================================================
set -eu

REPO="FabianoDicheti/caramel-lang-preview"
PREFIX="${CARAMEL_PREFIX:-$HOME/.local}"
VERSION="${CARAMEL_VERSION:-latest}"
FROM_SOURCE="${CARAMEL_FROM_SOURCE:-0}"
TMPDIR_C=""

say()  { printf '\033[1;33mcaramel\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mcaramel\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mcaramel error\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() { [ -n "$TMPDIR_C" ] && rm -rf "$TMPDIR_C"; }
trap cleanup EXIT INT HUP TERM

have() { command -v "$1" >/dev/null 2>&1; }

# --- Platform detection -----------------------------------------------------
detect_platform() {
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os" in
    Linux)  os_tag="linux" ;;
    Darwin) os_tag="macos" ;;
    *) die "unsupported OS '$os' - build from source: https://github.com/$REPO/blob/master/INSTALL.md" ;;
  esac
  case "$arch" in
    x86_64|amd64)  arch_tag="x86_64" ;;
    aarch64|arm64) arch_tag="arm64" ;;
    *) die "unsupported CPU '$arch' - build from source: https://github.com/$REPO/blob/master/INSTALL.md" ;;
  esac
  PLATFORM="${os_tag}-${arch_tag}"
}

fetch() {  # fetch <url> <dest>
  if have curl; then curl -fsSL "$1" -o "$2"
  elif have wget; then wget -qO "$2" "$1"
  else die "need curl or wget"; fi
}

fetch_stdout() {
  if have curl; then curl -fsSL "$1"
  elif have wget; then wget -qO- "$1"
  else die "need curl or wget"; fi
}

resolve_version() {
  [ "$VERSION" != "latest" ] && return 0
  say "looking up the latest release..."
  VERSION="$(fetch_stdout "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
             | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)" || true
  [ -n "$VERSION" ] || return 1
  return 0
}

# --- Install from a prebuilt release tarball --------------------------------
install_binaries() {
  url="https://github.com/$REPO/releases/download/$VERSION/caramel-$VERSION-$PLATFORM.tar.gz"
  TMPDIR_C="$(mktemp -d)"
  say "downloading caramel $VERSION for $PLATFORM"
  fetch "$url" "$TMPDIR_C/caramel.tar.gz" || return 1
  tar -xzf "$TMPDIR_C/caramel.tar.gz" -C "$TMPDIR_C" || return 1

  src="$(find "$TMPDIR_C" -type d -name bin | head -n 1)"
  [ -n "$src" ] || return 1

  mkdir -p "$PREFIX/bin" "$PREFIX/share/caramel"
  for tool in caramel-run caramel-lint crpk-gen; do
    [ -f "$src/$tool" ] || continue
    install -m 0755 "$src/$tool" "$PREFIX/bin/$tool"
  done
  ex="$(find "$TMPDIR_C" -type d -name examples | head -n 1)"
  [ -n "$ex" ] && cp -R "$ex" "$PREFIX/share/caramel/" 2>/dev/null || true
  return 0
}

# --- Fallback: build from source --------------------------------------------
install_from_source() {
  have git   || die "building from source needs git"
  have cmake || die "building from source needs cmake (apt install cmake / brew install cmake)"
  have c++ || have g++ || have clang++ || die "building from source needs a C++17 compiler (apt install build-essential)"

  TMPDIR_C="${TMPDIR_C:-$(mktemp -d)}"
  say "building from source (this takes a couple of minutes)"
  git clone --depth 1 "https://github.com/$REPO.git" "$TMPDIR_C/src" >/dev/null 2>&1 \
    || die "git clone failed"
  if [ "$VERSION" != "latest" ]; then
    (cd "$TMPDIR_C/src" && git fetch --depth 1 origin "refs/tags/$VERSION:refs/tags/$VERSION" >/dev/null 2>&1 \
      && git checkout -q "$VERSION") || warn "tag $VERSION not found, building the default branch"
  fi
  cmake -S "$TMPDIR_C/src" -B "$TMPDIR_C/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCARAMEL_BUILD_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" >/dev/null || die "cmake configure failed"
  cmake --build "$TMPDIR_C/build" --target caramel-run caramel-lint crpk-gen \
        -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" >/dev/null || die "build failed"
  cmake --install "$TMPDIR_C/build" >/dev/null || die "install failed"
}

# --- PATH advice ------------------------------------------------------------
path_hint() {
  case ":$PATH:" in
    *":$PREFIX/bin:"*) return 0 ;;
  esac
  shell_rc="$HOME/.profile"
  case "${SHELL:-}" in
    */zsh)  shell_rc="$HOME/.zshrc" ;;
    */bash) [ -f "$HOME/.bashrc" ] && shell_rc="$HOME/.bashrc" ;;
  esac
  warn ""
  warn "$PREFIX/bin is not on your PATH. Add it:"
  warn "    echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> $shell_rc"
  warn "    source $shell_rc"
}

main() {
  detect_platform
  if [ "$FROM_SOURCE" = "1" ]; then
    install_from_source
  elif ! resolve_version; then
    warn "no published release found - falling back to a source build"
    install_from_source
  elif ! install_binaries; then
    warn "no prebuilt binary for $PLATFORM at $VERSION - falling back to a source build"
    install_from_source
  fi

  [ -x "$PREFIX/bin/caramel-run" ] || die "installation finished but $PREFIX/bin/caramel-run is missing"

  say "installed to $PREFIX/bin:"
  for tool in caramel-run caramel-lint crpk-gen; do
    [ -x "$PREFIX/bin/$tool" ] && say "  $tool"
  done
  path_hint
  say ""
  say "try it:"
  say "  caramel-run $PREFIX/share/caramel/examples/matmul_relu.crml --flow layer \\"
  say "    --in x=2x2:1,2,3,4 --in w=2x2:5,6,7,8 --in bias=2x2:0,0,0,0"
}

main "$@"
