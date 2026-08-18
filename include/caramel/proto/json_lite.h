// ============================================================================
// Caramel Language - minimal JSON parsing for the CDP control plane
// ----------------------------------------------------------------------------
// Ticket:   lang_042 (Device Registry + Auth Sessions)
// Version:  1.0.0
// Standard: C++17
// ----------------------------------------------------------------------------
// Purpose-built, header-only parser for the flat JSON objects of
// PROTOCOL_SPEC.md (auth replies, /api/status, error bodies, job states).
// Placed under include/caramel/proto (not buried in src/net) because
// lang_043 reuses it to parse /api/execute and /api/job responses.
// No third-party dependencies.
//
// Supported member value types (everything the spec emits):
//   * string            (with the standard escapes; \uXXXX decodes BMP
//                        code points to UTF-8 - surrogate pairs are rejected,
//                        CDP control-plane strings are ASCII in practice)
//   * integer           (int64 range; JSON floats/exponents and out-of-range
//                        integers are tolerated but surface as Skipped)
//   * boolean, null
//   * array of strings  (the `ops` capability list)
//
// Forward compatibility: unknown keys MUST be ignored by callers, so members
// whose values are nested objects, mixed/nested arrays, floats, or oversized
// integers are consumed conservatively (bounded recursion, depth 16) and kept
// as JsonKind::Skipped rather than failing the whole parse. Whitespace and
// any key order are tolerated. Trailing garbage after the closing '}' fails.
//
// Error handling follows the repo idiom (proto/crpk.h): no exceptions;
// parseJsonObject returns std::nullopt on malformed input, typed getters
// return std::nullopt on a missing key or a type mismatch.
// ============================================================================
#ifndef CARAMEL_PROTO_JSON_LITE_H
#define CARAMEL_PROTO_JSON_LITE_H

#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace caramel::proto {

enum class JsonKind {
  String,
  Int,
  Bool,
  Null,
  StringArray,
  Skipped,  // valid JSON we do not model (float, nested object/array, ...)
};

struct JsonMember {
  std::string key;
  JsonKind kind = JsonKind::Skipped;
  std::string str;                // kind == String
  std::int64_t num = 0;           // kind == Int
  bool boolean = false;           // kind == Bool
  std::vector<std::string> arr;   // kind == StringArray
};

// A parsed flat object. find() returns the first member with the key
// (duplicate keys: first one wins); getters return nullopt on missing key or
// kind mismatch, which is how callers detect "missing or mistyped field".
class JsonObject {
 public:
  std::vector<JsonMember> members;

  const JsonMember *find(const std::string &key) const {
    for (const auto &m : members) {
      if (m.key == key) return &m;
    }
    return nullptr;
  }

  std::optional<std::string> getString(const std::string &key) const {
    const JsonMember *m = find(key);
    if (!m || m->kind != JsonKind::String) return std::nullopt;
    return m->str;
  }

  std::optional<std::int64_t> getInt(const std::string &key) const {
    const JsonMember *m = find(key);
    if (!m || m->kind != JsonKind::Int) return std::nullopt;
    return m->num;
  }

  std::optional<std::uint64_t> getUint(const std::string &key) const {
    const auto v = getInt(key);
    if (!v || *v < 0) return std::nullopt;
    return static_cast<std::uint64_t>(*v);
  }

  std::optional<bool> getBool(const std::string &key) const {
    const JsonMember *m = find(key);
    if (!m || m->kind != JsonKind::Bool) return std::nullopt;
    return m->boolean;
  }

  std::optional<std::vector<std::string>> getStringArray(
      const std::string &key) const {
    const JsonMember *m = find(key);
    if (!m || m->kind != JsonKind::StringArray) return std::nullopt;
    return m->arr;
  }
};

namespace jsondetail {

inline void skipWs(const std::string &s, std::size_t &i) {
  while (i < s.size() &&
         (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) {
    ++i;
  }
}

inline int hexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Parse a JSON string at s[i] == '"' into `out`; advances i past the closing
// quote. Raw control characters (< 0x20) and unknown escapes fail.
inline bool parseString(const std::string &s, std::size_t &i, std::string &out) {
  if (i >= s.size() || s[i] != '"') return false;
  ++i;
  out.clear();
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == '"') {
      ++i;
      return true;
    }
    if (c == '\\') {
      if (i + 1 >= s.size()) return false;
      const char e = s[i + 1];
      i += 2;
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (i + 4 > s.size()) return false;
          int cp = 0;
          for (int k = 0; k < 4; ++k) {
            const int h = hexVal(s[i + static_cast<std::size_t>(k)]);
            if (h < 0) return false;
            cp = cp * 16 + h;
          }
          i += 4;
          if (cp >= 0xD800 && cp <= 0xDFFF) return false;  // no surrogates
          if (cp < 0x80) {
            out += static_cast<char>(cp);
          } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          }
          break;
        }
        default:
          return false;
      }
      continue;
    }
    if (c < 0x20) return false;
    out += static_cast<char>(c);
    ++i;
  }
  return false;  // unterminated string
}

// Skip any well-formed JSON value (bounded recursion). Used for unknown
// nested objects/arrays and other shapes we keep as Skipped. Numbers are
// consumed leniently (digits/./e/E/+/-).
inline bool skipValue(const std::string &s, std::size_t &i, int depth) {
  if (depth <= 0) return false;
  skipWs(s, i);
  if (i >= s.size()) return false;
  const char c = s[i];
  if (c == '"') {
    std::string dummy;
    return parseString(s, i, dummy);
  }
  if (c == '{') {
    ++i;
    skipWs(s, i);
    if (i < s.size() && s[i] == '}') {
      ++i;
      return true;
    }
    for (;;) {
      std::string key;
      skipWs(s, i);
      if (!parseString(s, i, key)) return false;
      skipWs(s, i);
      if (i >= s.size() || s[i] != ':') return false;
      ++i;
      if (!skipValue(s, i, depth - 1)) return false;
      skipWs(s, i);
      if (i >= s.size()) return false;
      if (s[i] == ',') {
        ++i;
        continue;
      }
      if (s[i] == '}') {
        ++i;
        return true;
      }
      return false;
    }
  }
  if (c == '[') {
    ++i;
    skipWs(s, i);
    if (i < s.size() && s[i] == ']') {
      ++i;
      return true;
    }
    for (;;) {
      if (!skipValue(s, i, depth - 1)) return false;
      skipWs(s, i);
      if (i >= s.size()) return false;
      if (s[i] == ',') {
        ++i;
        continue;
      }
      if (s[i] == ']') {
        ++i;
        return true;
      }
      return false;
    }
  }
  if (c == 't') {
    if (s.compare(i, 4, "true") != 0) return false;
    i += 4;
    return true;
  }
  if (c == 'f') {
    if (s.compare(i, 5, "false") != 0) return false;
    i += 5;
    return true;
  }
  if (c == 'n') {
    if (s.compare(i, 4, "null") != 0) return false;
    i += 4;
    return true;
  }
  if (c == '-' || (c >= '0' && c <= '9')) {
    if (c == '-') ++i;
    bool any = false;
    while (i < s.size() &&
           ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' ||
            s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
      ++i;
      any = true;
    }
    return any;
  }
  return false;
}

}  // namespace jsondetail

// Parse one flat JSON object per the header comment. nullopt on malformed
// input (including trailing non-whitespace after the closing brace).
inline std::optional<JsonObject> parseJsonObject(const std::string &text) {
  using namespace jsondetail;
  constexpr int kSkipDepth = 16;
  std::size_t i = 0;
  skipWs(text, i);
  if (i >= text.size() || text[i] != '{') return std::nullopt;
  ++i;
  JsonObject obj;
  skipWs(text, i);
  if (i < text.size() && text[i] == '}') {
    ++i;
  } else {
    for (;;) {
      JsonMember m;
      skipWs(text, i);
      if (!parseString(text, i, m.key)) return std::nullopt;
      skipWs(text, i);
      if (i >= text.size() || text[i] != ':') return std::nullopt;
      ++i;
      skipWs(text, i);
      if (i >= text.size()) return std::nullopt;
      const char c = text[i];
      if (c == '"') {
        if (!parseString(text, i, m.str)) return std::nullopt;
        m.kind = JsonKind::String;
      } else if (c == 't' || c == 'f') {
        if (!skipValue(text, i, 1)) return std::nullopt;  // validates literal
        m.kind = JsonKind::Bool;
        m.boolean = (c == 't');
      } else if (c == 'n') {
        if (!skipValue(text, i, 1)) return std::nullopt;
        m.kind = JsonKind::Null;
      } else if (c == '-' || (c >= '0' && c <= '9')) {
        // Integer fast path; floats/exponents and int64 overflow fall back
        // to Skipped (they only ever appear in unknown, ignorable fields).
        const std::size_t start = i;
        std::size_t j = i;
        const bool neg = (text[j] == '-');
        if (neg) ++j;
        bool any = false;
        bool fits = true;
        std::uint64_t mag = 0;
        while (j < text.size() && text[j] >= '0' && text[j] <= '9') {
          any = true;
          if (mag > (std::numeric_limits<std::uint64_t>::max() - 9) / 10) {
            fits = false;
          } else {
            mag = mag * 10 + static_cast<std::uint64_t>(text[j] - '0');
          }
          ++j;
        }
        const bool integral =
            any && (j >= text.size() ||
                    (text[j] != '.' && text[j] != 'e' && text[j] != 'E'));
        const std::uint64_t lim =
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()) +
            (neg ? 1u : 0u);
        if (integral && fits && mag <= lim) {
          m.kind = JsonKind::Int;
          if (neg && mag == lim) {
            m.num = std::numeric_limits<std::int64_t>::min();
          } else {
            m.num = neg ? -static_cast<std::int64_t>(mag)
                        : static_cast<std::int64_t>(mag);
          }
          i = j;
        } else {
          i = start;
          if (!skipValue(text, i, 1)) return std::nullopt;
          m.kind = JsonKind::Skipped;
        }
      } else if (c == '[') {
        // Flat array of strings (the `ops` list); anything else - empty of
        // meaning to us - is consumed whole and kept as Skipped.
        const std::size_t start = i;
        std::size_t j = i + 1;
        skipWs(text, j);
        bool is_str_array = false;
        std::vector<std::string> items;
        if (j < text.size() && text[j] == ']') {
          is_str_array = true;
          ++j;
        } else if (j < text.size() && text[j] == '"') {
          is_str_array = true;
          for (;;) {
            std::string item;
            if (!parseString(text, j, item)) {
              is_str_array = false;
              break;
            }
            items.push_back(std::move(item));
            skipWs(text, j);
            if (j < text.size() && text[j] == ',') {
              ++j;
              skipWs(text, j);
              if (j >= text.size() || text[j] != '"') {
                is_str_array = false;  // mixed array -> Skipped
                break;
              }
              continue;
            }
            if (j < text.size() && text[j] == ']') {
              ++j;
              break;
            }
            is_str_array = false;
            break;
          }
        }
        if (is_str_array) {
          m.kind = JsonKind::StringArray;
          m.arr = std::move(items);
          i = j;
        } else {
          i = start;
          if (!skipValue(text, i, kSkipDepth)) return std::nullopt;
          m.kind = JsonKind::Skipped;
        }
      } else if (c == '{') {
        if (!skipValue(text, i, kSkipDepth)) return std::nullopt;
        m.kind = JsonKind::Skipped;
      } else {
        return std::nullopt;
      }
      obj.members.push_back(std::move(m));
      skipWs(text, i);
      if (i >= text.size()) return std::nullopt;
      if (text[i] == ',') {
        ++i;
        continue;
      }
      if (text[i] == '}') {
        ++i;
        break;
      }
      return std::nullopt;
    }
  }
  skipWs(text, i);
  if (i != text.size()) return std::nullopt;  // trailing garbage
  return obj;
}

// Escape a string for embedding in a JSON double-quoted literal (used to
// build the /api/auth body; must round-trip arbitrary passwords).
inline std::string jsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (const char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += ch;
        }
    }
  }
  return out;
}

}  // namespace caramel::proto

#endif  // CARAMEL_PROTO_JSON_LITE_H
