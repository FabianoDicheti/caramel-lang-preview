// ============================================================================
// Caramel Language - device:: syntax + @device decorator tests (lang_044)
// ----------------------------------------------------------------------------
// Ticket: lang_044 (Async Jobs Client + device:: Syntax in .crml)
// ----------------------------------------------------------------------------
// Language-surface coverage, no networking:
//   * grammar/parser/AST: device::alias { host/user/pass/token/port } blocks
//     with env("NAME") credentials, both leading and inline @device(alias)
//     decorator forms (bare and keyed), Device::Remote marking
//   * plaintext pass/token literals are ACCEPTED but emit a parser warning
//     that names the field and alias without echoing the secret value
//   * env(NAME) without quotes is a parse error
//   * flows without @device (or with the legacy fpga/cpu targets) resolve to
//     local execution via deviceAliasOf
//   * collectScriptDevices: alias -> DeviceConfig resolution incl. run-time
//     env() reads, port handling, duplicate-alias and missing-host errors
//   * quantFromProgram: crml::quant* directives -> CRPK quant block (dtype
//     selection per PROTOCOL_SPEC.md 6.5: bit width <= 8 -> int8, else int32)
// ============================================================================
#include "caramel/net/dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "caramel/ast/ast.h"
#include "caramel/parse/lexer.h"
#include "caramel/parse/parser.h"

using namespace caramel::net;
namespace ast = caramel::ast;
namespace parse = caramel::parse;
namespace proto = caramel::proto;

static int g_failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);   \
      ++g_failures;                                                 \
    }                                                               \
  } while (0)

// Every .crml program needs the quant preamble; share one spelling.
static const char kPreamble[] =
    "crml::quantmax=1000;\ncrml::quantmin=-1000;\ncrml::quantres=0;\n";

struct Parsed {
  std::unique_ptr<ast::Program> program;
  std::vector<parse::ParseError> errors;
  std::vector<parse::ParseError> warnings;
  bool ok = false;
};

static Parsed parseSrc(const std::string &src) {
  Parsed r;
  parse::Lexer lx(src);
  parse::Parser p(lx.tokenize());
  r.program = p.parse();
  r.ok = p.ok();
  r.errors = p.errors();
  r.warnings = p.warnings();
  return r;
}

static const ast::DeviceBlock *findDevice(const ast::Program &prog,
                                          const std::string &alias) {
  for (const auto &it : prog.items) {
    if (it->kind != ast::NodeKind::DeviceBlock) continue;
    const auto *db = static_cast<const ast::DeviceBlock *>(it.get());
    if (db->alias == alias) return db;
  }
  return nullptr;
}

static const ast::LambdaFlow *findFlow(const ast::Program &prog,
                                       const std::string &name) {
  for (const auto &it : prog.items) {
    if (it->kind != ast::NodeKind::LambdaFlow) continue;
    const auto *lf = static_cast<const ast::LambdaFlow *>(it.get());
    if (lf->name == name) return lf;
  }
  return nullptr;
}

static const ast::DeviceField *findField(const ast::DeviceBlock &db,
                                         const std::string &key) {
  for (const auto &f : db.fields)
    if (f.key == key) return &f;
  return nullptr;
}

// ---------------------------------------------------------------------------
// (a) Grammar/parser/AST: device:: blocks + @device decorator forms
// ---------------------------------------------------------------------------

static void test_device_blocks_and_inline_decorator() {
  const std::string src = std::string(kPreamble) +
      "device::bob   { host = \"192.168.0.42\"; user = \"fabiano\"; "
      "pass = env(\"BOB_PASS\"); }\n"
      "device::alice { host = \"192.168.0.43\"; token = env(\"ALICE_TOKEN\"); "
      "}\n"
      "calc::lambda_flow layer(x, w, bias) @device(bob) {\n"
      "    x w matmul p =\n"
      "    p bias elemwise_add s =\n"
      "    s relu result =\n"
      "} return result;\n";
  Parsed r = parseSrc(src);
  CHECK(r.ok);
  CHECK(r.warnings.empty());  // env() credentials: nothing to warn about

  const auto *bob = findDevice(*r.program, "bob");
  CHECK(bob != nullptr);
  if (bob) {
    CHECK(bob->device == ast::Device::Remote);  // lang_044 form, not fpga/cpu
    CHECK(bob->fields.size() == 3);
    const auto *host = findField(*bob, "host");
    CHECK(host && !host->is_env && host->value == "192.168.0.42");
    const auto *user = findField(*bob, "user");
    CHECK(user && !user->is_env && user->value == "fabiano");
    // env("NAME") stores the VARIABLE NAME; resolution is at run time and
    // the secret never lands in the AST.
    const auto *pass = findField(*bob, "pass");
    CHECK(pass && pass->is_env && pass->value == "BOB_PASS");
  }

  const auto *alice = findDevice(*r.program, "alice");
  CHECK(alice != nullptr);
  if (alice) {
    CHECK(alice->device == ast::Device::Remote);
    const auto *token = findField(*alice, "token");
    CHECK(token && token->is_env && token->value == "ALICE_TOKEN");
  }

  // Inline decorator (between the param list and the body) routes the flow.
  const auto *layer = findFlow(*r.program, "layer");
  CHECK(layer != nullptr);
  if (layer) {
    CHECK(layer->decorators.size() == 1);
    CHECK(layer->decorators[0].name == "device");
    CHECK(deviceAliasOf(*layer) == "bob");
    // Routing does not disturb the flow itself.
    CHECK(layer->params == (std::vector<std::string>{"x", "w", "bias"}));
    CHECK(layer->returns == (std::vector<std::string>{"result"}));
  }
}

static void test_leading_and_keyed_decorator_forms() {
  const std::string src = std::string(kPreamble) +
      "device::alice { host = \"10.0.0.1\"; token = env(\"T\"); }\n"
      "@device(alice)\n"
      "calc::lambda_flow f(x, w) {\n    x w matmul r =\n} return r;\n"
      "calc::lambda_flow g(y, v) @device{target: alice} {\n"
      "    y v matmul s =\n} return s;\n";
  Parsed r = parseSrc(src);
  CHECK(r.ok);
  const auto *f = findFlow(*r.program, "f");
  CHECK(f && deviceAliasOf(*f) == "alice");  // leading, bare argument
  const auto *g = findFlow(*r.program, "g");
  CHECK(g && deviceAliasOf(*g) == "alice");  // inline, keyed target:
}

static void test_flows_without_device_stay_local() {
  const std::string src = std::string(kPreamble) +
      "calc::lambda_flow plain(x, w) {\n    x w matmul r =\n} return r;\n"
      "calc::lambda_flow legacy(y, v) @device(fpga) {\n"
      "    y v matmul s =\n} return s;\n"
      "calc::lambda_flow legacy2(a, b) @device(cpu) {\n"
      "    a b matmul t =\n} return t;\n";
  Parsed r = parseSrc(src);
  CHECK(r.ok);
  const auto *plain = findFlow(*r.program, "plain");
  CHECK(plain && plain->decorators.empty());
  CHECK(plain && deviceAliasOf(*plain).empty());  // no @device -> local
  // Legacy hardware-routing targets are NOT remote aliases (lang_044 leaves
  // them to the local pipeline).
  const auto *legacy = findFlow(*r.program, "legacy");
  CHECK(legacy && deviceAliasOf(*legacy).empty());
  const auto *legacy2 = findFlow(*r.program, "legacy2");
  CHECK(legacy2 && deviceAliasOf(*legacy2).empty());
}

// ---------------------------------------------------------------------------
// (b) Plaintext credentials: accepted, but warned about (and never echoed)
// ---------------------------------------------------------------------------

static void test_plaintext_pass_accepted_with_warning() {
  const std::string src = std::string(kPreamble) +
      "device::bob { host = \"10.0.0.9\"; user = \"u\"; "
      "pass = \"hunter2\"; }\n"
      "calc::lambda_flow f(x, w) @device(bob) {\n"
      "    x w matmul r =\n} return r;\n";
  Parsed r = parseSrc(src);
  CHECK(r.ok);  // accepted: this is a warning, not an error
  CHECK(r.warnings.size() == 1);
  if (!r.warnings.empty()) {
    const std::string &msg = r.warnings[0].message;
    CHECK(msg.find("plaintext pass") != std::string::npos);
    CHECK(msg.find("device::bob") != std::string::npos);
    CHECK(msg.find("env(") != std::string::npos);  // actionable: suggests env
    CHECK(msg.find("hunter2") == std::string::npos);  // never echo the secret
  }
  // The literal still lands in the AST (it is used for the login).
  const auto *bob = findDevice(*r.program, "bob");
  const auto *pass = bob ? findField(*bob, "pass") : nullptr;
  CHECK(pass && !pass->is_env && pass->value == "hunter2");
}

static void test_plaintext_token_warns_env_forms_do_not() {
  const std::string plain = std::string(kPreamble) +
      "device::a { host = \"10.0.0.1\"; token = \"deadbeef\"; }\n";
  Parsed r1 = parseSrc(plain);
  CHECK(r1.ok);
  CHECK(r1.warnings.size() == 1);
  if (!r1.warnings.empty()) {
    CHECK(r1.warnings[0].message.find("plaintext token") != std::string::npos);
    CHECK(r1.warnings[0].message.find("deadbeef") == std::string::npos);
  }

  const std::string env_form = std::string(kPreamble) +
      "device::a { host = \"10.0.0.1\"; user = \"u\"; pass = env(\"P\"); }\n"
      "device::b { host = \"10.0.0.2\"; token = env(\"T\"); }\n";
  Parsed r2 = parseSrc(env_form);
  CHECK(r2.ok);
  CHECK(r2.warnings.empty());

  // Non-secret plaintext fields (host/user/port) never warn.
  const std::string non_secret = std::string(kPreamble) +
      "device::c { host = \"10.0.0.3\"; port = 4780; user = \"me\"; "
      "token = env(\"T\"); }\n";
  Parsed r3 = parseSrc(non_secret);
  CHECK(r3.ok);
  CHECK(r3.warnings.empty());
}

static void test_env_requires_quoted_name() {
  const std::string src = std::string(kPreamble) +
      "device::bob { host = \"10.0.0.9\"; pass = env(BOB_PASS); }\n";
  Parsed r = parseSrc(src);
  CHECK(!r.ok);
  bool found = false;
  for (const auto &e : r.errors) {
    if (e.message.find("env(") != std::string::npos &&
        e.message.find("device::bob") != std::string::npos) {
      found = true;
    }
  }
  CHECK(found);
}

// ---------------------------------------------------------------------------
// (c) collectScriptDevices: AST -> DeviceConfig (env reads at RUN time)
// ---------------------------------------------------------------------------

static void test_collect_script_devices() {
  ::setenv("CARAMEL_TEST_BOB_PASS", "s3cret-pw", 1);
  ::setenv("CARAMEL_TEST_ALICE_TOKEN", "0123456789abcdef0123456789abcdef", 1);
  const std::string src = std::string(kPreamble) +
      "device::bob   { host = \"192.168.0.42\"; user = \"fabiano\"; "
      "pass = env(\"CARAMEL_TEST_BOB_PASS\"); }\n"
      "device::alice { host = \"192.168.0.43\"; port = 5990; "
      "token = env(\"CARAMEL_TEST_ALICE_TOKEN\"); }\n"
      "device::carol { host = \"192.168.0.44:6001\"; "
      "token = env(\"CARAMEL_TEST_ALICE_TOKEN\"); }\n";
  Parsed r = parseSrc(src);
  CHECK(r.ok);

  std::vector<ScriptDevice> devs;
  std::string err;
  CHECK(collectScriptDevices(*r.program, devs, err));
  CHECK(err.empty());
  CHECK(devs.size() == 3);
  if (devs.size() == 3) {
    CHECK(devs[0].alias == "bob");
    CHECK(devs[0].config.alias == "bob");
    CHECK(devs[0].config.host == "192.168.0.42");
    CHECK(devs[0].config.port == 4780);  // spec Section 3 default port
    CHECK(devs[0].config.user == "fabiano");
    CHECK(devs[0].config.pass == "s3cret-pw");  // env read at collection time
    CHECK(devs[1].alias == "alice");
    CHECK(devs[1].config.port == 5990);  // explicit port = field
    CHECK(devs[1].config.token == "0123456789abcdef0123456789abcdef");
    CHECK(devs[2].config.host == "192.168.0.44");
    CHECK(devs[2].config.port == 6001);  // host:port spelling
  }
}

static void test_collect_rejects_bad_blocks() {
  std::vector<ScriptDevice> devs;
  std::string err;

  // Duplicate alias.
  Parsed dup = parseSrc(std::string(kPreamble) +
      "device::bob { host = \"10.0.0.1\"; token = env(\"T\"); }\n"
      "device::bob { host = \"10.0.0.2\"; token = env(\"T\"); }\n");
  CHECK(dup.ok);
  CHECK(!collectScriptDevices(*dup.program, devs, err));
  CHECK(err.find("duplicate") != std::string::npos);
  CHECK(err.find("device::bob") != std::string::npos);

  // Missing host.
  err.clear();
  Parsed nohost = parseSrc(std::string(kPreamble) +
      "device::bob { user = \"u\"; pass = env(\"P\"); }\n");
  CHECK(nohost.ok);
  CHECK(!collectScriptDevices(*nohost.program, devs, err));
  CHECK(err.find("device::bob") != std::string::npos);
  CHECK(err.find("host") != std::string::npos);

  // Invalid port field.
  err.clear();
  Parsed badport = parseSrc(std::string(kPreamble) +
      "device::bob { host = \"10.0.0.1\"; port = 99999; "
      "token = env(\"T\"); }\n");
  CHECK(badport.ok);
  CHECK(!collectScriptDevices(*badport.program, devs, err));
  CHECK(err.find("port") != std::string::npos);
}

// ---------------------------------------------------------------------------
// (d) quantFromProgram: directives -> CRPK quant block (dtype rule)
// ---------------------------------------------------------------------------

static void test_quant_from_program() {
  Parsed wide = parseSrc(std::string(kPreamble) +
      "calc::lambda_flow f(x, w) {\n    x w matmul r =\n} return r;\n");
  CHECK(wide.ok);
  proto::QuantBlock q;
  std::string err;
  CHECK(quantFromProgram(*wide.program, q, err));
  CHECK(q.quantres == 0);
  CHECK(q.quantmin == -1000);
  CHECK(q.quantmax == 1000);
  CHECK(q.dtype == proto::kDTypeInt32);  // 11-bit range -> int32

  // A range that fits 8 signed bits selects the int8 wire dtype.
  Parsed narrow = parseSrc(
      "crml::quantmax=100;\ncrml::quantmin=-100;\ncrml::quantres=0;\n"
      "calc::lambda_flow f(x, w) {\n    x w matmul r =\n} return r;\n");
  CHECK(narrow.ok);
  proto::QuantBlock q8;
  CHECK(quantFromProgram(*narrow.program, q8, err));
  CHECK(q8.dtype == proto::kDTypeInt8);
}

int main() {
  test_device_blocks_and_inline_decorator();
  test_leading_and_keyed_decorator_forms();
  test_flows_without_device_stay_local();
  test_plaintext_pass_accepted_with_warning();
  test_plaintext_token_warns_env_forms_do_not();
  test_env_requires_quoted_name();
  test_collect_script_devices();
  test_collect_rejects_bad_blocks();
  test_quant_from_program();
  if (g_failures == 0) {
    std::printf("OK: all device syntax tests passed\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAILED: %d check(s)\n", g_failures);
  return EXIT_FAILURE;
}
