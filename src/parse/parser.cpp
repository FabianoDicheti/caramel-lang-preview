// ============================================================================
// Caramel Language - RPN Parser implementation
// ----------------------------------------------------------------------------
// Ticket:  lang_005 (RPN Parser Implementation)
// Version: 1.0.0
// ============================================================================
#include "caramel/parse/parser.h"

#include <algorithm>
#include <set>

#include "caramel/ops/op_signature.h"
#include "caramel/parse/decorator.h"
#include "caramel/parse/literal_shape.h"

namespace caramel::parse {

using namespace caramel::ast;

namespace {

const std::set<std::string>& stack_ops() {
  static const std::set<std::string> s = {"dup", "swap", "drop", "over", "rot", "-rot"};
  return s;
}
const std::set<std::string>& combinators() {
  static const std::set<std::string> s = {
      "identity", "kestrel", "kite", "mockingbird", "bluebird", "cardinal",
      "starling", "thrush", "warbler", "owl", "compose", "bluebird_prime",
      "blackbird", "psi", "phoenix", "vireo", "ycombinator"};
  return s;
}
const std::set<std::string>& arith_ops() {
  static const std::set<std::string> s = {"add", "sub", "mul", "div"};
  return s;
}
const std::set<std::string>& reserved_words() {
  static const std::set<std::string> s = {
      "crml", "calc", "oc", "ot", "os", "port", "receiver", "responser",
      "memory", "quantmax", "quantmin", "quantres", "quantrange", "quant",
      "decimal_places", "quantmax_fwd", "quantmin_fwd", "quantres_fwd",
      "quantmax_grad", "quantmin_grad", "quantres_grad", "enable_autodiff",
      "autodiff", "gradient_checkpointing", "checkpoint_every",
      "routing_policy", "register_backward", "backpropagation", "device",
      "fpga", "cpu", "write", "lambda_flow", "lambda_symphony",
      "lambda_calculator", "channels", "return", "lambda", "end", "true",
      "false", "unknown", "number", "tensor", "text", "string",
      "dictionary", "dup", "swap", "drop", "over", "rot", "-rot", "add",
      "sub", "mul", "div", "and", "or", "not", "xor", "identity",
      "kestrel", "kite", "mockingbird", "bluebird", "cardinal",
      "starling", "thrush", "warbler", "owl", "compose",
      "bluebird_prime", "blackbird", "psi", "phoenix", "vireo",
      "ycombinator", "matmul", "conv1d", "conv2d", "conv3d",
      "elemwise_add", "elemwise_sub", "elemwise_mul", "elemwise_div",
      "scalar_mul", "relu", "sigmoid", "tanh", "softmax", "gelu", "silu",
      "softplus", "maxpool2d", "avgpool2d", "adaptive_avgpool2d",
      "batchnorm", "layernorm", "instancenorm", "groupnorm", "transpose",
      "reshape", "concat", "split", "tensor_sum", "tensor_mean",
      "tensor_max", "tensor_min", "argmax", "determinant", "inverse",
      "eigendecomp", "svd", "qr", "cholesky", "lu", "fft", "ifft",
      "fft2d", "scaled_dot_attention", "multihead_attention",
      "selective_scan", "s4_layer", "cross_entropy", "clip_gradient",
      "exp", "sqrt", "embedding_lookup", "tensor.init", "txt.read",
      "get_timestamp", "status", "profile", "zeros", "ones", "full", "eye",
      "diag", "range", "random", "band", "tridiag", "from_spectrum"};
  return s;
}
const std::set<std::string>& ref_namespaces() {
  static const std::set<std::string> s = {"port", "receiver", "responser", "memory"};
  return s;
}

// Deep-copy the expression node kinds the reducer produces (for `dup`).
ExprPtr clone_expr(const Expr* e);

template <typename T>
ExprPtr clone_as(const Expr* e) {
  return std::make_unique<T>(*static_cast<const T*>(e));
}

ExprPtr clone_expr(const Expr* e) {
  if (!e) return nullptr;
  switch (e->kind) {
    case NodeKind::NumberLiteral: return clone_as<NumberLiteral>(e);
    case NodeKind::StringLiteral: return clone_as<StringLiteral>(e);
    case NodeKind::VarRef: return clone_as<VarRef>(e);
    case NodeKind::QualifiedRef: return clone_as<QualifiedRef>(e);
    case NodeKind::CombinatorRef: return clone_as<CombinatorRef>(e);
    case NodeKind::OpApplication: {
      const auto* o = static_cast<const OpApplication*>(e);
      auto c = std::make_unique<OpApplication>();
      c->op = o->op;
      c->op_class = o->op_class;
      c->loc = o->loc;
      for (const auto& a : o->args) c->args.push_back(clone_expr(a.get()));
      return c;
    }
    case NodeKind::PropertyRef: {
      const auto* p = static_cast<const PropertyRef*>(e);
      auto c = std::make_unique<PropertyRef>();
      c->path = p->path;
      c->loc = p->loc;
      c->base = clone_expr(p->base.get());
      return c;
    }
    case NodeKind::IndexedRef: {
      const auto* r = static_cast<const IndexedRef*>(e);
      auto c = std::make_unique<IndexedRef>();
      c->indices = r->indices;
      c->loc = r->loc;
      c->base = clone_expr(r->base.get());
      return c;
    }
    default: {
      // Fallback: an opaque placeholder var (keeps dup total without crashing).
      auto v = std::make_unique<VarRef>();
      v->name = "<dup>";
      return v;
    }
  }
}

OpClass classify(const std::string& name) {
  if (arith_ops().count(name)) return OpClass::Arithmetic;
  const auto* sig = caramel::ops::OpRegistry::instance().lookup(name);
  if (sig) {
    using C = caramel::ops::OpCategory;
    switch (sig->category) {
      case C::Arithmetic: return OpClass::Arithmetic;
      case C::Comparison: return OpClass::Comparison;
      case C::Logic: return OpClass::Logic;
      default: return OpClass::Primitive;
    }
  }
  return OpClass::Primitive;
}

}  // namespace

// --------------------------------------------------------------------------
// token cursor
// --------------------------------------------------------------------------
const Token& Parser::peek(int ahead) const {
  std::size_t p = pos_ + static_cast<std::size_t>(ahead);
  if (p >= toks_.size()) return toks_.back();  // EOF sentinel
  return toks_[p];
}
const Token& Parser::advance() {
  const Token& t = toks_[pos_];
  if (pos_ + 1 < toks_.size()) ++pos_;
  return t;
}
bool Parser::match(TokKind k) {
  if (check(k)) { advance(); return true; }
  return false;
}
void Parser::skip_newlines() {
  while (check(TokKind::Newline)) advance();
}
void Parser::error(const std::string& msg) {
  errors_.push_back({msg, cur().loc});
}
void Parser::warning(const std::string& msg) {
  warnings_.push_back({msg, cur().loc});
}

bool Parser::is_quant_directive_at_cursor() const {
  if (!(check(TokKind::Identifier) && cur().text == "crml" &&
        peek(1).kind == TokKind::ColonColon && peek(2).kind == TokKind::Identifier))
    return false;
  const std::string& key = peek(2).text;
  return key.rfind("quant", 0) == 0 || key == "decimal_places";
}

int Parser::arity_of(const std::string& name) const {
  int a = caramel::ops::OpRegistry::instance().arity_of(name);
  if (a >= 0) return a;
  auto it = user_fn_arity_.find(name);
  if (it != user_fn_arity_.end()) return it->second;
  return -1;
}

// --------------------------------------------------------------------------
// top level
// --------------------------------------------------------------------------
std::unique_ptr<Program> Parser::parse() {
  auto program = std::make_unique<Program>();
  skip_newlines();
  if (!at_end() && !is_quant_directive_at_cursor())
    error("program must start with a quantization directive");
  while (!at_end()) {
    // leading decorators (lang_006) attach to the next definition / statement
    auto decs = parse_decorators();
    skip_newlines();
    if (at_end()) break;
    // calc::lambda_flow / lambda_calculator
    if (check(TokKind::Identifier) && cur().text == "calc" &&
        peek(1).kind == TokKind::ColonColon) {
      const std::string& kind = peek(2).text;
      if (kind == "lambda_flow") {
        if (auto f = parse_lambda_flow()) {
          // lang_044: parse_lambda_flow may have collected inline decorators
          // (after the param list); leading decorators come first.
          for (auto& d : f->decorators) decs.push_back(std::move(d));
          f->decorators = std::move(decs);
          program->items.push_back(std::move(f));
        }
      } else if (kind == "lambda_calculator") {
        if (auto c = parse_lambda_calculator()) program->items.push_back(std::move(c));
      } else {
        // lambda_symphony etc. deferred: skip its statement/line tolerantly.
        collect_statement_tokens();
      }
      skip_newlines();
      continue;
    }
    // crml:: directive
    if (check(TokKind::Identifier) && cur().text == "crml" &&
        peek(1).kind == TokKind::ColonColon) {
      if (auto d = parse_directive()) program->items.push_back(std::move(d));
      skip_newlines();
      continue;
    }
    // lang_044: device::alias { ... } remote-worker definition (decorators do
    // not apply to device blocks; any leading ones are dropped).
    if (check(TokKind::Identifier) && cur().text == "device" &&
        peek(1).kind == TokKind::ColonColon) {
      if (auto db = parse_device_block()) program->items.push_back(std::move(db));
      skip_newlines();
      continue;
    }
    // in::name = <literal>;  script-embedded input value (scalar or tensor
    // literal) that fills a flow parameter, so no --in is needed. --in still
    // overrides. Runs locally and remotely (it becomes a normal input).
    if (check(TokKind::Identifier) && cur().text == "in" &&
        peek(1).kind == TokKind::ColonColon) {
      auto line = collect_statement_tokens();   // whole `in :: name = ... ;`
      // tokens: in :: name = <value...> ;   -> name at [2], value = [4..end)
      if (line.size() >= 5 && line[3].kind == TokKind::Assign) {
        auto d = std::make_unique<Directive>();
        d->dkind = DirectiveKind::Input;
        d->loc.begin = line[0].loc;   // the `in` token, so lint diagnostics locate
        d->key = line[2].text;
        std::vector<Token> val(line.begin() + 4, line.end());
        d->valueExpr = reduce(val);
        program->items.push_back(std::move(d));
      } else {
        error("in:: expects  in::name = <literal>;");
      }
      skip_newlines();
      continue;
    }
    // status::alias;  host-side query that prints a declared worker's
    // GET /api/status metadata. Not a dataflow op; handled by the CLI driver.
    if (check(TokKind::Identifier) && cur().text == "status" &&
        peek(1).kind == TokKind::ColonColon) {
      auto line = collect_statement_tokens();   // whole `status :: alias ;`
      // tokens: status :: alias   -> alias at [2]
      if (line.size() >= 3 && line[2].kind == TokKind::Identifier) {
        auto d = std::make_unique<Directive>();
        d->dkind = DirectiveKind::Status;
        d->loc.begin = line[0].loc;   // the `status` token, for diagnostics
        d->key = line[2].text;
        program->items.push_back(std::move(d));
      } else {
        error("status:: expects  status::alias;");
      }
      skip_newlines();
      continue;
    }
    // profile::name;  host-side algebraic profile of an input matrix. Like
    // status::, this is an introspection action handled by the CLI driver, not
    // a dataflow op.
    if (check(TokKind::Identifier) && cur().text == "profile" &&
        peek(1).kind == TokKind::ColonColon) {
      auto line = collect_statement_tokens();   // whole `profile :: name ;`
      if (line.size() >= 3 && line[2].kind == TokKind::Identifier) {
        auto d = std::make_unique<Directive>();
        d->dkind = DirectiveKind::Profile;
        d->loc.begin = line[0].loc;
        d->key = line[2].text;
        program->items.push_back(std::move(d));
      } else {
        error("profile:: expects  profile::name;");
      }
      skip_newlines();
      continue;
    }
    // otherwise: a top-level statement (decorators attach to it)
    pending_decorators_ = std::move(decs);
    auto line = collect_statement_tokens();
    if (!line.empty()) {
      if (auto s = parse_statement(std::move(line))) program->items.push_back(std::move(s));
    }
    pending_decorators_.clear();
    skip_newlines();
  }
  if (!seen_any_quant_) {
    error("missing quantization preamble");
  } else {
    if (!seen_quantres_)
      error("quantization preamble must include quantres or decimal_places");
    if (!(seen_quantrange_ || seen_legacy_quant_ || (seen_quantmax_ && seen_quantmin_)))
      error("quantization preamble must include quantmax+quantmin, quantrange, or quant");
  }
  return program;
}

std::unique_ptr<Directive> Parser::parse_directive() {
  auto d = std::make_unique<Directive>();
  d->loc.begin = cur().loc;
  advance();                 // crml
  match(TokKind::ColonColon);
  if (!check(TokKind::Identifier)) { error("expected directive key after crml::"); return nullptr; }
  d->key = advance().text;
  if (d->key == "register_backward") {
    d->dkind = DirectiveKind::RegisterBackward;
    match(TokKind::LParen);
    while (!check(TokKind::RParen) && !at_end()) {
      if (check(TokKind::Identifier)) d->args.push_back(advance().text);
      else advance();
    }
    match(TokKind::RParen);
    match(TokKind::Semicolon);
    return d;
  }
  // key = value ;
  if (match(TokKind::Assign)) {
    if (check(TokKind::Number) || check(TokKind::Identifier)) d->value = advance().text;
    if (d->key.rfind("quant", 0) == 0 || d->key == "decimal_places") {
      d->dkind = DirectiveKind::Quant;
      seen_any_quant_ = true;
      if (d->key == "quantres" || d->key == "decimal_places") seen_quantres_ = true;
      if (d->key == "quantmax") seen_quantmax_ = true;
      if (d->key == "quantmin") seen_quantmin_ = true;
      if (d->key == "quantrange") seen_quantrange_ = true;
      if (d->key == "quant") seen_legacy_quant_ = true;
    } else if (d->key == "checkpoint_every") {
      d->dkind = DirectiveKind::Checkpoint;
    } else {
      d->dkind = DirectiveKind::Autodiff;
    }
  }
  match(TokKind::Semicolon);
  return d;
}

// lang_045: re-join a hyphen-split name (see parser.h). Appends every
// '-'-prefixed Identifier/Number token that starts exactly where the
// previous one ended (same line, no gap).
void Parser::absorb_hyphen_suffix(const Token& first, std::string& name) {
  uint32_t line = first.loc.line;
  uint32_t next_col = first.loc.column + (uint32_t)first.text.size();
  while ((check(TokKind::Identifier) || check(TokKind::Number)) &&
         !cur().text.empty() && cur().text[0] == '-' &&
         cur().loc.line == line && cur().loc.column == next_col) {
    next_col += (uint32_t)cur().text.size();
    name += advance().text;
  }
}

// lang_044: `device :: alias { key = value ; ... } [;]` where value is a
// string literal, a number, or env("NAME") (read at RUN time, never here).
// A plaintext `pass`/`token` string is accepted but emits a warning.
std::unique_ptr<DeviceBlock> Parser::parse_device_block() {
  auto db = std::make_unique<DeviceBlock>();
  db->loc.begin = cur().loc;
  db->device = Device::Remote;
  advance();                 // device
  match(TokKind::ColonColon);
  if (check(TokKind::Identifier)) {
    Token first = advance();
    db->alias = first.text;
    absorb_hyphen_suffix(first, db->alias);  // lang_045: "bob-i3" aliases
  }
  else error("expected device alias after device::");
  skip_newlines();
  if (!match(TokKind::LBrace)) {
    error("expected '{' to open device::" + db->alias + " block");
    return db;
  }
  skip_newlines();
  while (!check(TokKind::RBrace) && !at_end()) {
    if (!check(TokKind::Identifier)) { advance(); continue; }
    DeviceField field;
    field.loc.begin = cur().loc;
    field.key = advance().text;
    if (!match(TokKind::Assign)) {
      error("expected '=' after device field '" + field.key + "' in device::" +
            db->alias);
      while (!check(TokKind::Semicolon) && !check(TokKind::RBrace) && !at_end())
        advance();
      match(TokKind::Semicolon);
      skip_newlines();
      continue;
    }
    if (check(TokKind::Identifier) && cur().text == "env") {
      // env("NAME") -- record the variable NAME; resolution happens at run
      // time so the secret never lands in the AST.
      advance();  // env
      match(TokKind::LParen);
      if (check(TokKind::String)) {
        field.is_env = true;
        field.value = advance().text;
      } else {
        error("env(...) for device field '" + field.key + "' in device::" +
              db->alias + " requires a quoted environment-variable name");
      }
      match(TokKind::RParen);
    } else if (check(TokKind::String)) {
      field.value = advance().text;
      if (field.key == "pass" || field.key == "token") {
        warning("plaintext " + field.key + " in device::" + db->alias +
                " - use " + field.key +
                " = env(\"NAME\") to keep secrets out of source files");
      }
    } else if (check(TokKind::Number)) {
      field.value = advance().text;
    } else if (check(TokKind::Identifier)) {
      field.value = advance().text;  // bare word value, tolerated
    } else {
      error("expected a value after '=' for device field '" + field.key +
            "' in device::" + db->alias);
    }
    db->fields.push_back(std::move(field));
    while (!check(TokKind::Semicolon) && !check(TokKind::RBrace) &&
           !check(TokKind::Newline) && !at_end())
      advance();
    match(TokKind::Semicolon);
    skip_newlines();
  }
  match(TokKind::RBrace);
  match(TokKind::Semicolon);
  return db;
}

std::vector<Decorator> Parser::parse_decorators() {
  std::vector<Decorator> decs;
  for (;;) {
    skip_newlines();
    if (!(check(TokKind::Identifier) && cur().text.size() > 1 && cur().text[0] == '@'))
      break;
    if (cur().text == "@clock") break;  // block-level, handled as ClockSection
    Decorator d;
    d.name = cur().text.substr(1);  // strip '@'
    d.loc.begin = cur().loc;
    advance();
    // optional ( ... ) or { ... } arguments
    if (check(TokKind::LParen) || check(TokKind::LBrace)) {
      TokKind close = check(TokKind::LParen) ? TokKind::RParen : TokKind::RBrace;
      advance();
      auto args = std::make_unique<ObjectParams>();
      while (!check(close) && !at_end()) {
        if (check(TokKind::Identifier) && peek(1).kind == TokKind::Colon) {
          ObjectField f;
          f.key = advance().text;  // key
          advance();               // ':'
          if (check(TokKind::Number)) {
            auto n = std::make_unique<NumberLiteral>();
            n->lexeme = cur().text; n->is_float = cur().is_float; advance();
            f.value = std::move(n);
          } else if (check(TokKind::Identifier)) {
            auto vr = std::make_unique<VarRef>(); vr->name = advance().text;
            f.value = std::move(vr);
          }
          args->fields.push_back(std::move(f));
          while (!check(TokKind::Comma) && !check(close) && !at_end()) advance();
        } else if (check(TokKind::Identifier) || check(TokKind::Number)) {
          // lang_044: bare positional argument, e.g. @device(bob). Stored as
          // an ObjectField with an EMPTY key.
          ObjectField f;
          if (check(TokKind::Number)) {
            auto n = std::make_unique<NumberLiteral>();
            n->lexeme = cur().text; n->is_float = cur().is_float; advance();
            f.value = std::move(n);
          } else {
            auto vr = std::make_unique<VarRef>();
            Token first = advance();
            vr->name = first.text;
            absorb_hyphen_suffix(first, vr->name);  // lang_045: @device(bob-i3)
            f.value = std::move(vr);
          }
          args->fields.push_back(std::move(f));
          while (!check(TokKind::Comma) && !check(close) && !at_end()) advance();
        } else {
          advance();
        }
        if (check(TokKind::Comma)) advance();
      }
      match(close);
      d.args = std::move(args);
    }
    // Validate against the built-in registry; unknown names are accepted as
    // user decorators but flagged if they look like a typo of a built-in.
    const auto& reg = DecoratorRegistry::instance();
    if (!reg.is_builtin(d.name)) {
      // user-defined decorator: allowed, no diagnostic in v1.0
    } else if (reg.lookup(d.name)->args_required && !d.args) {
      error("decorator @" + d.name + " requires arguments");
    }
    decs.push_back(std::move(d));
  }
  return decs;
}

std::vector<std::string> Parser::parse_param_list() {
  std::vector<std::string> params;
  match(TokKind::LParen);
  while (!check(TokKind::RParen) && !at_end()) {
    if (check(TokKind::Identifier)) params.push_back(advance().text);
    else advance();  // skip commas/newlines
  }
  match(TokKind::RParen);
  return params;
}

std::unique_ptr<LambdaFlow> Parser::parse_lambda_flow() {
  auto f = std::make_unique<LambdaFlow>();
  f->loc.begin = cur().loc;
  advance();                 // calc
  match(TokKind::ColonColon);
  advance();                 // lambda_flow
  if (check(TokKind::Identifier)) f->name = advance().text;
  f->params = parse_param_list();
  user_fn_arity_[f->name] = static_cast<int>(f->params.size());
  // lang_044: inline decorators between the param list and the body, e.g.
  //   calc::lambda_flow layer(x, w) @device(bob) { ... }
  // parse() prepends any leading decorators afterwards.
  {
    auto inline_decs = parse_decorators();
    for (auto& d : inline_decs) f->decorators.push_back(std::move(d));
  }
  f->body = parse_block();
  // return clause:  return name (, name)* ;
  skip_newlines();
  if (check(TokKind::Identifier) && cur().text == "return") {
    advance();
    // A return list may wrap across lines. Stopping unconditionally at the
    // first Newline silently DROPPED every name after it — a 19-output flow
    // written across three lines returned the first seven and reported success,
    // which is the worst possible failure for a list of results.
    //
    // A newline only continues the list when the line ended on a comma; if the
    // list ended on a name, the newline still terminates it. That keeps a
    // missing semicolon from swallowing whatever follows.
    bool expect_more = true;   // just after `return`, a name must come next
    while (!check(TokKind::Semicolon) && !at_end()) {
      if (check(TokKind::Newline)) {
        if (!expect_more) break;
        advance();
      } else if (check(TokKind::Identifier)) {
        f->returns.push_back(advance().text);
        expect_more = false;
      } else if (check(TokKind::Comma)) {
        advance();
        expect_more = true;
      } else {
        advance();
      }
    }
    match(TokKind::Semicolon);
  }
  return f;
}

std::unique_ptr<LambdaCalculator> Parser::parse_lambda_calculator() {
  auto c = std::make_unique<LambdaCalculator>();
  c->loc.begin = cur().loc;
  advance();                 // calc
  match(TokKind::ColonColon);
  advance();                 // lambda_calculator
  // ( channels = N )
  match(TokKind::LParen);
  while (!check(TokKind::RParen) && !at_end()) {
    if (check(TokKind::Identifier) && cur().text == "channels") {
      advance();
      match(TokKind::Assign);
      if (check(TokKind::Number)) c->channels = std::stoi(advance().text);
    } else {
      advance();
    }
  }
  match(TokKind::RParen);
  c->body = parse_block();
  return c;
}

std::unique_ptr<Block> Parser::parse_block() {
  auto block = std::make_unique<Block>();
  skip_newlines();
  if (!match(TokKind::LBrace)) { error("expected '{' to open block"); return block; }
  skip_newlines();
  while (!check(TokKind::RBrace) && !at_end()) {
    if (check(TokKind::Identifier) && cur().text == "@clock") {
      block->is_clocked = true;
      if (auto cs = parse_clock_section()) block->clocks.push_back(std::move(cs));
    } else {
      pending_decorators_ = parse_decorators();  // lang_006: per-statement decorators
      skip_newlines();
      if (check(TokKind::RBrace) || at_end()) { pending_decorators_.clear(); break; }
      if (check(TokKind::Identifier) && cur().text == "@clock") {
        pending_decorators_.clear();
        continue;
      }
      auto line = collect_statement_tokens();
      if (!line.empty()) {
        if (auto s = parse_statement(std::move(line))) block->statements.push_back(std::move(s));
      }
      pending_decorators_.clear();
    }
    skip_newlines();
  }
  match(TokKind::RBrace);
  return block;
}

std::unique_ptr<ClockSection> Parser::parse_clock_section() {
  auto cs = std::make_unique<ClockSection>();
  cs->loc.begin = cur().loc;
  advance();                 // @clock
  match(TokKind::LParen);
  if (check(TokKind::Number)) cs->level = std::stoi(advance().text);
  match(TokKind::RParen);
  match(TokKind::Colon);
  skip_newlines();
  // statements until next @clock or }
  uint32_t body_column = 0;
  while (!at_end() && !check(TokKind::RBrace) &&
         !(check(TokKind::Identifier) && cur().text == "@clock")) {
    if (body_column == 0) body_column = cur().loc.column;
    else if (cur().loc.column < body_column) break;
    auto line = collect_statement_tokens();
    if (!line.empty()) {
      if (auto s = parse_statement(std::move(line))) cs->statements.push_back(std::move(s));
    }
    skip_newlines();
  }
  return cs;
}

// Gather one logical statement's tokens up to a terminator at bracket depth 0.
// Consumes the Newline/Semicolon terminator; leaves a closing RBrace in place.
std::vector<Token> Parser::collect_statement_tokens() {
  std::vector<Token> line;
  int depth = 0;
  skip_newlines();
  while (!at_end()) {
    const Token& t = cur();
    if (depth == 0) {
      if (t.kind == TokKind::Newline) { advance(); break; }
      if (t.kind == TokKind::Semicolon) { advance(); break; }
      if (t.kind == TokKind::RBrace) break;  // end of block, do not consume
    }
    if (t.kind == TokKind::LBrace || t.kind == TokKind::LBracket || t.kind == TokKind::LParen) ++depth;
    if (t.kind == TokKind::RBrace || t.kind == TokKind::RBracket || t.kind == TokKind::RParen) --depth;
    if (t.kind == TokKind::Newline) { advance(); continue; }  // skip embedded newlines
    line.push_back(t);
    advance();
  }
  return line;
}

// --------------------------------------------------------------------------
// statement dispatch
// --------------------------------------------------------------------------
std::unique_ptr<Stmt> Parser::parse_statement(std::vector<Token> line) {
  if (line.empty()) return nullptr;

  // print:  return ( ... )
  if (line[0].kind == TokKind::Identifier && line[0].text == "return" &&
      line.size() >= 2 && line[1].kind == TokKind::LParen) {
    auto p = std::make_unique<PrintStatement>();
    p->loc.begin = line[0].loc;
    // inner tokens between the parens
    std::vector<Token> inner(line.begin() + 2, line.end() - (line.back().kind == TokKind::RParen ? 1 : 0));
    if (inner.size() == 1 && inner[0].kind == TokKind::String) {
      auto s = std::make_unique<StringLiteral>();
      s->value = inner[0].text;
      p->value = std::move(s);
    } else {
      p->value = reduce(inner);
    }
    return p;
  }

  // memory write:  <value> memory :: write
  if (line.size() >= 3 && line.back().text == "write" &&
      line[line.size() - 2].kind == TokKind::ColonColon &&
      line[line.size() - 3].text == "memory") {
    auto m = std::make_unique<MemoryWrite>();
    std::vector<Token> val(line.begin(), line.end() - 3);
    m->value = reduce(val);
    return m;
  }

  // assignment:  <value> <targets> [type] =
  if (line.back().kind == TokKind::Assign) {
    auto a = std::make_unique<Assignment>();
    a->loc.begin = line.front().loc;
    a->decorators = std::move(pending_decorators_);  // lang_006
    pending_decorators_.clear();
    std::size_t end = line.size() - 1;  // exclude '='

    // optional type annotation at the tail: IDENT(oc|ot|os) :: IDENT (:: IDENT)*
    // find a ColonColon whose left identifier is a type namespace.
    auto is_type_ns = [](const std::string& s) { return s == "oc" || s == "ot" || s == "os"; };
    for (std::size_t i = 0; i + 1 < end; ++i) {
      if (line[i].kind == TokKind::Identifier && is_type_ns(line[i].text) &&
          line[i + 1].kind == TokKind::ColonColon) {
        TypeAnnotation ta;
        ta.ns = line[i].text == "oc" ? TypeNamespace::Oc
              : line[i].text == "ot" ? TypeNamespace::Ot : TypeNamespace::Os;
        std::size_t j = i + 2;
        if (j < end && line[j].kind == TokKind::Identifier) ta.name = line[j++].text;
        while (j + 1 < end && line[j].kind == TokKind::ColonColon &&
               line[j + 1].kind == TokKind::Identifier) {
          ta.modifiers.push_back(line[j + 1].text);
          j += 2;
        }
        a->type = ta;
        end = i;  // value/targets are before the type annotation
        break;
      }
    }

    // trailing targets:  TARGET (COMMA TARGET)*  (read backward). A TARGET is a
    // plain identifier or a qualified name (ns :: member), e.g. receiver::x,
    // responser::out, memory::W. See [A-5].
    std::vector<std::string> targets;
    long k = static_cast<long>(end) - 1;
    auto read_target = [&](long& idx) -> std::optional<std::string> {
      // qualified: IDENT :: (IDENT|Number)
      if (idx - 2 >= 0 && (line[idx].kind == TokKind::Identifier || line[idx].kind == TokKind::Number) &&
          line[idx - 1].kind == TokKind::ColonColon &&
          line[idx - 2].kind == TokKind::Identifier) {
        std::string s = line[idx - 2].text + "::" + line[idx].text;
        if (reserved_words().count(s) || (line[idx].kind == TokKind::Identifier &&
                                          reserved_words().count(line[idx].text))) {
          errors_.push_back({"reserved word cannot be used as an assignment target",
                             line[idx].loc});
          return std::nullopt;
        }
        idx -= 3;
        return s;
      }
      if (idx >= 0 && line[idx].kind == TokKind::Identifier) {
        if (reserved_words().count(line[idx].text)) {
          errors_.push_back({"reserved word cannot be used as an assignment target",
                             line[idx].loc});
          return std::nullopt;
        }
        std::string s = line[idx].text;
        idx -= 1;
        return s;
      }
      return std::nullopt;
    };
    if (auto t0 = read_target(k)) {
      targets.push_back(*t0);
      while (k >= 0 && line[k].kind == TokKind::Comma) {
        long save = k;
        --k;
        if (auto tn = read_target(k)) { targets.push_back(*tn); }
        else { k = save; break; }
      }
    }
    std::reverse(targets.begin(), targets.end());
    a->targets = targets;

    std::vector<Token> value(line.begin(), line.begin() + (k + 1));
    a->value = reduce(value);
    return a;
  }

  // Unrecognized statement shape in v1.0: record but do not fail the parse.
  return nullptr;
}

// --------------------------------------------------------------------------
// RPN reduction
// --------------------------------------------------------------------------
ExprPtr Parser::reduce(const std::vector<Token>& v) {
  std::vector<ExprPtr> stack;
  std::unique_ptr<ObjectParams> pending;  // object params awaiting the next op

  auto pop = [&]() -> ExprPtr {
    if (stack.empty()) return nullptr;
    ExprPtr e = std::move(stack.back());
    stack.pop_back();
    return e;
  };
  auto pop_required = [&](const Token& tok, const std::string& op) -> ExprPtr {
    ExprPtr e = pop();
    if (!e) errors_.push_back({"stack underflow while reducing operator '" + op + "'", tok.loc});
    return e;
  };

  std::size_t i = 0;
  while (i < v.size()) {
    const Token& t = v[i];
    switch (t.kind) {
      case TokKind::Number: {
        auto n = std::make_unique<NumberLiteral>();
        n->lexeme = t.text; n->is_float = t.is_float; n->loc.begin = t.loc;
        stack.push_back(std::move(n));
        ++i; break;
      }
      case TokKind::String: {
        auto s = std::make_unique<StringLiteral>();
        s->value = t.text; s->loc.begin = t.loc;
        stack.push_back(std::move(s));
        ++i; break;
      }
      case TokKind::LBracket: {
        // tensor literal: collect a balanced [...] slice and parse nested numbers
        int depth = 0; std::size_t j = i;
        for (; j < v.size(); ++j) {
          if (v[j].kind == TokKind::LBracket) ++depth;
          else if (v[j].kind == TokKind::RBracket) { if (--depth == 0) { ++j; break; } }
        }
        std::function<ExprPtr(std::size_t&, std::size_t)> parse_tensor =
            [&](std::size_t& p, std::size_t stop) -> ExprPtr {
          auto tl = std::make_unique<TensorLiteral>();
          ++p;  // skip '['
          while (p < stop && v[p].kind != TokKind::RBracket) {
            if (v[p].kind == TokKind::LBracket) {
              tl->elements.push_back(parse_tensor(p, stop));
            } else if (v[p].kind == TokKind::Number) {
              auto n = std::make_unique<NumberLiteral>();
              n->lexeme = v[p].text; n->is_float = v[p].is_float;
              tl->elements.push_back(std::move(n));
              ++p;
            } else {
              ++p;  // skip commas
            }
          }
          if (p < stop && v[p].kind == TokKind::RBracket) ++p;
          return tl;
        };
        std::size_t p = i;
        ExprPtr tensor = parse_tensor(p, j);
        // lang_007: validate the matrix literal is rectangular; report ragged dims.
        std::string shape_err;
        if (!inferTensorLiteralShape(static_cast<TensorLiteral &>(*tensor),
                                     &shape_err)) {
          errors_.push_back({shape_err, t.loc});
        }
        stack.push_back(std::move(tensor));
        i = j; break;
      }
      case TokKind::LBrace: {
        // object parameters { key: value, ... }
        auto op = std::make_unique<ObjectParams>();
        op->loc.begin = t.loc;
        ++i;  // '{'
        while (i < v.size() && v[i].kind != TokKind::RBrace) {
          if (v[i].kind == TokKind::Identifier && i + 1 < v.size() &&
              v[i + 1].kind == TokKind::Colon) {
            ObjectField f;
            f.key = v[i].text;
            i += 2;
            // value: a single literal/identifier (v1.0)
            if (i < v.size() && v[i].kind == TokKind::Number) {
              auto n = std::make_unique<NumberLiteral>();
              n->lexeme = v[i].text; n->is_float = v[i].is_float; f.value = std::move(n);
            } else if (i < v.size() && v[i].kind == TokKind::Identifier) {
              auto vr = std::make_unique<VarRef>(); vr->name = v[i].text; f.value = std::move(vr);
            }
            op->fields.push_back(std::move(f));
            // advance to comma or closing brace
            while (i < v.size() && v[i].kind != TokKind::Comma && v[i].kind != TokKind::RBrace) ++i;
          }
          if (i < v.size() && v[i].kind == TokKind::Comma) ++i;
        }
        if (i < v.size() && v[i].kind == TokKind::RBrace) ++i;
        pending = std::move(op);
        break;
      }
      case TokKind::Compare: {
        auto o = std::make_unique<OpApplication>();
        o->op = t.text; o->op_class = OpClass::Comparison; o->loc.begin = t.loc;
        ExprPtr b = pop_required(t, t.text), a = pop_required(t, t.text);
        if (a) o->args.push_back(std::move(a));
        if (b) o->args.push_back(std::move(b));
        stack.push_back(std::move(o));
        ++i; break;
      }
      case TokKind::Identifier: {
        const std::string& name = t.text;
        // qualified ref:  ns :: member
        if (i + 2 < v.size() && v[i + 1].kind == TokKind::ColonColon &&
            ref_namespaces().count(name)) {
          auto q = std::make_unique<QualifiedRef>();
          q->ns = name; q->member = v[i + 2].text; q->loc.begin = t.loc;
          stack.push_back(std::move(q));
          i += 3; break;
        }
        // dotted reserved op: tensor.init / txt.read
        if (i + 2 < v.size() && v[i + 1].kind == TokKind::Dot &&
            v[i + 2].kind == TokKind::Identifier) {
          const std::string dotted = name + "." + v[i + 2].text;
          int ar = arity_of(dotted);
          if (ar >= 0) {
            auto o = std::make_unique<OpApplication>();
            o->op = dotted; o->op_class = classify(dotted); o->loc.begin = t.loc;
            std::vector<ExprPtr> args;
            for (int n = 0; n < ar; ++n) {
              if (auto e = pop_required(t, dotted)) args.push_back(std::move(e));
            }
            std::reverse(args.begin(), args.end());
            o->args = std::move(args);
            stack.push_back(std::move(o));
            i += 3; break;
          }
        }
        // property ref:  ident . ident (. ident)*
        if (i + 2 < v.size() && v[i + 1].kind == TokKind::Dot &&
            v[i + 2].kind == TokKind::Identifier) {
          auto base = std::make_unique<VarRef>(); base->name = name;
          auto pr = std::make_unique<PropertyRef>();
          pr->base = std::move(base); pr->loc.begin = t.loc;
          std::size_t j = i + 1;
          while (j + 1 < v.size() && v[j].kind == TokKind::Dot &&
                 v[j + 1].kind == TokKind::Identifier) {
            pr->path.push_back(v[j + 1].text);
            j += 2;
          }
          stack.push_back(std::move(pr));
          i = j; break;
        }
        // indexed ref:  ident [ ... ] ...
        if (i + 1 < v.size() && v[i + 1].kind == TokKind::LBracket) {
          auto base = std::make_unique<VarRef>(); base->name = name;
          auto ir = std::make_unique<IndexedRef>();
          ir->base = std::move(base); ir->loc.begin = t.loc;
          std::size_t j = i + 1;
          while (j < v.size() && v[j].kind == TokKind::LBracket) {
            Index idx;
            ++j;  // '['
            // [int] or [lo:hi]
            std::optional<int64_t> lo, hi; bool slice = false;
            if (j < v.size() && v[j].kind == TokKind::Number) lo = std::stoll(v[j++].text);
            if (j < v.size() && v[j].kind == TokKind::Colon) { slice = true; ++j;
              if (j < v.size() && v[j].kind == TokKind::Number) hi = std::stoll(v[j++].text);
            }
            idx.is_slice = slice; idx.first = lo; idx.second = hi;
            ir->indices.push_back(idx);
            if (j < v.size() && v[j].kind == TokKind::RBracket) ++j;
          }
          stack.push_back(std::move(ir));
          i = j; break;
        }
        // stack ops
        if (stack_ops().count(name)) {
          const std::size_t before = stack.size();
          if ((name == "dup" || name == "drop") && before < 1)
            errors_.push_back({"stack underflow while reducing stack op '" + name + "'", t.loc});
          else if ((name == "swap" || name == "over") && before < 2)
            errors_.push_back({"stack underflow while reducing stack op '" + name + "'", t.loc});
          else if ((name == "rot" || name == "-rot") && before < 3)
            errors_.push_back({"stack underflow while reducing stack op '" + name + "'", t.loc});
          else if (name == "dup") { stack.push_back(clone_expr(stack.back().get())); }
          else if (name == "drop") { pop(); }
          else if (name == "swap") { std::swap(stack[stack.size()-1], stack[stack.size()-2]); }
          else if (name == "over") { stack.push_back(clone_expr(stack[stack.size()-2].get())); }
          else if (name == "rot") { auto a=std::move(stack[stack.size()-3]); stack.erase(stack.end()-3); stack.push_back(std::move(a)); }
          else if (name == "-rot") { auto c=std::move(stack.back()); stack.pop_back(); stack.insert(stack.end()-2, std::move(c)); }
          ++i; break;
        }
        // operator (primitive or user function)
        int ar = arity_of(name);
        if (ar >= 0) {
          auto o = std::make_unique<OpApplication>();
          o->op = name; o->op_class = classify(name); o->loc.begin = t.loc;
          std::vector<ExprPtr> args;
          for (int n = 0; n < ar; ++n) {
            if (auto e = pop_required(t, name)) args.push_back(std::move(e));
          }
          std::reverse(args.begin(), args.end());
          o->args = std::move(args);
          if (pending) { o->args.push_back(std::move(pending)); pending.reset(); }
          stack.push_back(std::move(o));
          ++i; break;
        }
        if (reserved_words().count(name)) {
          errors_.push_back({"reserved word '" + name + "' is not implemented as an operator in this parser", t.loc});
          ++i; break;
        }
        // combinator -> operand (reduction semantics deferred to later ticket)
        if (combinators().count(name)) {
          auto c = std::make_unique<CombinatorRef>(); c->name = name; c->loc.begin = t.loc;
          stack.push_back(std::move(c));
          ++i; break;
        }
        // plain variable reference
        auto vr = std::make_unique<VarRef>(); vr->name = name; vr->loc.begin = t.loc;
        stack.push_back(std::move(vr));
        ++i; break;
      }
      default:
        ++i; break;  // skip stray punctuation/newlines
    }
  }

  if (stack.empty()) return nullptr;
  return std::move(stack.back());
}

}  // namespace caramel::parse
