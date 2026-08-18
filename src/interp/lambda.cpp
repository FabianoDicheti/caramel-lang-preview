// ============================================================================
// Caramel Language - CPU lambda-calculus evaluator implementation
// ----------------------------------------------------------------------------
// Ticket:  lang_017 (CPU-Based Lambda Evaluator)
// Version: 1.0.0
// ============================================================================
#include "caramel/interp/lambda.h"

namespace caramel::interp {

// ---- smart constructors ----------------------------------------------------
static std::shared_ptr<Term> mk(TKind k) {
  auto t = std::make_shared<Term>();
  t->kind = k;
  return t;
}
TermPtr var(std::string n) { auto t = mk(TKind::Var); t->name = std::move(n); return t; }
TermPtr lit(int64_t v) { auto t = mk(TKind::Lit); t->lit = v; return t; }
TermPtr abs(std::string p, TermPtr b) {
  auto t = mk(TKind::Abs);
  t->param = std::move(p); t->body = std::move(b); return t;
}
TermPtr app(TermPtr f, TermPtr a) {
  auto t = mk(TKind::App);
  t->fn = std::move(f); t->arg = std::move(a); return t;
}
TermPtr prim(std::string n) { auto t = mk(TKind::Prim); t->name = std::move(n); return t; }

TermPtr apply(TermPtr fn, std::vector<TermPtr> args) {
  TermPtr acc = std::move(fn);
  for (auto &a : args) acc = app(acc, a);
  return acc;
}

TermPtr combinator(const std::string &name) {
  if (name == "identity") return abs("x", var("x"));
  if (name == "kestrel") return abs("x", abs("y", var("x")));   // K (true)
  if (name == "kite") return abs("x", abs("y", var("y")));      // KI (false)
  if (name == "compose")  // B: \f g x. f (g x)
    return abs("f", abs("g", abs("x", app(var("f"), app(var("g"), var("x"))))));
  return var(name);  // unknown -> a free variable
}

// ---- runtime values --------------------------------------------------------
namespace {

struct RT;
using RTPtr = std::shared_ptr<RT>;
using Env = std::unordered_map<std::string, RTPtr>;

struct RT {
  enum K { Int, Closure, Partial } k = Int;
  int64_t i = 0;
  // closure
  std::string param;
  TermPtr body;
  Env env;
  // partial primitive application
  std::string prim;
  std::vector<int64_t> args;
};

RTPtr rtInt(int64_t v) { auto r = std::make_shared<RT>(); r->k = RT::Int; r->i = v; return r; }

int primArity(const std::string &p) { return p == "not" ? 1 : 2; }

std::optional<int64_t> applyPrim(const std::string &p, const std::vector<int64_t> &a,
                                 std::string &err) {
  auto B = [](bool b) { return int64_t(b ? 1 : 0); };
  if (p == "not") return B(a[0] == 0);
  const int64_t x = a[0], y = a[1];
  if (p == "add") return x + y;
  if (p == "sub") return x - y;
  if (p == "mul") return x * y;
  if (p == "div") return y != 0 ? x / y : 0;
  if (p == "gt" || p == ">") return B(x > y);
  if (p == "lt" || p == "<") return B(x < y);
  if (p == "eq" || p == "==") return B(x == y);
  if (p == "ge" || p == ">=") return B(x >= y);
  if (p == "le" || p == "<=") return B(x <= y);
  if (p == "ne" || p == "!=") return B(x != y);
  if (p == "and") return B(x != 0 && y != 0);
  if (p == "or") return B(x != 0 || y != 0);
  if (p == "xor") return B((x != 0) != (y != 0));
  err = "unknown primitive '" + p + "'";
  return std::nullopt;
}

RTPtr evalRT(const TermPtr &t, const Env &env, std::string &err);

RTPtr applyRT(const RTPtr &f, const RTPtr &a, std::string &err) {
  if (!f || !err.empty()) return nullptr;
  if (f->k == RT::Closure) {
    Env e = f->env;
    e[f->param] = a;
    return evalRT(f->body, e, err);
  }
  if (f->k == RT::Partial) {
    if (a->k != RT::Int) { err = "primitive applied to a non-integer"; return nullptr; }
    auto r = std::make_shared<RT>(*f);
    r->args.push_back(a->i);
    if (static_cast<int>(r->args.size()) == primArity(r->prim)) {
      auto v = applyPrim(r->prim, r->args, err);
      if (!v) return nullptr;
      return rtInt(*v);
    }
    return r;  // still partially applied
  }
  err = "applied a non-function";
  return nullptr;
}

RTPtr evalRT(const TermPtr &t, const Env &env, std::string &err) {
  if (!t) { err = "null term"; return nullptr; }
  switch (t->kind) {
    case TKind::Lit: return rtInt(t->lit);
    case TKind::Var: {
      auto it = env.find(t->name);
      if (it == env.end()) { err = "unbound variable '" + t->name + "'"; return nullptr; }
      return it->second;
    }
    case TKind::Abs: {
      auto r = std::make_shared<RT>();
      r->k = RT::Closure; r->param = t->param; r->body = t->body; r->env = env;
      return r;
    }
    case TKind::Prim: {
      auto r = std::make_shared<RT>();
      r->k = RT::Partial; r->prim = t->name;
      return r;
    }
    case TKind::App: {
      auto f = evalRT(t->fn, env, err);
      if (!err.empty()) return nullptr;
      auto a = evalRT(t->arg, env, err);
      if (!err.empty()) return nullptr;
      return applyRT(f, a, err);
    }
  }
  err = "bad term"; return nullptr;
}

}  // namespace

EvalResult eval(const TermPtr &term,
                const std::unordered_map<std::string, int64_t> &env) {
  Env e;
  for (auto &kv : env) e[kv.first] = rtInt(kv.second);
  std::string err;
  RTPtr r = evalRT(term, e, err);
  EvalResult out;
  if (!err.empty()) { out.error = err; return out; }
  if (!r || r->k != RT::Int) { out.error = "program did not reduce to an integer"; return out; }
  out.value = r->i;
  return out;
}

// ---- comparison / logic op evaluator for the dataflow Interpreter ----------
OpEvaluator lambdaOpEvaluator() {
  return [](const std::string &op, const std::vector<Value> &in,
            std::string *error) -> std::optional<Value> {
    auto bin = [&](int64_t (*fn)(int64_t, int64_t)) -> std::optional<Value> {
      if (in.size() != 2) { if (error) *error = op + " expects 2 operands"; return std::nullopt; }
      const Value &a = in[0], &b = in[1];
      if (a.is_scalar() && b.is_scalar()) return Value::scalar(fn(a.data[0], b.data[0]));
      if (a.same_shape(b)) {
        Value r; r.dims = a.dims; r.data.resize(a.data.size());
        for (size_t i = 0; i < a.data.size(); ++i) r.data[i] = fn(a.data[i], b.data[i]);
        return r;
      }
      if (a.is_scalar() || b.is_scalar()) {
        const Value &t = a.is_scalar() ? b : a;
        const int64_t s = a.is_scalar() ? a.data[0] : b.data[0];
        Value r; r.dims = t.dims; r.data.resize(t.data.size());
        for (size_t i = 0; i < t.data.size(); ++i)
          r.data[i] = a.is_scalar() ? fn(s, t.data[i]) : fn(t.data[i], s);
        return r;
      }
      if (error) *error = op + " operands have incompatible shapes";
      return std::nullopt;
    };
    if (op == ">") return bin([](int64_t a, int64_t b){ return int64_t(a > b); });
    if (op == "<") return bin([](int64_t a, int64_t b){ return int64_t(a < b); });
    if (op == "==") return bin([](int64_t a, int64_t b){ return int64_t(a == b); });
    if (op == ">=") return bin([](int64_t a, int64_t b){ return int64_t(a >= b); });
    if (op == "<=") return bin([](int64_t a, int64_t b){ return int64_t(a <= b); });
    if (op == "!=") return bin([](int64_t a, int64_t b){ return int64_t(a != b); });
    if (op == "and") return bin([](int64_t a, int64_t b){ return int64_t(a != 0 && b != 0); });
    if (op == "or")  return bin([](int64_t a, int64_t b){ return int64_t(a != 0 || b != 0); });
    if (op == "xor") return bin([](int64_t a, int64_t b){ return int64_t((a != 0) != (b != 0)); });
    if (op == "not") {
      if (in.size() != 1) { if (error) *error = "not expects 1 operand"; return std::nullopt; }
      Value r = in[0];
      for (auto &e : r.data) e = (e == 0) ? 1 : 0;
      return r;
    }
    return std::nullopt;  // not a comparison/logic op
  };
}

}  // namespace caramel::interp
