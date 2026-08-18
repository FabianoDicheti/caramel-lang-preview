#include "caramel/lint/linter.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace caramel::lint {
namespace {

using ast::NodeKind;
using ast::SourceLocation;

struct Symbol {
  SourceLocation loc{};
  bool used = false;
  bool parameter = false;
};

class Linter {
 public:
  std::vector<Diagnostic> run(const ast::Program& program) {
    collectGlobals(program);
    lintTopLevel(program);
    for (const auto& item : program.items) {
      switch (item->kind) {
        case NodeKind::LambdaFlow:
          lintFlow(static_cast<const ast::LambdaFlow&>(*item));
          break;
        case NodeKind::LambdaSymphony:
          lintSymphony(static_cast<const ast::LambdaSymphony&>(*item));
          break;
        case NodeKind::LambdaCalculator:
          lintCalculator(static_cast<const ast::LambdaCalculator&>(*item));
          break;
        default:
          break;
      }
    }
    std::stable_sort(diagnostics_.begin(), diagnostics_.end(),
                     [](const Diagnostic& a, const Diagnostic& b) {
                       if (a.loc.offset != b.loc.offset)
                         return a.loc.offset < b.loc.offset;
                       if (a.code != b.code) return a.code < b.code;
                       return a.message < b.message;
                     });
    return std::move(diagnostics_);
  }

 private:
  using Scope = std::unordered_map<std::string, Symbol>;

  void report(Severity severity, std::string code, std::string message,
              SourceLocation loc) {
    diagnostics_.push_back(
        {severity, std::move(code), std::move(message), loc});
  }

  void collectGlobals(const ast::Program& program) {
    for (const auto& item : program.items) {
      std::string name;
      if (item->kind == NodeKind::LambdaFlow)
        name = static_cast<const ast::LambdaFlow&>(*item).name;
      else if (item->kind == NodeKind::LambdaSymphony)
        name = static_cast<const ast::LambdaSymphony&>(*item).name;
      else
        continue;
      auto [it, inserted] = callables_.emplace(name, item->loc.begin);
      if (!inserted)
        report(Severity::Error, "E001",
               "duplicate function definition '" + name + "'", item->loc.begin);
    }
  }

  static void appendStatements(const ast::Block* block,
                               std::vector<const ast::Stmt*>& out) {
    if (!block) return;
    for (const auto& stmt : block->statements) out.push_back(stmt.get());
    for (const auto& clock : block->clocks)
      for (const auto& stmt : clock->statements) out.push_back(stmt.get());
  }

  void collectAssignment(const ast::Stmt& stmt, Scope& scope) {
    if (stmt.kind == NodeKind::Assignment) {
      const auto& assignment = static_cast<const ast::Assignment&>(stmt);
      std::unordered_set<std::string> within;
      for (const auto& target : assignment.targets) {
        if (!within.insert(target).second) {
          report(Severity::Error, "E002",
                 "duplicate assignment target '" + target + "'",
                 assignment.loc.begin);
          continue;
        }
        // receiver::, responser::, memory::, etc. are external bindings, not
        // lexical variables whose local usage can be measured.
        if (target.find("::") != std::string::npos) continue;
        auto [it, inserted] =
            scope.emplace(target, Symbol{assignment.loc.begin, false, false});
        if (!inserted) {
          report(Severity::Warning, "W002",
                 "variable '" + target + "' is assigned more than once",
                 assignment.loc.begin);
        }
      }
    } else if (stmt.kind == NodeKind::BackpropStatement) {
      const auto& backprop = static_cast<const ast::BackpropStatement&>(stmt);
      for (const auto& target : backprop.targets)
        scope.emplace(target, Symbol{backprop.loc.begin, false, false});
    } else if (stmt.kind == NodeKind::DeviceBlock) {
      const auto& device = static_cast<const ast::DeviceBlock&>(stmt);
      for (const auto& nested : device.body) collectAssignment(*nested, scope);
    }
  }

  void collectBlockAssignments(const ast::Block* block, Scope& scope) {
    std::vector<const ast::Stmt*> statements;
    appendStatements(block, statements);
    for (const auto* statement : statements) collectAssignment(*statement, scope);
  }

  void lintExpr(const ast::Expr* expr, Scope& scope,
                std::unordered_set<std::string>& lambdaBound) {
    if (!expr) return;
    switch (expr->kind) {
      case NodeKind::VarRef: {
        const auto& var = static_cast<const ast::VarRef&>(*expr);
        if (lambdaBound.count(var.name)) return;
        auto it = scope.find(var.name);
        if (it != scope.end()) {
          it->second.used = true;
        } else if (!callables_.count(var.name) &&
                   !globalInputs_.count(var.name) &&
                   !globalVariables_.count(var.name)) {
          report(Severity::Error, "E003",
                 "undefined variable '" + var.name + "'", var.loc.begin);
        }
        break;
      }
      case NodeKind::TensorLiteral: {
        const auto& tensor = static_cast<const ast::TensorLiteral&>(*expr);
        for (const auto& element : tensor.elements)
          lintExpr(element.get(), scope, lambdaBound);
        break;
      }
      case NodeKind::IndexedRef:
        lintExpr(static_cast<const ast::IndexedRef&>(*expr).base.get(), scope,
                 lambdaBound);
        break;
      case NodeKind::PropertyRef:
        lintExpr(static_cast<const ast::PropertyRef&>(*expr).base.get(), scope,
                 lambdaBound);
        break;
      case NodeKind::ObjectParams: {
        const auto& object = static_cast<const ast::ObjectParams&>(*expr);
        std::unordered_set<std::string> keys;
        for (const auto& field : object.fields) {
          if (!keys.insert(field.key).second)
            report(Severity::Warning, "W003",
                   "duplicate object parameter '" + field.key + "'",
                   object.loc.begin);
          // A bare identifier in an object value can be an enum-like option
          // (for example activation: relu), not necessarily a variable.
          if (field.value && field.value->kind != NodeKind::VarRef)
            lintExpr(field.value.get(), scope, lambdaBound);
        }
        break;
      }
      case NodeKind::LambdaExpr: {
        const auto& lambda = static_cast<const ast::LambdaExpr&>(*expr);
        auto nestedBound = lambdaBound;
        nestedBound.insert(lambda.params.begin(), lambda.params.end());
        lintExpr(lambda.body.get(), scope, nestedBound);
        break;
      }
      case NodeKind::OpApplication: {
        const auto& operation = static_cast<const ast::OpApplication&>(*expr);
        for (const auto& argument : operation.args)
          lintExpr(argument.get(), scope, lambdaBound);
        break;
      }
      default:
        break;
    }
  }

  void lintStatement(const ast::Stmt& stmt, Scope& scope) {
    std::unordered_set<std::string> bound;
    switch (stmt.kind) {
      case NodeKind::Assignment:
        lintExpr(static_cast<const ast::Assignment&>(stmt).value.get(), scope,
                 bound);
        break;
      case NodeKind::PrintStatement:
        lintExpr(static_cast<const ast::PrintStatement&>(stmt).value.get(), scope,
                 bound);
        break;
      case NodeKind::MemoryWrite:
        lintExpr(static_cast<const ast::MemoryWrite&>(stmt).value.get(), scope,
                 bound);
        break;
      case NodeKind::BackpropStatement: {
        const auto& backprop =
            static_cast<const ast::BackpropStatement&>(stmt);
        lintExpr(backprop.loss.get(), scope, bound);
        lintExpr(backprop.config.get(), scope, bound);
        break;
      }
      case NodeKind::DeviceBlock: {
        const auto& device = static_cast<const ast::DeviceBlock&>(stmt);
        lintDeviceFields(device);
        for (const auto& nested : device.body) lintStatement(*nested, scope);
        break;
      }
      default:
        break;
    }
  }

  void lintBlock(const ast::Block* block, Scope& scope) {
    std::vector<const ast::Stmt*> statements;
    appendStatements(block, statements);
    for (const auto* statement : statements) lintStatement(*statement, scope);
  }

  void reportUnused(const Scope& scope) {
    for (const auto& [name, symbol] : scope) {
      if (!symbol.used)
        report(Severity::Warning, symbol.parameter ? "W001" : "W004",
               std::string(symbol.parameter ? "unused parameter '" :
                                              "unused variable '") +
                   name + "'",
               symbol.loc);
    }
  }

  void lintFunction(const std::vector<std::string>& params,
                    const ast::Block* body,
                    const std::vector<std::string>& returns,
                    SourceLocation loc) {
    Scope scope;
    for (const auto& param : params) {
      auto [it, inserted] =
          scope.emplace(param, Symbol{loc, false, true});
      if (!inserted)
        report(Severity::Error, "E004",
               "duplicate parameter '" + param + "'", loc);
    }
    collectBlockAssignments(body, scope);
    lintBlock(body, scope);
    for (const auto& name : returns) {
      auto it = scope.find(name);
      if (it == scope.end()) {
        report(Severity::Error, "E005",
               "return references undefined variable '" + name + "'", loc);
      } else {
        it->second.used = true;
      }
    }
    reportUnused(scope);
  }

  void lintFlow(const ast::LambdaFlow& flow) {
    lintFunction(flow.params, flow.body.get(), flow.returns, flow.loc.begin);
  }

  void lintSymphony(const ast::LambdaSymphony& symphony) {
    lintFunction(symphony.params, symphony.body.get(), symphony.returns,
                 symphony.loc.begin);
  }

  void lintCalculator(const ast::LambdaCalculator& calculator) {
    Scope scope;
    collectBlockAssignments(calculator.body.get(), scope);
    lintBlock(calculator.body.get(), scope);
    reportUnused(scope);
  }

  void lintDeviceFields(const ast::DeviceBlock& device) {
    std::unordered_set<std::string> keys;
    for (const auto& field : device.fields) {
      if (!keys.insert(field.key).second)
        report(Severity::Warning, "W005",
               "duplicate device field '" + field.key + "'", field.loc.begin);
    }
  }

  void lintTopLevel(const ast::Program& program) {
    Scope scope;
    for (const auto& item : program.items) {
      if (item->kind == NodeKind::Directive) {
        const auto& directive = static_cast<const ast::Directive&>(*item);
        if (directive.dkind == ast::DirectiveKind::Input) {
          if (!globalInputs_.insert(directive.key).second)
            report(Severity::Warning, "W006",
                   "input '" + directive.key + "' is defined more than once",
                   directive.loc.begin);
        }
      } else if (auto* statement = dynamic_cast<const ast::Stmt*>(item.get())) {
        collectAssignment(*statement, scope);
      }
    }
    for (const auto& [name, symbol] : scope) {
      (void)symbol;
      globalVariables_.insert(name);
    }
    for (const auto& item : program.items) {
      if (auto* statement = dynamic_cast<const ast::Stmt*>(item.get()))
        lintStatement(*statement, scope);
      if (item->kind == NodeKind::Directive) {
        const auto& directive = static_cast<const ast::Directive&>(*item);
        if (directive.object) {
          std::unordered_set<std::string> bound;
          lintExpr(directive.object.get(), scope, bound);
        }
      }
    }
    reportUnused(scope);
  }

  std::vector<Diagnostic> diagnostics_;
  std::unordered_map<std::string, SourceLocation> callables_;
  std::unordered_set<std::string> globalInputs_;
  std::unordered_set<std::string> globalVariables_;
};

}  // namespace

std::vector<Diagnostic> lint(const ast::Program& program) {
  return Linter().run(program);
}

const char* severityName(Severity severity) {
  return severity == Severity::Error ? "error" : "warning";
}

}  // namespace caramel::lint
