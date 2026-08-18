// ============================================================================
// Caramel Language - Decorator registry
// ----------------------------------------------------------------------------
// Ticket:  lang_006 (Decorator System)
// Version: 1.0.0
// ============================================================================
#include "caramel/parse/decorator.h"

#include <algorithm>

namespace caramel::parse {

DecoratorRegistry::DecoratorRegistry() {
  auto add = [&](std::string name, DecoratorTarget target, DecoratorKind kind,
                 std::vector<std::string> params, bool args_required) {
    table_.push_back({std::move(name), target, kind, std::move(params), args_required});
  };

  // @clock(n): parallel scheduling phase (handled structurally as ClockSection,
  // registered here so validation recognizes it).
  add("clock", DecoratorTarget::Block, DecoratorKind::Scheduling, {}, true);

  // @quantize{quantmax, quantmin, quantres}: per-tensor quantization override.
  add("quantize", DecoratorTarget::Statement, DecoratorKind::Quantization,
      {"quantmax", "quantmin", "quantres", "bits"}, true);

  // @tile{tile_m, tile_n, tile_k}: tiling hint for matmul/conv lowering.
  add("tile", DecoratorTarget::Statement, DecoratorKind::Tiling,
      {"tile_m", "tile_n", "tile_k"}, true);

  // @device{target}: force hardware routing (fpga/cpu) for the statement/flow.
  add("device", DecoratorTarget::Any, DecoratorKind::Routing, {"target"}, true);

  // @autodiff: enable automatic differentiation for the decorated flow.
  add("autodiff", DecoratorTarget::Definition, DecoratorKind::Autodiff, {}, false);

  // @prefetch{size}: DMA prefetch hint for the decorated statement.
  add("prefetch", DecoratorTarget::Statement, DecoratorKind::Memory, {"size"}, false);
}

const DecoratorRegistry& DecoratorRegistry::instance() {
  static const DecoratorRegistry registry;
  return registry;
}

const DecoratorSchema* DecoratorRegistry::lookup(const std::string& name) const {
  auto it = std::find_if(table_.begin(), table_.end(),
                         [&](const DecoratorSchema& s) { return s.name == name; });
  return it == table_.end() ? nullptr : &*it;
}

}  // namespace caramel::parse
