// ============================================================================
// Caramel Language - Matrix/tensor literal shape inference
// ----------------------------------------------------------------------------
// Ticket:   lang_007 (Matrix Initialization Syntax)
// Version:  1.0.0
// ----------------------------------------------------------------------------
// Infers the Shape of a nested matrix literal ([[1,2,3],[4,5,6]] -> [2,3]) and
// validates that it is rectangular (no ragged rows / mixed scalar-and-nested
// levels). Used by the parser to diagnose dimension mismatches in constant
// matrices and by the type checker to type tensor literals.
// ============================================================================
#ifndef CARAMEL_PARSE_LITERAL_SHAPE_H
#define CARAMEL_PARSE_LITERAL_SHAPE_H

#include <optional>
#include <string>

#include "caramel/ast/ast.h"
#include "caramel/types/type.h"

namespace caramel::parse {

// Infer the shape of a tensor literal. On a ragged / malformed literal, returns
// nullopt and (if `error` is non-null) sets a human-readable message.
std::optional<caramel::types::Shape> inferTensorLiteralShape(
    const caramel::ast::TensorLiteral &lit, std::string *error = nullptr);

// True if `lit` is a well-formed rectangular literal.
bool isRectangular(const caramel::ast::TensorLiteral &lit);

}  // namespace caramel::parse

#endif  // CARAMEL_PARSE_LITERAL_SHAPE_H
