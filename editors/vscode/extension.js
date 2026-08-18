// Caramel VS Code extension.
//
// Everything here is a thin wrapper over the two CLIs that ship with the
// language: caramel-lint provides diagnostics, caramel-run executes a file.
// No language server, no daemon - if the CLIs are on PATH the editor works.
'use strict';

const vscode = require('vscode');
const cp = require('child_process');
const os = require('os');
const fs = require('fs');
const path = require('path');

const LANG = 'caramel';

// caramel-lint prints:  <path>:<line>:<col>: <severity>[<CODE>]: <message>
const DIAGNOSTIC_RE = /^(.*?):(\d+):(\d+):\s*(error|warning|note)(?:\[([^\]]+)\])?:\s*(.*)$/;

let diagnostics;
let output;
let lintTimer;

function config() {
  return vscode.workspace.getConfiguration('caramel');
}

function log(message) {
  output.appendLine(message);
}

// VS Code expands ${workspaceFolder} in launch/task configs but not in plain
// settings, so a path like "${workspaceFolder}/build/caramel-run" - the natural
// way to point at a local build - has to be expanded here.
function resolvePath(value) {
  if (!value) return value;
  let resolved = value;
  const folders = vscode.workspace.workspaceFolders;
  if (folders && folders.length) {
    resolved = resolved.replace(/\$\{workspaceFolder\}/g, folders[0].uri.fsPath);
  }
  if (resolved.startsWith('~/')) resolved = path.join(os.homedir(), resolved.slice(2));
  return resolved;
}

// --- Diagnostics ------------------------------------------------------------

// caramel-lint reads a path, so unsaved buffers are linted through a temp copy.
function lintDocument(document) {
  if (document.languageId !== LANG) return;
  if (!config().get('lint.enable', true)) {
    diagnostics.delete(document.uri);
    return;
  }

  const lintPath = resolvePath(config().get('lintPath', 'caramel-lint'));
  let target = document.uri.fsPath;
  let temp = null;
  if (document.isDirty || document.isUntitled) {
    temp = path.join(os.tmpdir(), `caramel-lint-${process.pid}-${Date.now()}.crml`);
    try {
      fs.writeFileSync(temp, document.getText());
      target = temp;
    } catch (err) {
      temp = null;
    }
  }

  cp.execFile(lintPath, [target], { timeout: 10000 }, (error, stdout, stderr) => {
    if (temp) { try { fs.unlinkSync(temp); } catch (e) { /* best effort */ } }

    // ENOENT means the toolchain is not installed; say so once instead of
    // spamming a diagnostic per keystroke.
    if (error && error.code === 'ENOENT') {
      reportMissingTool(lintPath);
      return;
    }

    const found = [];
    for (const line of `${stderr || ''}\n${stdout || ''}`.split(/\r?\n/)) {
      const m = DIAGNOSTIC_RE.exec(line.trim());
      if (!m) continue;
      const [, , lineNo, colNo, severity, code, message] = m;
      // caramel-lint is 1-based; VS Code is 0-based.
      const row = Math.max(0, parseInt(lineNo, 10) - 1);
      const col = Math.max(0, parseInt(colNo, 10) - 1);
      const wordRange = document.getWordRangeAtPosition(new vscode.Position(row, col));
      const range = wordRange || new vscode.Range(row, col, row, col + 1);

      const diagnostic = new vscode.Diagnostic(
        range,
        message,
        severity === 'error' ? vscode.DiagnosticSeverity.Error
          : severity === 'warning' ? vscode.DiagnosticSeverity.Warning
            : vscode.DiagnosticSeverity.Information);
      diagnostic.source = 'caramel-lint';
      if (code) diagnostic.code = code;
      found.push(diagnostic);
    }
    diagnostics.set(document.uri, found);
  });
}

let warnedMissing = false;
function reportMissingTool(tool) {
  if (warnedMissing) return;
  warnedMissing = true;
  const install = 'Installation instructions';
  vscode.window.showWarningMessage(
    `Caramel: '${tool}' was not found on your PATH. Install the Caramel toolchain, or set caramel.lintPath / caramel.runPath.`,
    install, 'Open settings').then((choice) => {
      if (choice === install) {
        vscode.env.openExternal(vscode.Uri.parse(
          'https://github.com/FabianoDicheti/caramel-lang-preview/blob/master/INSTALL.md'));
      } else if (choice === 'Open settings') {
        vscode.commands.executeCommand('workbench.action.openSettings', 'caramel');
      }
    });
}

function scheduleLint(document) {
  clearTimeout(lintTimer);
  lintTimer = setTimeout(() => lintDocument(document), 300);
}

// --- Run --------------------------------------------------------------------

function runCurrentFile() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== LANG) {
    vscode.window.showInformationMessage('Caramel: open a .crml file first.');
    return;
  }

  editor.document.save().then(() => {
    const runPath = resolvePath(config().get('runPath', 'caramel-run'));
    const flow = config().get('run.flow', '');
    const inputs = config().get('run.inputs', []);

    const args = [editor.document.uri.fsPath];
    if (flow) args.push('--flow', flow);
    for (const input of inputs) args.push('--in', input);

    output.show(true);
    log(`\n$ ${runPath} ${args.join(' ')}`);
    cp.execFile(runPath, args, { timeout: 60000 }, (error, stdout, stderr) => {
      if (error && error.code === 'ENOENT') {
        reportMissingTool(runPath);
        return;
      }
      if (stdout) log(stdout.trimEnd());
      if (stderr) log(stderr.trimEnd());
      if (error && error.code) log(`[exit ${error.code}]`);
    });
  });
}

// --- Completions and hovers -------------------------------------------------
//
// Deliberately data-driven and small: the same names the TextMate grammar
// highlights, with one line of documentation each so hovering explains an op.

const OPS = {
  matmul: 'Matrix product. RPN: `a b matmul out =`',
  elemwise_add: 'Element-wise addition of two tensors of the same shape.',
  elemwise_sub: 'Element-wise subtraction.',
  elemwise_mul: 'Element-wise (Hadamard) product.',
  elemwise_div: 'Element-wise division. Division by zero yields 0.',
  scalar_mul: 'Multiply every element by a scalar.',
  add: 'Scalar/tensor addition. `a b add out =`',
  sub: 'Scalar/tensor subtraction.',
  mul: 'Scalar/tensor multiplication.',
  div: 'Scalar/tensor division (integer).',
  relu: 'Rectified linear unit: max(0, x), element-wise.',
  sigmoid: 'Logistic activation.',
  tanh: 'Hyperbolic tangent activation.',
  softmax: 'Softmax over the last axis.',
  gelu: 'Gaussian error linear unit.',
  silu: 'Sigmoid linear unit (swish).',
  transpose: 'Transpose a matrix.',
  reshape: 'Reshape a tensor.',
  concat: 'Concatenate tensors.',
  split: 'Split a tensor.',
  tensor_sum: 'Sum of all elements.',
  tensor_mean: 'Mean of all elements (integer division).',
  tensor_max: 'Largest element.',
  tensor_min: 'Smallest element.',
  argmax: 'Index of the largest element.',
  conv1d: '1-D convolution.',
  conv2d: '2-D convolution.',
  conv3d: '3-D convolution.',
  maxpool2d: '2-D max pooling.',
  avgpool2d: '2-D average pooling.',
  batchnorm: 'Batch normalization.',
  layernorm: 'Layer normalization.',
  determinant: 'Matrix determinant.',
  inverse: 'Matrix inverse.',
  eigendecomp: 'Eigendecomposition.',
  svd: 'Singular value decomposition.',
  qr: 'QR decomposition.',
  cholesky: 'Cholesky decomposition.',
  lu: 'LU decomposition.'
};

const GENERATORS = {
  zeros: '`R C zeros` - R by C matrix of zeros.',
  ones: '`R C ones` - R by C matrix of ones.',
  full: '`R C V full` - R by C matrix filled with V.',
  eye: '`N eye` - N by N identity matrix.',
  diag: '`[a,b,c] diag` - diagonal matrix from a vector.',
  range: '`N range` - vector [0, 1, ... N-1].',
  band: '`N LOWER UPPER FILL band` - banded matrix.',
  tridiag: '`N SUB DIAG SUPER tridiag` - tridiagonal matrix.',
  from_spectrum: '`[l1,l2,...] from_spectrum` - matrix with the given spectrum.',
  random: '`R C random` - reproducible pseudo-random matrix (fixed seed).'
};

const DIRECTIVES = {
  'crml::quantmax': 'Upper bound of the quantization range. Required preamble.',
  'crml::quantmin': 'Lower bound of the quantization range. Required preamble.',
  'crml::quantres': 'Quantization resolution/step. Required preamble.',
  'crml::quantrange': 'Combined quantization range directive.',
  'crml::decimal_places': 'Decimal places retained when quantizing.',
  'crml::enable_autodiff': 'Enable autodiff metadata for the program.',
  'crml::checkpoint_every': 'Gradient checkpoint interval.',
  'calc::lambda_flow': 'Named flow: parameters, RPN body, and a return clause.',
  'calc::lambda_symphony': 'Higher-level grouping of computations.',
  'calc::lambda_calculator': 'Calculator body with an optional channel count.',
  'in::': 'Script-embedded input value: `in::name = <literal>;`',
  'profile::': 'Print the algebraic profile of an input matrix: `profile::name;`',
  'status::': "Print a declared worker's status metadata: `status::alias;`",
  'memory::write': 'Write a value to device memory.'
};

const COMBINATORS = [
  'identity', 'kestrel', 'kite', 'mockingbird', 'bluebird', 'cardinal',
  'starling', 'thrush', 'warbler', 'owl', 'compose', 'blackbird', 'psi',
  'phoenix', 'vireo', 'ycombinator'
];

const DECORATORS = {
  clock: '`@clock(n):` - mark a computation phase.',
  tile: '`@tile {rows: R, cols: C}` - tiling hint.',
  device: '`@device {...}` - target a device.',
  autodiff: '`@autodiff` - request autodiff metadata.',
  quantize: '`@quantize {...}` - per-statement quantization.',
  checkpoint: '`@checkpoint` - gradient checkpoint marker.'
};

function completionItems() {
  const items = [];
  const add = (label, kind, doc, detail) => {
    const item = new vscode.CompletionItem(label, kind);
    if (doc) item.documentation = new vscode.MarkdownString(doc);
    if (detail) item.detail = detail;
    items.push(item);
  };

  for (const [name, doc] of Object.entries(OPS)) {
    add(name, vscode.CompletionItemKind.Function, doc, 'Caramel op');
  }
  for (const [name, doc] of Object.entries(GENERATORS)) {
    add(name, vscode.CompletionItemKind.Constructor, doc, 'tensor generator');
  }
  for (const [name, doc] of Object.entries(DIRECTIVES)) {
    add(name, vscode.CompletionItemKind.Keyword, doc, 'directive');
  }
  for (const name of COMBINATORS) {
    add(name, vscode.CompletionItemKind.Function, 'Lambda-calculus combinator.', 'combinator');
  }
  for (const [name, doc] of Object.entries(DECORATORS)) {
    add('@' + name, vscode.CompletionItemKind.Property, doc, 'decorator');
  }
  for (const name of ['dup', 'swap', 'drop', 'over', 'rot', '-rot']) {
    add(name, vscode.CompletionItemKind.Operator, 'Stack operation.', 'stack op');
  }
  return items;
}

function hoverFor(word) {
  if (OPS[word]) return `**${word}** — ${OPS[word]}`;
  if (GENERATORS[word]) return `**${word}** — ${GENERATORS[word]}`;
  if (DECORATORS[word]) return `**@${word}** — ${DECORATORS[word]}`;
  if (COMBINATORS.includes(word)) return `**${word}** — lambda-calculus combinator.`;
  for (const [key, doc] of Object.entries(DIRECTIVES)) {
    if (key.endsWith('::') ? key.startsWith(word) : key.split('::')[1] === word) {
      return `**${key}** — ${doc}`;
    }
  }
  return null;
}

// --- Activation -------------------------------------------------------------

function activate(context) {
  diagnostics = vscode.languages.createDiagnosticCollection('caramel');
  output = vscode.window.createOutputChannel('Caramel');
  context.subscriptions.push(diagnostics, output);

  context.subscriptions.push(
    vscode.commands.registerCommand('caramel.runFile', runCurrentFile),
    vscode.commands.registerCommand('caramel.showOutput', () => output.show()),
    vscode.commands.registerCommand('caramel.lintFile', () => {
      const editor = vscode.window.activeTextEditor;
      if (editor) lintDocument(editor.document);
    }));

  context.subscriptions.push(
    vscode.languages.registerCompletionItemProvider(LANG, {
      provideCompletionItems: () => completionItems()
    }, ':', '@'),
    vscode.languages.registerHoverProvider(LANG, {
      provideHover(document, position) {
        const range = document.getWordRangeAtPosition(position, /[A-Za-z_][A-Za-z0-9_:]*/);
        if (!range) return null;
        const text = hoverFor(document.getText(range));
        return text ? new vscode.Hover(new vscode.MarkdownString(text)) : null;
      }
    }));

  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument(lintDocument),
    vscode.workspace.onDidSaveTextDocument(lintDocument),
    vscode.workspace.onDidCloseTextDocument((d) => diagnostics.delete(d.uri)),
    vscode.workspace.onDidChangeTextDocument((event) => {
      if (config().get('lint.run', 'onType') === 'onType') scheduleLint(event.document);
    }));

  vscode.workspace.textDocuments.forEach(lintDocument);
}

function deactivate() {
  clearTimeout(lintTimer);
}

module.exports = { activate, deactivate };
