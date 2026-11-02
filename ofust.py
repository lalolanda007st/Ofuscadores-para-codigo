#!/usr/bin/env python3
"""pyobf — ofuscador de código Python basado en AST.

Técnicas:
  * Renombrado de identificadores (funciones, clases, argumentos y variables
    locales) a nombres sin significado, respetando builtins, imports y la API
    pública declarada en __all__.
  * Ofuscación de cadenas de texto: cada literal str se codifica en base64 y se
    decodifica en tiempo de ejecución mediante un pequeño helper inyectado.

Uso:
    python pyobf.py entrada.py -o salida.py
    python pyobf.py entrada.py --rename-globals --keep main,API_KEY
"""
from __future__ import annotations

import argparse
import ast
import base64
import builtins
import keyword
import sys

_BUILTINS = set(dir(builtins)) | set(keyword.kwlist)
# Nombres que nunca se renombran por razones de corrección.
_ALWAYS_KEEP = {"self", "cls", "__all__"}


def _make_name_factory():
    """Genera nombres ofuscados únicos y visualmente confusos (l/I/1/O/0)."""
    alphabet = "lI1O0"
    counter = 0

    def next_name() -> str:
        nonlocal counter
        counter += 1
        n = counter
        out = []
        while True:
            n, r = divmod(n, len(alphabet))
            out.append(alphabet[r])
            if n == 0:
                break
        return "_" + "".join(reversed(out))

    return next_name


class _Collector(ast.NodeVisitor):
    """Recoge nombres candidatos a renombrar y nombres a preservar."""

    def __init__(self, rename_globals: bool):
        self.rename_globals = rename_globals
        self.defined: set[str] = set()      # candidatos a renombrar
        self.preserve: set[str] = set()     # imports, __all__, globals, etc.
        self._depth = 0                     # 0 == ámbito de módulo

    # --- imports: sus nombres visibles nunca se tocan ---
    def visit_Import(self, node: ast.Import):
        for a in node.names:
            self.preserve.add((a.asname or a.name).split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for a in node.names:
            self.preserve.add(a.asname or a.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        # Un 'global x' en una función referencia un nombre de módulo.
        for n in node.names:
            if not self.rename_globals:
                self.preserve.add(n)
        self.generic_visit(node)

    def _consider(self, name: str, is_module_level: bool):
        if name in _BUILTINS or name in _ALWAYS_KEEP:
            return
        if name.startswith("__") and name.endswith("__"):
            return  # dunders
        if is_module_level and not self.rename_globals:
            self.preserve.add(name)
            return
        self.defined.add(name)

    def _visit_scope(self, node, names):
        is_mod = self._depth == 0
        for n in names:
            self._consider(n, is_mod)
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_FunctionDef(self, node):
        self._handle_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._handle_func(node)

    def _handle_func(self, node, method=False):
        is_mod = self._depth == 0
        if method:
            # Los métodos se invocan como atributos (obj.metodo): preservar
            # el nombre. Aun así renombramos sus parámetros y locales.
            self.preserve.add(node.name)
        else:
            self._consider(node.name, is_mod)
        a = node.args
        argnames = [x.arg for x in (a.posonlyargs + a.args + a.kwonlyargs)]
        if a.vararg:
            argnames.append(a.vararg.arg)
        if a.kwarg:
            argnames.append(a.kwarg.arg)
        for n in argnames:
            self._consider(n, False)  # los args siempre son locales
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_ClassDef(self, node):
        self._consider(node.name, self._depth == 0)
        self._depth += 1
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._handle_func(child, method=True)
            elif isinstance(child, ast.ClassDef):
                self.preserve.add(child.name)  # clase anidada = atributo
                self.visit(child)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                # Atributos de clase: accesibles por atributo → preservar.
                targets = child.targets if isinstance(child, ast.Assign) \
                    else [child.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.preserve.add(t.id)
                if child.value is not None:
                    self.visit(child.value)  # escanear strings/imports
            else:
                self.visit(child)
        self._depth -= 1

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self._consider(node.id, self._depth == 0)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Captura __all__ para preservar la API pública.
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "__all__":
                vals = getattr(node.value, "elts", [])
                for v in vals:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        self.preserve.add(v.value)
        self.generic_visit(node)


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]):
        self.m = mapping

    def visit_Name(self, node):
        if node.id in self.m:
            node.id = self.m[node.id]
        return node

    def _rename_func(self, node):
        if node.name in self.m:
            node.name = self.m[node.name]
        a = node.args
        for x in a.posonlyargs + a.args + a.kwonlyargs:
            if x.arg in self.m:
                x.arg = self.m[x.arg]
        if a.vararg and a.vararg.arg in self.m:
            a.vararg.arg = self.m[a.vararg.arg]
        if a.kwarg and a.kwarg.arg in self.m:
            a.kwarg.arg = self.m[a.kwarg.arg]
        self.generic_visit(node)
        return node

    visit_FunctionDef = _rename_func
    visit_AsyncFunctionDef = _rename_func

    def visit_ClassDef(self, node):
        if node.name in self.m:
            node.name = self.m[node.name]
        self.generic_visit(node)
        return node

    def visit_Global(self, node):
        node.names = [self.m.get(n, n) for n in node.names]
        return node

    def visit_Nonlocal(self, node):
        node.names = [self.m.get(n, n) for n in node.names]
        return node


class _StringObf(ast.NodeTransformer):
    """Reemplaza literales str por _s('<base64>')."""

    def __init__(self, helper: str, skip_docstrings: bool):
        self.helper = helper
        self.skip_docstrings = skip_docstrings
        self._docstrings: set[int] = set()

    def mark_docstrings(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    self._docstrings.add(id(body[0].value))

    def visit_JoinedStr(self, node):
        # No tocar el interior de f-strings.
        return node

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return node
        if self.skip_docstrings and id(node) in self._docstrings:
            return node
        enc = base64.b64encode(node.value.encode("utf-8")).decode("ascii")
        call = ast.Call(
            func=ast.Name(id=self.helper, ctx=ast.Load()),
            args=[ast.Constant(value=enc)],
            keywords=[],
        )
        return ast.copy_location(call, node)


_HELPER_SRC = (
    "import base64 as {b}\n"
    "def {s}(x):\n"
    "    return {b}.b64decode(x).decode('utf-8')\n"
)


def obfuscate(source: str, *, rename=True, strings=True,
              rename_globals=False, keep=None, skip_docstrings=True) -> str:
    keep = set(keep or [])
    tree = ast.parse(source)

    mapping: dict[str, str] = {}
    if rename:
        col = _Collector(rename_globals)
        col.visit(tree)
        candidates = col.defined - col.preserve - keep
        factory = _make_name_factory()
        used = set(col.preserve) | keep | _BUILTINS
        for name in sorted(candidates):
            new = factory()
            while new in used:
                new = factory()
            mapping[name] = new
            used.add(new)
        _Renamer(mapping).visit(tree)

    helper_names = {}
    if strings:
        # Nombres del helper que no colisionen con el mapping ni builtins.
        factory = _make_name_factory()
        taken = set(mapping.values()) | _BUILTINS | set(keep)
        for key in ("s", "b"):
            n = factory()
            while n in taken:
                n = factory()
            helper_names[key] = n
            taken.add(n)
        obf = _StringObf(helper_names["s"], skip_docstrings)
        obf.mark_docstrings(tree)
        obf.visit(tree)

    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)

    if strings:
        header = _HELPER_SRC.format(s=helper_names["s"], b=helper_names["b"])
        out = header + "\n" + out
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Ofuscador de código Python (renombrado + cadenas).")
    p.add_argument("input", help="archivo .py de entrada")
    p.add_argument("-o", "--output",
                   help="archivo de salida (por defecto: stdout)")
    p.add_argument("--no-rename", action="store_true",
                   help="no renombrar identificadores")
    p.add_argument("--no-strings", action="store_true",
                   help="no ofuscar cadenas")
    p.add_argument("--rename-globals", action="store_true",
                   help="renombrar también nombres de nivel de módulo "
                        "(¡rompe la API pública del módulo!)")
    p.add_argument("--keep", default="",
                   help="lista separada por comas de nombres a preservar")
    p.add_argument("--keep-docstrings", action="store_true",
                   help="no ofuscar los docstrings")
    args = p.parse_args(argv)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        p.error(f"no se pudo leer {args.input}: {e}")

    result = obfuscate(
        src,
        rename=not args.no_rename,
        strings=not args.no_strings,
        rename_globals=args.rename_globals,
        keep=[k.strip() for k in args.keep.split(",") if k.strip()],
        skip_docstrings=not args.keep_docstrings,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Escrito {len(result)} bytes en {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
