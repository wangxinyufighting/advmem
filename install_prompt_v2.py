"""将英文prompt适配到已有工程；默认只预览，--apply才写入。只用标准库。

python install_prompt_v2.py --project /path/to/full_memory_lab
python install_prompt_v2.py --project /path/to/full_memory_lab --apply
python install_prompt_v2.py --project /path/to/full_memory_lab --with-gate --apply
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

GENERATOR = '''    def generate(self, pack, qtype, date, n_questions=4, nonce=""):
        """使用英文v2出题；返回协议不变，不读取官方问题或答案。"""
        from memory_prompts import generate_questions
        return generate_questions(
            self.model, self.full, pack, qtype, date, n_questions,
            marks=getattr(self, "marks", None), nonce=nonce,
            audit_context=getattr(self, "audit_context", None),
        )
'''


def find_function(tree, name, class_name=None):
    roots = tree.body
    if class_name:
        classes = [n for n in roots if isinstance(n, ast.ClassDef) and n.name == class_name]
        if len(classes) != 1:
            raise ValueError(f"Expected one {class_name} class")
        roots = classes[0].body
    found = [n for n in roots if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(found) != 1 or isinstance(found[0], ast.AsyncFunctionDef):
        raise ValueError(f"Expected one synchronous {class_name or ''}.{name} function")
    return found[0]


def replace_expressions(source, replacements):
    """ast列偏移以UTF-8字节计；不以字符偏移误切含中文的源码。"""
    lines = source.encode('utf-8').splitlines(keepends=True)
    starts, total = [], 0
    for line in lines:
        starts.append(total)
        total += len(line)
    edits = []
    for node, value in replacements:
        begin = starts[node.lineno - 1] + node.col_offset
        end = starts[node.end_lineno - 1] + node.end_col_offset
        edits.append((begin, end, value.encode('utf-8')))
    edits.sort(reverse=True)
    blob = source.encode('utf-8')
    last = len(blob)
    for begin, end, value in edits:
        if end > last:
            raise ValueError("Overlapping AST edits")
        blob = blob[:begin] + value + blob[end:]
        last = begin
    return blob.decode('utf-8')


def patch_agents(source, with_gate=False):
    tree = ast.parse(source)
    generate = find_function(tree, "generate", "Attacker")
    names = [a.arg for a in generate.args.posonlyargs + generate.args.args]
    if names != ["self", "pack", "qtype", "date", "n_questions", "nonce"] or generate.args.vararg or generate.args.kwarg or generate.args.kwonlyargs:
        raise ValueError("Attacker.generate signature differs; inspect and integrate manually, no files changed")
    if generate.decorator_list:
        raise ValueError("Decorated generate() needs manual integration")
    # 检查已知的模型与原文属性，拒绝在未知接口上猜写。
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attacker")
    attrs = {n.attr for n in ast.walk(cls) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"}
    if not {"model", "full"} <= attrs:
        raise ValueError("Attacker does not expose self.model/self.full")
    lines = source.splitlines(keepends=True)
    source = ''.join(lines[:generate.lineno - 1]) + GENERATOR + ''.join(lines[generate.end_lineno:])
    if with_gate:
        tree = ast.parse(source)
        gate = find_function(tree, "gate")
        params = {a.arg for a in gate.args.args + gate.args.kwonlyargs}
        if not {"full", "date"} <= params:
            raise ValueError("gate requires known full/date arguments for this patch")
        assignments = {}
        for n in ast.walk(gate):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                if n.targets[0].id in {"screen", "decision", "support"}:
                    if n.targets[0].id in assignments:
                        raise ValueError("Ambiguous gate assignment")
                    assignments[n.targets[0].id] = n.value
        if set(assignments) != {"screen", "decision", "support"}:
            raise ValueError("gate layout differs; expected screen, decision, support assignments")
        edits = []
        for key, constant in [("screen", "GATE_SCREEN_SYSTEM"), ("decision", "GATE_ORACLE_SYSTEM")]:
            call = assignments[key]
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != "json" or len(call.args) < 2:
                raise ValueError(f"Unexpected gate {key} call")
            edits.append((call.args[0], constant))
        call = assignments["support"]
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError("Unexpected support comparison")
        if call.func.id == "grade":
            if len(call.args) != 5 or call.keywords:
                raise ValueError("Unexpected support grade signature")
            args = ', '.join(ast.get_source_segment(source, node) for node in call.args)
            edits.append((call, f'verify_support({args}, evidence=render_evidence(full, item["E"]), date=date)'))
        elif call.func.id != "verify_support":
            raise ValueError("Unrecognized support verifier; no automatic overwrite")
        for n in ast.walk(gate):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "render" and isinstance(n.func.value, ast.Name) and n.func.value.id == "full":
                if len(n.args) != 1 or n.keywords:
                    raise ValueError("Unexpected full.render usage in gate")
                edits.append((n, f'render_evidence(full, {ast.get_source_segment(source, n.args[0])})'))
        source = replace_expressions(source, edits)
        imported = "from memory_prompts import GATE_SCREEN_SYSTEM, GATE_ORACLE_SYSTEM, verify_support, render_evidence\n"
        if imported not in source:
            tree = ast.parse(source)
            end = 0
            for i, node in enumerate(tree.body):
                if (i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)) or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                    end = node.end_lineno
                else:
                    break
            lines = source.splitlines(keepends=True)
            source = ''.join(lines[:end]) + imported + ''.join(lines[end:])
    compile(source, "agents.py", "exec")
    return source


def patch_llm(source):
    """只翻译已知的JSON输出指令，不更改重试/代理/模型参数。"""
    method = find_function(ast.parse(source), "json", "Client")
    edits = []
    replacements = {"仅返回一个有效 JSON 对象。": "Return exactly one valid JSON object.",
                    "只返回一个有效 JSON 对象。": "Return exactly one valid JSON object."}
    for node in ast.walk(method):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            for old, new in replacements.items():
                value = value.replace(old, new)
            if value != node.value:
                edits.append((node, repr(value)))
    result = replace_expressions(source, edits)
    compile(result, "llm.py", "exec")
    return result, len(edits)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path('.'))
    p.add_argument("--with-gate", action="store_true", help="同时更换独立oracle/screen/support；保持reader判分grade不变")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args(argv)
    project = args.project.resolve()
    source_file = Path(__file__).with_name('memory_prompts.py')
    agents = project / 'agents.py'
    llm = project / 'llm.py'
    if not agents.is_file() or not llm.is_file():
        p.error("Project needs existing agents.py and llm.py; this package is not a standalone benchmark")
    try:
        planned_agents = patch_agents(agents.read_text(encoding='utf-8'), args.with_gate)
        planned_llm, count = patch_llm(llm.read_text(encoding='utf-8'))
    except (SyntaxError, ValueError) as exc:
        p.error(str(exc))
    plan = {agents: planned_agents, llm: planned_llm, project / 'memory_prompts.py': source_file.read_text(encoding='utf-8')}
    for path, text in plan.items():
        compile(text, str(path), 'exec')
        status = 'unchanged' if path.exists() and path.read_text(encoding='utf-8') == text else 'update'
        print(f'{status}: {path}')
    print(f'Known JSON instruction suffix replacements: {count}')
    if count == 0:
        print('No known Chinese suffix found; inspect Client.json if your version adds other instructions.')
    if not args.apply:
        print('Dry run only. Add --apply to write files after reviewing the plan.')
        return
    for path, text in plan.items():
        if path.exists() and path.read_text(encoding='utf-8') == text:
            continue
        if path.exists():
            raw = path.read_bytes()
            backup = path.with_name(path.name + '.pre_prompt_v2.' + hashlib.sha256(raw).hexdigest()[:12] + '.bak')
            if not backup.exists():
                backup.write_bytes(raw)
        temp = path.with_name(path.name + '.prompt_v2.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    print('Installed. Start a NEW output/cache namespace; do not resume a run made with old prompts.')


if __name__ == '__main__':
    main()
