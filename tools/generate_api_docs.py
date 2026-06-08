#!/usr/bin/env python3
"""AST-based API Reference Documentation Generator for Subvocal SDK.

Extracts docstrings and signatures from the python codebase without importing modules.
Outputs structured Docusaurus-compatible markdown reference files.
"""

import os
import ast
from typing import List, Dict, Any

SDK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sdk"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "docs", "api"))


def get_arg_signature(node: ast.arguments) -> str:
    """Reconstruct function argument signature from AST node."""
    args = []
    # Positional/Keyword arguments
    for i, arg in enumerate(node.args):
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        # Add default value if present
        default_idx = len(node.defaults) - len(node.args) + i
        if default_idx >= 0:
            arg_str += f" = {ast.unparse(node.defaults[default_idx])}"
        args.append(arg_str)
        
    # Vararg (*args)
    if node.vararg:
        arg_str = f"*{node.vararg.arg}"
        if node.vararg.annotation:
            arg_str += f": {ast.unparse(node.vararg.annotation)}"
        args.append(arg_str)
        
    # Keyword-only arguments
    for i, kwarg in enumerate(node.kwonlyargs):
        kw_str = kwarg.arg
        if kwarg.annotation:
            kw_str += f": {ast.unparse(kwarg.annotation)}"
        if node.kw_defaults and node.kw_defaults[i]:
            kw_str += f" = {ast.unparse(node.kw_defaults[i])}"
        args.append(kw_str)
        
    # Kwarg (**kwargs)
    if node.kwarg:
        arg_str = f"**{node.kwarg.arg}"
        if node.kwarg.annotation:
            arg_str += f": {ast.unparse(node.kwarg.annotation)}"
        args.append(arg_str)
        
    return ", ".join(args)


class ModuleParser(ast.NodeVisitor):
    """AST visitor to extract docstrings, classes, and methods."""
    
    def __init__(self, filepath: str, relative_path: str):
        self.filepath = filepath
        self.relative_path = relative_path
        self.module_doc = ""
        self.classes: List[Dict[str, Any]] = []
        self.functions: List[Dict[str, Any]] = []
        self.current_class: Dict[str, Any] = None
        
    def parse(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=self.filepath)
        
        # Extract module-level docstring
        self.module_doc = ast.get_docstring(tree) or ""
        self.visit(tree)
        
    def visit_ClassDef(self, node: ast.ClassDef):
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node) or "No description.",
            "bases": [ast.unparse(b) for b in node.bases],
            "methods": []
        }
        self.classes.append(class_info)
        
        # Visit children with current_class context active
        old_class = self.current_class
        self.current_class = class_info
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Ignore private helper methods
        if node.name.startswith("_") and node.name != "__init__":
            return
            
        args_sig = get_arg_signature(node.args)
        ret_annotation = ""
        if node.returns:
            ret_annotation = f" -> {ast.unparse(node.returns)}"
            
        method_info = {
            "name": node.name,
            "signature": f"def {node.name}({args_sig}){ret_annotation}",
            "docstring": ast.get_docstring(node) or "No description."
        }
        
        if self.current_class:
            self.current_class["methods"].append(method_info)
        else:
            self.functions.append(method_info)


def escape_braces(text: str) -> str:
    """Escapes curly braces in markdown text, except when inside code blocks or inline code."""
    parts = text.split("```")
    for i in range(len(parts)):
        if i % 2 == 0:  # Outside code blocks
            subparts = parts[i].split("`")
            for j in range(len(subparts)):
                if j % 2 == 0:  # Outside inline code
                    subparts[j] = subparts[j].replace("{", "\\{").replace("}", "\\}")
            parts[i] = "`".join(subparts)
    return "```".join(parts)


def format_docstring(doc: str) -> str:
    """Format docstrings nicely for Markdown."""
    if not doc:
        return ""
    
    # Escape braces to prevent MDX acorn parse errors
    doc = escape_braces(doc)
    
    lines = doc.split("\n")
    formatted_lines = []
    in_args = False
    in_returns = False
    
    for line in lines:
        stripped = line.strip()
        if stripped == "Args:" or stripped.startswith("Arguments:"):
            formatted_lines.append("\n**Arguments:**\n")
            in_args = True
            in_returns = False
            continue
        elif stripped == "Returns:":
            formatted_lines.append("\n**Returns:**\n")
            in_args = False
            in_returns = True
            continue
        elif stripped == "Raises:":
            formatted_lines.append("\n**Raises:**\n")
            in_args = False
            in_returns = False
            continue
        
        if in_args and stripped:
            # Format list of arguments
            if ":" in stripped:
                parts = stripped.split(":", 1)
                formatted_lines.append(f"- ` {parts[0].strip()} `: {parts[1].strip()}")
            else:
                formatted_lines.append(f"- {stripped}")
        elif in_returns and stripped:
            formatted_lines.append(f"- {stripped}")
        else:
            # Normal text lines
            formatted_lines.append(line)
            
    return "\n".join(formatted_lines)


def generate_module_markdown(parser: ModuleParser) -> str:
    """Converts parsed AST structure to a clean Docusaurus markdown file."""
    module_name = parser.relative_path.replace(".py", "").replace("/", ".").replace(".__init__", "")
    
    md = []
    md.append(f"---\ntitle: {module_name}\nsidebar_label: {module_name.split('.')[-1]}\n---\n")
    
    if parser.module_doc:
        md.append(f"{format_docstring(parser.module_doc)}\n")
        
    if parser.classes:
        md.append("## Classes\n")
        for cls in parser.classes:
            bases_str = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
            md.append(f"### `class {cls['name']}{bases_str}`\n")
            md.append(f"{format_docstring(cls['docstring'])}\n")
            
            if cls["methods"]:
                md.append("#### Methods\n")
                for method in cls["methods"]:
                    md.append(f"##### `{method['name']}`\n")
                    md.append(f"```python\n{method['signature']}\n```\n")
                    md.append(f"{format_docstring(method['docstring'])}\n")
                    
    if parser.functions:
        md.append("## Functions\n")
        for func in parser.functions:
            md.append(f"### `{func['name']}`\n")
            md.append(f"```python\n{func['signature']}\n```\n")
            md.append(f"{format_docstring(func['docstring'])}\n")
            
    return "\n".join(md)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating API references from {SDK_DIR} into {OUTPUT_DIR}...")
    
    # We target key packages
    target_packages = ["core", "context", "hardware", "emg_core", "mcp"]
    
    for pkg in target_packages:
        pkg_dir = os.path.join(SDK_DIR, pkg)
        if not os.path.isdir(pkg_dir):
            continue
            
        pkg_output_dir = os.path.join(OUTPUT_DIR, pkg)
        os.makedirs(pkg_output_dir, exist_ok=True)
        
        for root, _, files in os.walk(pkg_dir):
            for file in files:
                if not file.endswith(".py") or file.startswith("test_") or file == "eval_runner.py" or file == "eval_set.py" or file == "test_intent_runner.py":
                    continue
                    
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, SDK_DIR)
                
                print(f"Parsing {rel_path}...")
                parser = ModuleParser(full_path, rel_path)
                try:
                    parser.parse()
                    md_content = generate_module_markdown(parser)
                    
                    # Compute output file path
                    out_filename = file.replace(".py", ".md")
                    if out_filename == "__init__.md":
                        out_filename = "index.md"
                    out_filepath = os.path.join(pkg_output_dir, out_filename)
                    
                    with open(out_filepath, "w", encoding="utf-8") as out_f:
                        out_f.write(md_content)
                except Exception as e:
                    print(f"Error parsing {rel_path}: {e}")
                    
    # Generate sidebar grouping category index
    api_index = """---
title: API Reference Overview
sidebar_label: Overview
---

Welcome to the Subvocal SDK API Reference. This documentation is automatically generated directly from the source code docstrings.

Select a module from the sidebar to inspect classes, methods, parameters, and return types.
"""
    with open(os.path.join(OUTPUT_DIR, "index.md"), "w") as f:
        f.write(api_index)
        
    print("API Reference generation complete.")


if __name__ == "__main__":
    main()
